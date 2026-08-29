"""Cached autoregressive generation (P4-04).

Upstream re-runs a full forward pass over the whole window for every generated bar:
24 steps x 512 tokens of work to produce 24 tokens. This computes the same
arithmetic with a per-layer KV cache — one prefill over the context, then one token
per step.

**Why this is exact rather than merely close.** RoPE is applied at positions
0..len-1 of whatever slice reaches an attention layer, so upstream's *sliding*
window — which starts as soon as `context + pred_len > max_context` — re-rotates
every token on every step, and no cache can reproduce that. When the window does not
slide, positions are stable: rotating each token once at its absolute position gives
the same keys and values the recompute would. So `cached_inference` refuses the
sliding case instead of quietly returning different numbers; feed it a context short
enough that `context + pred_len <= max_context`, or pass `use_cache=False`.

`tests/test_parity.py` is the standing proof: same seed, token-identical output
against `_kronos.auto_regressive_inference`, plus MC moments over many paths.
"""

from __future__ import annotations

import numpy as np
import torch

from ._kronos import sample_from_logits


def _expand(t: torch.Tensor, sample_count: int) -> torch.Tensor:
    """(batch, len, d) -> (batch * sample_count, len, d), upstream's MC layout."""
    return (
        t.unsqueeze(1)
        .repeat(1, sample_count, 1, 1)
        .reshape(-1, t.size(1), t.size(2))
        .to(t.device)
    )


def _step(model, tokens_pre, tokens_post, stamp, caches, offset, hidden_so_far, T, top_k, top_p):
    """One decode step: s1 from the cached window, then s2 conditioned on it."""
    s1_logits, hidden = model.decode_s1(
        tokens_pre, tokens_post, stamp, caches=caches, offset=offset
    )
    sample_pre = sample_from_logits(
        s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
    )
    hidden_so_far = torch.cat([hidden_so_far, hidden], dim=1)
    # Upstream feeds the whole window and keeps the last row; the last row only ever
    # needed its own hidden state as the residual and the window as keys/values.
    s2_logits = model.decode_s2(hidden[:, -1:, :], sample_pre, kv_states=hidden_so_far)
    sample_post = sample_from_logits(
        s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p, sample_logits=True
    )
    return sample_pre, sample_post, hidden_so_far


def cached_inference(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
    reduce="mean",
):
    """Same contract as `_kronos.auto_regressive_inference`, with a KV cache.

    Raises when the window would slide — never silently falls back, because a silent
    fallback is how an 8x speedup becomes a number nobody can explain.
    """
    context_len = x.size(1)
    if context_len + pred_len > max_context:
        raise ValueError(
            f"cached generation needs context + pred_len <= max_context "
            f"({context_len} + {pred_len} > {max_context}); shorten the context, or "
            f"pass use_cache=False for upstream's sliding window"
        )

    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        x = _expand(x, sample_count)
        x_stamp = _expand(x_stamp, sample_count)
        y_stamp = _expand(y_stamp, sample_count)

        pre, post = tokenizer.encode(x, half=True)
        batch = pre.size(0)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        caches = [[] for _ in model.transformer]

        generated_pre = pre.new_empty(batch, pred_len)
        generated_post = post.new_empty(batch, pred_len)
        hidden = torch.zeros(batch, 0, model.d_model, device=x.device, dtype=x.dtype)

        steps = range(pred_len)
        if verbose:
            from tqdm import trange

            steps = trange(pred_len)
        for i in steps:
            if i == 0:  # prefill: the whole context in one pass, filling every cache
                tokens_pre, tokens_post = pre, post
                stamp, offset = full_stamp[:, :context_len, :], 0
            else:
                offset = context_len + i - 1
                tokens_pre = generated_pre[:, i - 1 : i]
                tokens_post = generated_post[:, i - 1 : i]
                stamp = full_stamp[:, offset : offset + 1, :]

            sample_pre, sample_post, hidden = _step(
                model, tokens_pre, tokens_post, stamp, caches, offset, hidden, T, top_k, top_p
            )
            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

        z = tokenizer.decode(
            [torch.cat([pre, generated_pre], dim=1), torch.cat([post, generated_post], dim=1)],
            half=True,
        )
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))

        preds = z.cpu().numpy()
        if reduce == "mean":
            preds = np.mean(preds, axis=1)
        return preds


__all__ = ["cached_inference"]
