"""Registry + Axiom* compat layer (P0-06). CPU-only, no network: `from_pretrained`
is the process boundary and gets stubbed."""

import axiom_model
import pytest
from axiom_model import registry


def test_resolve_unknown_name_raises_with_the_registered_names():
    with pytest.raises(KeyError) as exc:
        registry.resolve("axiom-zero-enormous")

    assert "axiom-zero-small" in str(exc.value)


def test_default_device_is_cpu_when_no_gpu_is_present(monkeypatch):
    monkeypatch.setattr(registry.torch.cuda, "is_available", lambda: False)

    assert registry.default_device() == "cpu"


def test_default_device_is_cuda_when_a_gpu_is_present(monkeypatch):
    # ROCm also presents as "cuda" in torch — see CLAUDE.md.
    monkeypatch.setattr(registry.torch.cuda, "is_available", lambda: True)

    assert registry.default_device() == "cuda:0"


def test_load_predictor_passes_the_registered_sources_and_context(monkeypatch):
    loaded = {}

    def fake_from_pretrained(cls):
        def _load(source):
            loaded[cls] = source
            return f"<{cls}>"

        return _load

    monkeypatch.setattr(
        axiom_model.AxiomTokenizer, "from_pretrained", fake_from_pretrained("tokenizer")
    )
    monkeypatch.setattr(axiom_model.Axiom, "from_pretrained", fake_from_pretrained("model"))
    monkeypatch.setattr(registry, "AxiomPredictor", lambda *a, **kw: (a, kw))

    (model, tokenizer), kwargs = registry.load_predictor("axiom-zero-mini", device="cpu")

    assert loaded == {
        "model": "NeoQuasar/Kronos-mini",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
    }
    assert (model, tokenizer) == ("<model>", "<tokenizer>")
    assert kwargs == {"device": "cpu", "max_context": 2048}


def test_axiom_classes_stay_weight_compatible_with_upstream_kronos():
    # CLAUDE.md: Axiom* must keep loading NeoQuasar/Kronos-* checkpoints.
    from axiom_model._kronos import Kronos, KronosPredictor, KronosTokenizer

    assert issubclass(axiom_model.Axiom, Kronos)
    assert issubclass(axiom_model.AxiomTokenizer, KronosTokenizer)
    assert issubclass(axiom_model.AxiomPredictor, KronosPredictor)
    assert hasattr(axiom_model.Axiom, "from_pretrained")
