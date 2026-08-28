"""Workspace smoke test: all packages import."""


def test_workspace_imports():
    import axiom_data  # noqa: F401
    import axiom_eval  # noqa: F401
    import axiom_model  # noqa: F401
    import axiom_signals  # noqa: F401
    import axiom_trader  # noqa: F401
