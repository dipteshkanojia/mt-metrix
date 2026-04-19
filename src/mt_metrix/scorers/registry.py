"""Plugin registry for scorer families.

A scorer family (``"comet"``, ``"tower"``, ``"sacrebleu"``, …) registers a
factory callable ``(ScorerConfig) -> Scorer``. The runner calls
``build_scorer(cfg)`` which dispatches to the right factory by
``cfg.family``.

Adding a new family:

1. Implement a class conforming to :class:`Scorer` in a new module under
   ``src/mt_metrix/scorers/``.
2. In the same module, call ``register_scorer("myfamily", MyScorer.from_config)``.
3. Import the module from ``_bootstrap`` below so it's loaded at package init.
"""
from __future__ import annotations

from typing import Callable

from mt_metrix.scorers.base import Scorer, ScorerConfig

SCORER_REGISTRY: dict[str, Callable[[ScorerConfig], Scorer]] = {}


def register_scorer(family: str, factory: Callable[[ScorerConfig], Scorer]) -> None:
    """Register a factory under a family name. Idempotent: re-registering wins."""
    SCORER_REGISTRY[family] = factory


def build_scorer(cfg: ScorerConfig) -> Scorer:
    """Instantiate a scorer for the given config, dispatching by family."""
    if cfg.family not in SCORER_REGISTRY:
        families = ", ".join(sorted(SCORER_REGISTRY)) or "<none registered yet>"
        raise KeyError(
            f"Unknown scorer family {cfg.family!r}. Registered families: {families}"
        )
    return SCORER_REGISTRY[cfg.family](cfg)


def _bootstrap() -> None:
    """Import scorer modules so their ``register_scorer`` calls fire.

    Isolated in a function so test code can call it idempotently. Import
    errors are swallowed per-family (e.g. ``unbabel-comet`` not installed)
    so the registry still exposes whatever families *are* available.
    """
    import importlib
    import logging

    log = logging.getLogger(__name__)
    for module_name in (
        "mt_metrix.scorers.comet",
        "mt_metrix.scorers.tower",
        "mt_metrix.scorers.sacrebleu_scorer",
    ):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            log.debug("skipping %s (%s)", module_name, e)
