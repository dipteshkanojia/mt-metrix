"""Scorer plugins for mt-metrix.

Every scorer implements the :class:`Scorer` protocol from ``base`` and is
registered in ``registry`` so it can be picked up from a config's
``scorers: [ref: <family>/<name>]``.
"""

from mt_metrix.scorers.base import Scorer, ScorerConfig
from mt_metrix.scorers.registry import SCORER_REGISTRY, build_scorer, register_scorer

__all__ = ["Scorer", "ScorerConfig", "SCORER_REGISTRY", "build_scorer", "register_scorer"]
