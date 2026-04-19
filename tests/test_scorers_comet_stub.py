"""Slow integration test for the COMET scorer.

Downloads a small public (non-gated) COMET checkpoint and scores the tiny
fixture. Opt-in via ``MT_METRIX_RUN_SLOW=1``.

Uses ``eamt22-cometinho-da`` (distilled, ~60 MB) so this stays under a
minute on CPU.
"""
from __future__ import annotations

import pytest

pytest.importorskip("comet")

from mt_metrix.io.schema import Segment
from mt_metrix.scorers.base import ScorerConfig


pytestmark = pytest.mark.slow


def test_comet_scorer_end_to_end():
    from mt_metrix.scorers.comet import CometScorer

    cfg = ScorerConfig(
        family="comet",
        name="cometinho",
        model="Unbabel/eamt22-cometinho-da",
        params={"batch_size": 8, "gpus": 0, "num_workers": 0, "progress_bar": False},
    )
    scorer = CometScorer(cfg)
    scorer.load()
    try:
        segments = [
            Segment(
                source="Hello world.",
                target="Bonjour le monde.",
                reference="Bonjour le monde.",
                segment_id="s1",
                lang_pair="en-fr",
            ),
            Segment(
                source="How are you?",
                target="Comment allez-vous?",
                reference="Comment vas-tu?",
                segment_id="s2",
                lang_pair="en-fr",
            ),
        ]
        scores = scorer.score(segments)
    finally:
        scorer.unload()

    assert len(scores) == 2
    for s in scores:
        assert -1.0 <= s.score <= 2.0  # COMET scores are typically in [-0.5, 1.5]
