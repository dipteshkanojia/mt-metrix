"""Tests for the sacrebleu scorer — the one family we can run without a GPU."""
from __future__ import annotations

import pytest

pytest.importorskip("sacrebleu")

from mt_metrix.io.schema import Segment
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.sacrebleu_scorer import SacreBleuScorer


def _mk_segs() -> list[Segment]:
    # Simple, language-agnostic test material — avoids tokenizer gotchas.
    return [
        Segment(source="a b c", target="the cat sat", reference="the cat sat",
                segment_id="s1", lang_pair="en-en"),
        Segment(source="d e f", target="dogs run fast", reference="dogs run quick",
                segment_id="s2", lang_pair="en-en"),
        Segment(source="g h i", target="the sky is blue", reference="the sky is blue",
                segment_id="s3", lang_pair="en-en"),
    ]


def test_sacrebleu_bleu_scores_segments():
    cfg = ScorerConfig(family="sacrebleu", name="bleu",
                       params={"metric": "bleu", "tokenize": "13a"})
    scorer = SacreBleuScorer(cfg)
    scorer.load()
    scores = scorer.score(_mk_segs())
    assert len(scores) == 3
    # Exact matches → BLEU should be 100 on s1 and s3
    assert scores[0].score == pytest.approx(100.0, rel=1e-3)
    assert scores[2].score == pytest.approx(100.0, rel=1e-3)
    # Near-match on s2 → BLEU < 100
    assert scores[1].score < 100.0
    # corpus_score populated after score()
    assert scorer.corpus_score is not None
    assert scorer.corpus_score["metric"] == "bleu"


def test_sacrebleu_chrf_runs():
    cfg = ScorerConfig(family="sacrebleu", name="chrf",
                       params={"metric": "chrf", "chrf_word_order": 0})
    scorer = SacreBleuScorer(cfg)
    scorer.load()
    scores = scorer.score(_mk_segs())
    assert len(scores) == 3
    assert all(0.0 <= s.score <= 100.0 for s in scores)
    assert scorer.corpus_score["metric"] == "chrf"


def test_sacrebleu_chrfpp_runs():
    cfg = ScorerConfig(family="sacrebleu", name="chrf++",
                       params={"metric": "chrf", "chrf_word_order": 2})
    scorer = SacreBleuScorer(cfg)
    scorer.load()
    scores = scorer.score(_mk_segs())
    assert len(scores) == 3
    assert all(0.0 <= s.score <= 100.0 for s in scores)


def test_sacrebleu_ter_runs():
    cfg = ScorerConfig(family="sacrebleu", name="ter", params={"metric": "ter"})
    scorer = SacreBleuScorer(cfg)
    scorer.load()
    scores = scorer.score(_mk_segs())
    assert len(scores) == 3
    # Exact matches → TER = 0
    assert scores[0].score == pytest.approx(0.0, abs=1e-6)
    # Non-identical → TER > 0
    assert scores[1].score > 0.0


def test_sacrebleu_needs_reference():
    cfg = ScorerConfig(family="sacrebleu", name="bleu", params={"metric": "bleu"})
    scorer = SacreBleuScorer(cfg)
    assert scorer.needs_reference is True


def test_sacrebleu_unknown_metric_raises():
    cfg = ScorerConfig(family="sacrebleu", name="bogus", params={"metric": "bogus"})
    with pytest.raises(ValueError, match="metric'"):
        SacreBleuScorer(cfg)
