"""Tests for the scorer registry."""
from __future__ import annotations

import pytest

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import (
    SCORER_REGISTRY,
    _bootstrap,
    build_scorer,
    register_scorer,
)


class _FakeScorer:
    def __init__(self, cfg: ScorerConfig) -> None:
        self._cfg = cfg

    @property
    def config(self): return self._cfg
    @property
    def name(self): return self._cfg.name
    @property
    def family(self): return "fake"
    @property
    def needs_reference(self): return False

    def load(self): ...
    def score(self, segments): return [SegmentScore(segment_id=s.segment_id, score=1.0) for s in segments]
    def unload(self): ...


def test_register_and_build_round_trip():
    register_scorer("fake", _FakeScorer)
    cfg = ScorerConfig(family="fake", name="fake-1", params={})
    scorer = build_scorer(cfg)
    assert scorer.family == "fake"
    segs = [Segment(source="a", target="b", segment_id="s1")]
    scorer.load()
    assert scorer.score(segs)[0].score == 1.0
    scorer.unload()


def test_build_unknown_family_raises():
    with pytest.raises(KeyError, match="Unknown scorer family"):
        build_scorer(ScorerConfig(family="nonsense-42", name="x", params={}))


def test_bootstrap_registers_sacrebleu():
    """sacrebleu is a pure-python dep so bootstrap should always register it."""
    # clear and re-bootstrap to confirm
    _bootstrap()
    assert "sacrebleu" in SCORER_REGISTRY
