"""Tests for the unified Segment / SegmentScore schema."""
from __future__ import annotations

from mt_metrix.io.schema import Segment, SegmentScore, PredictionSpace


def test_segment_has_reference_true_when_set():
    seg = Segment(source="hi", target="salut", reference="coucou")
    assert seg.has_reference() is True


def test_segment_has_reference_false_when_none():
    seg = Segment(source="hi", target="salut")
    assert seg.has_reference() is False


def test_segment_has_reference_false_when_empty_string():
    seg = Segment(source="hi", target="salut", reference="")
    assert seg.has_reference() is False


def test_segment_has_gold():
    assert Segment(source="a", target="b", gold_z=0.5).has_gold() is True
    assert Segment(source="a", target="b").has_gold() is False
    # gold of 0.0 is a valid score — should still register as present
    assert Segment(source="a", target="b", gold_raw=0.0).has_gold() is True


def test_segment_defaults():
    seg = Segment(source="a", target="b")
    assert seg.reference is None
    assert seg.gold is None
    assert seg.lang_pair == ""
    assert seg.domain == "general"
    assert seg.segment_id == ""
    assert seg.meta == {}


def test_segment_score_extra_is_independent_per_instance():
    a = SegmentScore(segment_id="s1", score=0.5)
    b = SegmentScore(segment_id="s2", score=0.7)
    a.extra["flag"] = True
    assert "flag" not in b.extra


def test_segment_carries_both_gold_columns():
    s = Segment(source="hello", target="hola", gold_raw=72.5, gold_z=0.8)
    assert s.gold_raw == 72.5
    assert s.gold_z == 0.8


def test_segment_gold_property_prefers_z():
    """Backward compat: `seg.gold` returns gold_z if set, else gold_raw.

    Z-scored gold is the historic default for QE datasets, so existing
    callers that read `seg.gold` keep getting the same signal.
    """
    s_both = Segment(source="a", target="b", gold_raw=70.0, gold_z=0.5)
    assert s_both.gold == 0.5

    s_raw_only = Segment(source="a", target="b", gold_raw=70.0)
    assert s_raw_only.gold == 70.0

    s_z_only = Segment(source="a", target="b", gold_z=0.5)
    assert s_z_only.gold == 0.5

    s_neither = Segment(source="a", target="b")
    assert s_neither.gold is None


def test_segment_has_gold_still_works():
    assert not Segment(source="a", target="b").has_gold()
    assert Segment(source="a", target="b", gold_raw=1.0).has_gold()
    assert Segment(source="a", target="b", gold_z=1.0).has_gold()


def test_prediction_space_literal_values():
    """The PredictionSpace alias must accept the two values the rest of the
    code routes on. Anything else is a typo the type checker should catch."""
    assert PredictionSpace.__args__ == ("raw_da", "z_da")
