"""Tests for the unified Segment / SegmentScore schema."""
from __future__ import annotations

from mt_metrix.io.schema import Segment, SegmentScore


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
    assert Segment(source="a", target="b", gold=0.5).has_gold() is True
    assert Segment(source="a", target="b", gold=None).has_gold() is False
    # gold of 0.0 is a valid score — should still register as present
    assert Segment(source="a", target="b", gold=0.0).has_gold() is True


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
