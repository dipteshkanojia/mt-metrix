"""Tests for the output writers (TSV, JSONL, summary JSON + correlations)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.io.writers import (
    _soft_pairwise_accuracy,
    write_segments_jsonl,
    write_segments_tsv,
    write_summary,
)


@pytest.fixture
def small_segments():
    return [
        Segment(
            source="hello",
            target="bonjour",
            reference="salut",
            gold_raw=0.8,
            lang_pair="en-fr",
            domain="general",
            segment_id=f"s{i:02d}",
        )
        for i in range(5)
    ]


@pytest.fixture
def small_scores():
    return {
        "bleu": [SegmentScore(segment_id=f"s{i:02d}", score=0.5 + i * 0.1) for i in range(5)],
        "chrf": [
            SegmentScore(segment_id=f"s{i:02d}", score=0.6 + i * 0.05, extra={"n": i})
            for i in range(5)
        ],
    }


def test_write_tsv_round_trip(tmp_path: Path, small_segments, small_scores):
    import pandas as pd

    out = tmp_path / "segments.tsv"
    write_segments_tsv(out, small_segments, small_scores)
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == [
        "segment_id", "lang_pair", "domain", "source", "target",
        "reference", "gold", "bleu", "chrf",
    ]
    assert len(df) == 5
    assert df["bleu"].iloc[0] == pytest.approx(0.5)
    assert df["chrf"].iloc[4] == pytest.approx(0.8)


def test_write_jsonl_includes_extras(tmp_path: Path, small_segments, small_scores):
    out = tmp_path / "segments.jsonl"
    write_segments_jsonl(out, small_segments, small_scores)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[0]["segment_id"] == "s00"
    assert rows[0]["scores"]["bleu"]["score"] == pytest.approx(0.5)
    assert rows[2]["scores"]["chrf"]["extra"]["n"] == 2


def test_write_summary_shape(tmp_path: Path, small_segments, small_scores):
    out = tmp_path / "summary.json"
    write_summary(
        out,
        small_segments,
        small_scores,
        skipped_metrics=[{"name": "ter", "reason": "dataset-has-no-references"}],
        run_metadata={"run_id": "test"},
        corpus_scores={"bleu": {"corpus_score": 25.5}},
    )
    payload = json.loads(out.read_text())
    assert payload["n_segments"] == 5
    assert set(payload["metrics"]) == {"bleu", "chrf"}
    assert payload["metrics"]["bleu"]["n"] == 5
    assert payload["metrics"]["bleu"]["correlation_vs_gold"] is not None
    # bleu is perfectly monotonic with a constant gold → Spearman should be nan/None
    # (we picked increasing scores vs. constant gold so Pearson is nan)
    # but with non-constant gold we'd see a real correlation; this fixture has
    # gold=0.8 for all, so gold variance is 0 and correlations end up None/NaN
    # — the summary should tolerate that without crashing:
    assert isinstance(payload["metrics"]["bleu"]["correlation_vs_gold"], dict)
    assert payload["skipped_metrics"][0]["name"] == "ter"
    assert payload["corpus_scores"]["bleu"]["corpus_score"] == pytest.approx(25.5)
    assert payload["run_metadata"]["run_id"] == "test"


def test_write_summary_correlation_when_gold_varies(tmp_path: Path):
    segs = [
        Segment(source="a", target="b", gold_raw=g, segment_id=f"s{i}")
        for i, g in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    # bleu here is perfectly correlated with gold
    scores = {
        "bleu": [SegmentScore(segment_id=f"s{i}", score=g) for i, g in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]
    }
    out = tmp_path / "summary.json"
    write_summary(
        out,
        segs,
        scores,
        skipped_metrics=[],
        run_metadata={},
    )
    payload = json.loads(out.read_text())
    corr = payload["metrics"]["bleu"]["correlation_vs_gold"]
    assert corr["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert corr["spearman"] == pytest.approx(1.0, abs=1e-6)
    assert corr["kendall"] == pytest.approx(1.0, abs=1e-6)
    assert corr["spa"] == pytest.approx(1.0, abs=1e-6)
    assert corr["n"] == 5


def test_soft_pairwise_accuracy_perfect():
    # identical ranking → SPA = 1.0
    assert _soft_pairwise_accuracy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_soft_pairwise_accuracy_inverted():
    # inverted ranking → SPA = 0.0
    assert _soft_pairwise_accuracy([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_soft_pairwise_accuracy_ties_get_half_weight():
    # All gold tied → every pair counts as 0.5 × (1 - match).
    # With preds varying and gold all equal, matches = 0, so score = 0.5 for each
    pred = [0.1, 0.2, 0.3]
    gold = [1.0, 1.0, 1.0]
    assert _soft_pairwise_accuracy(pred, gold) == pytest.approx(0.5)
