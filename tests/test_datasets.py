"""Tests for the dataset loaders — column mapping + local TSV/CSV/JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mt_metrix.config import DatasetConfig
from mt_metrix.io.datasets import _resolve_column, load_dataset_from_config


# ---------------------------------------------------------------------------
# _resolve_column — column directives
# ---------------------------------------------------------------------------

def test_resolve_column_bare_name():
    row = {"src": "hello", "tgt": "bonjour"}
    assert _resolve_column(row, "src") == "hello"


def test_resolve_column_constant_directive():
    row = {"src": "hello"}
    assert _resolve_column(row, "@constant:legal") == "legal"


def test_resolve_column_from_directive():
    row = {"lp": "en-fr", "tgt": "salut"}
    assert _resolve_column(row, "@from:lp") == "en-fr"


def test_resolve_column_none_spec():
    assert _resolve_column({"a": 1}, None) is None


def test_resolve_column_missing_name_returns_none():
    assert _resolve_column({"a": 1}, "b") is None


# ---------------------------------------------------------------------------
# Local TSV loader
# ---------------------------------------------------------------------------

def test_load_local_tsv_with_reference(fixtures_dir: Path):
    cfg = DatasetConfig(
        kind="local",
        params={
            "path": str(fixtures_dir / "tiny_with_ref.tsv"),
            "lang_pair": "en-fr",
            "domain": "general",
        },
        columns={
            "source": "source",
            "target": "target",
            "reference": "reference",
            "gold": "gold",
            "segment_id": "segment_id",
            "lang_pair": "lang_pair",
            "domain": "domain",
        },
    )
    segs = load_dataset_from_config(cfg)
    assert len(segs) == 10
    assert segs[0].segment_id == "s001"
    assert segs[0].source.startswith("The cat")
    assert segs[0].has_reference() is True
    assert segs[0].gold == pytest.approx(0.82)
    assert segs[0].lang_pair == "en-fr"
    assert segs[0].domain == "general"


def test_load_local_tsv_no_reference(fixtures_dir: Path):
    cfg = DatasetConfig(
        kind="local",
        params={
            "path": str(fixtures_dir / "tiny_no_ref.tsv"),
            "lang_pair": "en-fr",
            "domain": "general",
        },
        columns={
            "source": "source",
            "target": "target",
            "gold": "gold",
            "segment_id": "segment_id",
            "lang_pair": "lang_pair",
            "domain": "domain",
        },
    )
    segs = load_dataset_from_config(cfg)
    assert len(segs) == 10
    assert all(not s.has_reference() for s in segs)
    assert all(s.has_gold() for s in segs)


def test_load_local_tsv_limit(fixtures_dir: Path):
    cfg = DatasetConfig(
        kind="local",
        params={
            "path": str(fixtures_dir / "tiny_with_ref.tsv"),
            "limit": 3,
            "lang_pair": "en-fr",
        },
        columns={"source": "source", "target": "target", "segment_id": "segment_id"},
    )
    segs = load_dataset_from_config(cfg)
    assert len(segs) == 3


def test_load_local_constant_directive(fixtures_dir: Path):
    cfg = DatasetConfig(
        kind="local",
        params={"path": str(fixtures_dir / "tiny_no_ref.tsv")},
        columns={
            "source": "source",
            "target": "target",
            "segment_id": "segment_id",
            "domain": "@constant:legal",
            "lang_pair": "@from:lang_pair",
        },
    )
    segs = load_dataset_from_config(cfg)
    assert all(s.domain == "legal" for s in segs)
    assert all(s.lang_pair == "en-fr" for s in segs)


def test_load_local_missing_file_raises(tmp_path: Path):
    cfg = DatasetConfig(
        kind="local",
        params={"path": str(tmp_path / "nope.tsv")},
    )
    with pytest.raises(FileNotFoundError):
        load_dataset_from_config(cfg)


def test_load_local_jsonl(tmp_path: Path):
    p = tmp_path / "mini.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"src": "a", "tgt": "b", "id": "1"},
                {"src": "c", "tgt": "d", "id": "2"},
            ]
        ),
        encoding="utf-8",
    )
    cfg = DatasetConfig(
        kind="local",
        params={"path": str(p), "lang_pair": "en-fr", "domain": "general"},
        columns={"source": "src", "target": "tgt", "segment_id": "id"},
    )
    segs = load_dataset_from_config(cfg)
    assert len(segs) == 2
    assert segs[0].source == "a"
    assert segs[1].segment_id == "2"


def test_unknown_kind_raises():
    cfg = DatasetConfig(kind="fake-kind", params={})
    with pytest.raises(ValueError, match="unknown dataset kind"):
        load_dataset_from_config(cfg)


def test_missing_source_or_target_raises(tmp_path: Path):
    # Point `target` at a column name that doesn't exist → _resolve_column
    # returns None → _row_to_segment raises.
    p = tmp_path / "bad.tsv"
    p.write_text("source\tother\nhello\tthere\n", encoding="utf-8")
    cfg = DatasetConfig(
        kind="local",
        params={"path": str(p)},
        columns={"source": "source", "target": "nope"},
    )
    with pytest.raises(ValueError, match="missing source or target"):
        load_dataset_from_config(cfg)


# ---------------------------------------------------------------------------
# HuggingFace multi-subset loader
# ---------------------------------------------------------------------------

class _FakeHFDataset:
    """Minimal stand-in for a ``datasets.Dataset`` — just iterable dicts."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return _FakeHFDataset([self._rows[i] for i in indices])


def _patch_load_dataset(monkeypatch, subsets: dict[str, list[dict]]):
    """Patch ``datasets.load_dataset`` to return rows keyed by subset name."""
    import datasets as hf_datasets

    def fake_load(repo, *, split=None, name=None, cache_dir=None, **_kw):
        if name not in subsets:
            raise ValueError(f"unknown fake subset {name!r}; have {list(subsets)}")
        return _FakeHFDataset(list(subsets[name]))

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load)


def test_load_hf_single_subset(monkeypatch):
    _patch_load_dataset(monkeypatch, {
        "en-gujarati": [
            {"source_text": "hello", "target_text": "namaste",
             "z_mean": 0.8, "language_pair": "en-gu"},
            {"source_text": "world", "target_text": "duniya",
             "z_mean": 0.5, "language_pair": "en-gu"},
        ],
    })
    cfg = DatasetConfig(
        kind="huggingface",
        params={"repo": "fake/Legal-QE", "config": "en-gujarati", "split": "test",
                "domain": "legal"},
        columns={
            "source": "source_text",
            "target": "target_text",
            "gold": "z_mean",
            "lang_pair": "@from:language_pair",
            "domain": "@constant:legal",
        },
    )
    segs = load_dataset_from_config(cfg)
    assert len(segs) == 2
    assert segs[0].lang_pair == "en-gu"
    assert segs[0].domain == "legal"
    assert segs[1].source == "world"


def test_load_hf_multi_subset_concatenates(monkeypatch):
    _patch_load_dataset(monkeypatch, {
        "en-gujarati": [
            {"source_text": "a", "target_text": "A", "z_mean": 0.9,
             "language_pair": "en-gu"},
        ],
        "en-tamil": [
            {"source_text": "b", "target_text": "B", "z_mean": 0.7,
             "language_pair": "en-ta"},
            {"source_text": "c", "target_text": "C", "z_mean": 0.3,
             "language_pair": "en-ta"},
        ],
        "en-telugu": [
            {"source_text": "d", "target_text": "D", "z_mean": 0.6,
             "language_pair": "en-te"},
        ],
    })
    cfg = DatasetConfig(
        kind="huggingface",
        params={
            "repo": "fake/Legal-QE",
            "configs": ["en-gujarati", "en-tamil", "en-telugu"],
            "split": "test",
            "domain": "legal",
        },
        columns={
            "source": "source_text",
            "target": "target_text",
            "gold": "z_mean",
            "lang_pair": "@from:language_pair",
            "domain": "@constant:legal",
        },
    )
    segs = load_dataset_from_config(cfg)
    # 1 + 2 + 1 = 4 segments, in the order the subsets were listed
    assert [s.lang_pair for s in segs] == ["en-gu", "en-ta", "en-ta", "en-te"]
    assert all(s.domain == "legal" for s in segs)
    # Segment IDs are unique across subsets (global counter, not per-subset)
    assert len({s.segment_id for s in segs}) == 4


def test_load_hf_multi_subset_limit_caps_total(monkeypatch):
    _patch_load_dataset(monkeypatch, {
        "a": [{"source_text": f"s{i}", "target_text": f"t{i}", "z_mean": 0.1,
               "language_pair": "en-a"} for i in range(5)],
        "b": [{"source_text": f"s{i}", "target_text": f"t{i}", "z_mean": 0.1,
               "language_pair": "en-b"} for i in range(5)],
    })
    cfg = DatasetConfig(
        kind="huggingface",
        params={"repo": "fake/X", "configs": ["a", "b"], "split": "test", "limit": 7},
        columns={
            "source": "source_text", "target": "target_text", "gold": "z_mean",
            "lang_pair": "@from:language_pair", "domain": "@constant:test",
        },
    )
    segs = load_dataset_from_config(cfg)
    # limit applies to total across subsets: all 5 of "a" then 2 of "b"
    assert len(segs) == 7
    assert [s.lang_pair for s in segs] == ["en-a"] * 5 + ["en-b"] * 2


def test_load_hf_rejects_both_config_and_configs(monkeypatch):
    _patch_load_dataset(monkeypatch, {"x": []})
    cfg = DatasetConfig(
        kind="huggingface",
        params={"repo": "fake/X", "config": "x", "configs": ["x"]},
        columns={"source": "s", "target": "t"},
    )
    with pytest.raises(ValueError, match="cannot set both"):
        load_dataset_from_config(cfg)


def test_load_hf_rejects_non_list_configs(monkeypatch):
    _patch_load_dataset(monkeypatch, {"x": []})
    cfg = DatasetConfig(
        kind="huggingface",
        params={"repo": "fake/X", "configs": "not-a-list"},
        columns={"source": "s", "target": "t"},
    )
    with pytest.raises(ValueError, match="must be a list"):
        load_dataset_from_config(cfg)


# ---------------------------------------------------------------------------
# _row_to_segment — gold_raw / gold_z / legacy gold
# ---------------------------------------------------------------------------

import logging

from mt_metrix.io.datasets import _row_to_segment


def test_row_to_segment_both_gold_columns():
    row = {"src": "hi", "tgt": "hola", "raw": "72.5", "z": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 72.5
    assert seg.gold_z == 0.8


def test_row_to_segment_only_gold_raw():
    row = {"src": "hi", "tgt": "hola", "raw": "72.5"}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 72.5
    assert seg.gold_z is None


def test_row_to_segment_only_gold_z():
    row = {"src": "hi", "tgt": "hola", "z": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw is None
    assert seg.gold_z == 0.8


def test_row_to_segment_legacy_gold_loads_into_gold_raw_with_warning(caplog):
    """Legacy `gold: z_mean` column maps MUST continue to work but should
    emit a deprecation-warning log so users know to migrate. It routes into
    gold_raw by default (conservative: unknown-space → raw)."""
    row = {"src": "hi", "tgt": "hola", "g": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold": "g"}
    with caplog.at_level(logging.WARNING, logger="mt_metrix.io.datasets"):
        seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 0.8
    assert seg.gold_z is None
    assert any("legacy `gold:` column key" in r.message for r in caplog.records)


def test_row_to_segment_invalid_gold_values_become_none():
    row = {"src": "hi", "tgt": "hola", "raw": "n/a", "z": ""}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw is None
    assert seg.gold_z is None
