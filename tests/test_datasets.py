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
