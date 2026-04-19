"""End-to-end runner test — local TSV + sacrebleu, no GPU, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")

from mt_metrix.config import load_run_config
from mt_metrix.runner import run as run_pipeline


def test_runner_with_refs_writes_all_outputs(
    project_root: Path, fixtures_dir: Path, tmp_path: Path
):
    cfg = load_run_config(
        fixtures_dir / "tiny_run_with_ref.yaml",
        catalogue_roots=[project_root],
    )
    cfg.output.root = str(tmp_path)
    out_dir = run_pipeline(cfg)

    assert out_dir.is_dir()
    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "segments.tsv").is_file()
    assert (out_dir / "segments.jsonl").is_file()
    assert (out_dir / "summary.json").is_file()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_segments"] == 10
    # All four sacrebleu metrics produce scores
    assert set(summary["metrics"]) == {"bleu", "chrf", "chrf++", "ter"}
    # Correlation against gold is computed
    for name in ("bleu", "chrf", "chrf++", "ter"):
        assert summary["metrics"][name]["n"] == 10
        assert summary["metrics"][name]["correlation_vs_gold"] is not None
    # corpus scores present for all four
    assert set(summary["corpus_scores"]) == {"bleu", "chrf", "chrf++", "ter"}


def test_runner_without_refs_skips_ref_metrics(
    project_root: Path, fixtures_dir: Path, tmp_path: Path
):
    cfg = load_run_config(
        fixtures_dir / "tiny_run_no_ref.yaml",
        catalogue_roots=[project_root],
    )
    cfg.output.root = str(tmp_path)
    out_dir = run_pipeline(cfg)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_segments"] == 10
    assert summary["metrics"] == {}
    names = {s["name"] for s in summary["skipped_metrics"]}
    assert names == {"bleu", "chrf", "chrf++", "ter"}
    assert all(s["reason"] == "dataset-has-no-references" for s in summary["skipped_metrics"])
