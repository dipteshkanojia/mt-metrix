"""End-to-end runner test — local TSV + sacrebleu, no GPU, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")

from mt_metrix.config import load_run_config
from mt_metrix.runner import run as run_pipeline
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import SCORER_REGISTRY, register_scorer


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


def test_runner_persists_across_scorer_crash(
    project_root: Path, fixtures_dir: Path, tmp_path: Path
):
    """A scorer raising mid-loop must not wipe earlier scorers' results.

    Regression guard for the 2026-04-19 OOM: four matrix jobs died at the
    wmt23-cometkiwi-da-xxl checkpoint load; because the old runner wrote
    outputs only after the full scorer loop finished, the two surviving
    scorers' (cometkiwi-da / cometkiwi-xl) in-memory scores were lost.
    The runner now persists after every scorer — this test locks that in.
    """
    # Load config up-front so we can hand the eventual out_dir to the
    # raising scorer — it'll peek at summary.json mid-run to prove the
    # earlier scorer was persisted BEFORE this one even started scoring.
    cfg = load_run_config(
        fixtures_dir / "tiny_run_with_ref.yaml",
        catalogue_roots=[project_root],
    )
    cfg.output.root = str(tmp_path)
    expected_out_dir = cfg.output_dir()

    observed: dict[str, object] = {}

    class _RaisingScorer:
        def __init__(self, cfg: ScorerConfig) -> None:
            self._cfg = cfg

        @property
        def config(self): return self._cfg
        @property
        def name(self): return self._cfg.name
        @property
        def family(self): return "raising"
        @property
        def needs_reference(self): return False

        def load(self): pass

        def score(self, segments):
            # Runner must have persisted the prior scorer to disk before
            # this one runs — peek at summary.json to prove it.
            summary_path = expected_out_dir / "summary.json"
            observed["summary_existed_mid_loop"] = summary_path.is_file()
            if summary_path.is_file():
                observed["metrics_mid_loop"] = set(
                    json.loads(summary_path.read_text())["metrics"].keys()
                )
            raise RuntimeError("simulated host-RAM OOM during score")

        def unload(self): pass

    register_scorer("raising", _RaisingScorer)
    try:
        # Stash the raising scorer in the middle so we test BOTH properties:
        # (1) the scorer before it gets persisted before the crash, and
        # (2) the scorer after it still runs and gets persisted too.
        sacrebleu_bleu = cfg.scorers[0]  # sacrebleu/bleu from fixture
        other_sacrebleus = cfg.scorers[1:]
        raising = ScorerConfig(
            family="raising", name="raising-mid-loop", model=None, params={}
        )
        cfg.scorers = [sacrebleu_bleu, raising, *other_sacrebleus]

        out_dir = run_pipeline(cfg)
    finally:
        SCORER_REGISTRY.pop("raising", None)

    # The raising scorer saw summary.json on disk with bleu already in it
    # — proves incremental persistence, not just final-write ordering.
    assert observed.get("summary_existed_mid_loop") is True
    assert sacrebleu_bleu.name in observed["metrics_mid_loop"]

    # summary.json + segments.tsv both exist
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "segments.tsv").is_file()

    summary = json.loads((out_dir / "summary.json").read_text())

    # Scorer that ran BEFORE the crash is persisted
    assert sacrebleu_bleu.name in summary["metrics"]
    assert summary["metrics"][sacrebleu_bleu.name]["n"] == 10

    # Scorers AFTER the crash also ran and are persisted
    for s in other_sacrebleus:
        assert s.name in summary["metrics"]

    # The raising scorer is recorded as skipped with the right reason
    skipped = {s["name"]: s["reason"] for s in summary["skipped_metrics"]}
    assert "raising-mid-loop" in skipped
    assert skipped["raising-mid-loop"].startswith("runtime:")
