"""Tests for the mt-metrix CLI — focuses on argparse wiring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sacrebleu")

from mt_metrix.cli import build_parser, main


def test_parser_exposes_subcommands():
    parser = build_parser()
    # argparse stores subparser choices on the _SubParsersAction
    subs_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subs_action.choices) == {
        "score", "submit", "list-models", "list-datasets",
        "correlate", "tabulate", "download",
    }


def test_cli_list_models_runs(capsys, project_root: Path, monkeypatch):
    monkeypatch.chdir(project_root)
    rc = main(["list-models", "--family", "sacrebleu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sacrebleu/bleu" in out
    assert "sacrebleu/chrf" in out


def test_cli_list_datasets_runs(capsys, project_root: Path, monkeypatch):
    monkeypatch.chdir(project_root)
    rc = main(["list-datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "surrey_legal" in out


def test_cli_score_end_to_end(project_root: Path, fixtures_dir: Path, tmp_path: Path):
    rc = main([
        "score",
        "--config", str(fixtures_dir / "tiny_run_with_ref.yaml"),
        "--output-root", str(tmp_path),
    ])
    assert rc == 0
    run_dir = tmp_path / "test_tiny_with_ref"
    assert (run_dir / "summary.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_segments"] == 10


def test_cli_correlate_from_existing_run(
    project_root: Path, fixtures_dir: Path, tmp_path: Path, capsys
):
    # First produce a run
    main([
        "score",
        "--config", str(fixtures_dir / "tiny_run_with_ref.yaml"),
        "--output-root", str(tmp_path),
    ])
    capsys.readouterr()  # clear
    rc = main(["correlate", "--run", str(tmp_path / "test_tiny_with_ref")])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Expect one correlation entry per metric column
    assert set(payload) >= {"bleu", "chrf", "chrf++", "ter"}


def test_cli_submit_dry_run_invokes_wrapper(
    project_root: Path, fixtures_dir: Path, capsys, monkeypatch
):
    import subprocess
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout="dry-run mode: pre-flight OK, not submitting.\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(project_root)
    rc = main([
        "submit",
        "--config", str(fixtures_dir / "tiny_run_with_ref.yaml"),
        "--dry-run",
    ])
    assert rc == 0
    assert calls, "submit.sh was not invoked"
    cmd = calls[0]
    assert cmd[0].endswith("scripts/submit.sh")
    assert "--dry-run" in cmd
    out = capsys.readouterr().out
    assert "[dry-run]" in out


def test_cli_submit_forwards_partition_and_gpus(
    project_root: Path, fixtures_dir: Path, monkeypatch
):
    import subprocess
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="Submitted batch job 99\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(project_root)
    main([
        "submit",
        "--config", str(fixtures_dir / "tiny_run_with_ref.yaml"),
        "--partition", "rtx_a6000_risk",
        "--gpus", "1",
        "--time", "02:00:00",
    ])
    cmd = calls[0]
    assert "-p" in cmd and "rtx_a6000_risk" in cmd
    assert "--gres=gpu:1" in cmd
    assert "--time=02:00:00" in cmd
