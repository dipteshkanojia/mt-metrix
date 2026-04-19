"""Preflight guard for Marian-format COMET checkpoints.

Modern unbabel-comet's ``download_model()`` hardcodes
``<snapshot>/checkpoints/model.ckpt`` but the three Unbabel ``-marian`` HF
repos ship ``<snapshot>/checkpoints/marian.model.bin`` instead — so the
hardcoded path does not exist and ``load_from_checkpoint()`` raises an
unhelpful ``Invalid checkpoint path`` that sent a previous investigation
chasing an imaginary HF licence-gating issue.

The scorer now calls ``_raise_if_marian_layout()`` between
``download_model()`` and ``load_from_checkpoint()`` so the failure is loud
and self-describing. These tests pin that behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mt_metrix.scorers.comet import (
    UnsupportedMarianCheckpointError,
    _raise_if_marian_layout,
)


def test_raises_when_marian_bin_sits_alongside_missing_ckpt(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    checkpoints = snapshot / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "marian.model.bin").write_bytes(b"not a real marian bin")
    missing_ckpt = checkpoints / "model.ckpt"

    with pytest.raises(UnsupportedMarianCheckpointError) as exc:
        _raise_if_marian_layout("Unbabel/wmt21-comet-qe-mqm-marian", missing_ckpt)

    msg = str(exc.value)
    assert "Unbabel/wmt21-comet-qe-mqm-marian" in msg
    assert "marian.model.bin" in msg
    assert "unbabel-comet" in msg


def test_silent_when_ckpt_present(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    checkpoints = snapshot / "checkpoints"
    checkpoints.mkdir(parents=True)
    ckpt = checkpoints / "model.ckpt"
    ckpt.write_bytes(b"real ckpt")
    # Marian bin also present — shouldn't matter, the modern .ckpt is there.
    (checkpoints / "marian.model.bin").write_bytes(b"noise")

    _raise_if_marian_layout("Unbabel/wmt22-comet-da", ckpt)  # no exception


def test_silent_when_ckpt_missing_and_no_marian_bin(tmp_path: Path):
    """A truly missing checkpoint (no Marian sibling either) is a different
    failure — let unbabel-comet's own Invalid checkpoint path raise."""
    snapshot = tmp_path / "snapshot"
    checkpoints = snapshot / "checkpoints"
    checkpoints.mkdir(parents=True)
    missing_ckpt = checkpoints / "model.ckpt"

    _raise_if_marian_layout("Unbabel/whatever", missing_ckpt)  # no exception
