"""Unit tests for the XXL COMET auto-downshift introduced 2026-04-23.

The 2026-04-19 full-matrix run OOMed on XCOMET-XXL / CometKiwi-XXL at
batch_size=8 on 48 GB L40s cards, even though the catalogue defaults
batch_size=8 per the COMET-XXL paper (which tests on 80 GB A100). Rather
than pin batch_size per partition, the scorer now detects VRAM at
runtime and downshifts XXL to batch=4 on <60 GiB cards while leaving
A100 80 GB runs at batch=8 for speed.

These tests pin the resolver logic so a refactor can't reintroduce the
OOM failure mode. They don't touch torch / CUDA directly — VRAM is
passed in as a parameter so the pure branching logic is exercised
without a GPU.
"""
from __future__ import annotations

import pytest

from mt_metrix.scorers.comet import (
    _XXL_FULL_BATCH_VRAM_GB_THRESHOLD,
    _XXL_SAFE_BATCH_SIZE,
    _is_xxl,
    _resolve_xxl_batch_size,
)


# --- _is_xxl --------------------------------------------------------------

IS_XXL_CASES = [
    # Must match XXL family (case-insensitive substring).
    ("Unbabel/wmt23-cometkiwi-da-xxl", True),
    ("Unbabel/XCOMET-XXL", True),
    ("Unbabel/XCOMET-XXL-qe", True),
    ("some/custom-XXL-fork", True),
    # Must NOT match non-XXL siblings.
    ("Unbabel/wmt23-cometkiwi-da-xl", False),
    ("Unbabel/XCOMET-XL", False),
    ("Unbabel/wmt22-cometkiwi-da", False),
    ("Unbabel/wmt22-comet-da", False),
    ("Unbabel/eamt22-cometinho-da", False),
    # Defensive edges.
    ("", False),
    (None, False),
]


@pytest.mark.parametrize("model_id,expected", IS_XXL_CASES)
def test_is_xxl(model_id, expected):
    assert _is_xxl(model_id) is expected


# --- _resolve_xxl_batch_size ----------------------------------------------

def test_non_xxl_models_are_never_downshifted():
    """XL / DA / Kiwi-base are unaffected by the downshift rule."""
    for model in [
        "Unbabel/wmt22-cometkiwi-da",
        "Unbabel/wmt23-cometkiwi-da-xl",
        "Unbabel/XCOMET-XL",
        "Unbabel/wmt22-comet-da",
    ]:
        # Even on a tiny 11 GB 2080ti the resolver must return the original.
        assert _resolve_xxl_batch_size(model, 16, 11.0) == 16
        assert _resolve_xxl_batch_size(model, 64, 11.0) == 64
        assert _resolve_xxl_batch_size(model, 4, 11.0) == 4


def test_xxl_on_a100_80gb_keeps_full_batch():
    """A100 has 80 GiB — threshold is 60 GiB so batch=8 must survive."""
    for model in [
        "Unbabel/wmt23-cometkiwi-da-xxl",
        "Unbabel/XCOMET-XXL",
        "Unbabel/XCOMET-XXL-qe",
    ]:
        assert _resolve_xxl_batch_size(model, 8, 80.0) == 8


def test_xxl_on_48gb_l40s_downshifts_to_safe_batch():
    """48 GiB L40s / RTX8000 / A6000 — must downshift batch=8 to 4."""
    for model in [
        "Unbabel/wmt23-cometkiwi-da-xxl",
        "Unbabel/XCOMET-XXL",
        "Unbabel/XCOMET-XXL-qe",
    ]:
        assert _resolve_xxl_batch_size(model, 8, 48.0) == _XXL_SAFE_BATCH_SIZE


def test_xxl_threshold_boundary_is_inclusive_at_60gb():
    """60 GiB exactly is treated as >= threshold (no downshift) so we
    don't accidentally downshift on a future 60 GiB card without cause.
    """
    model = "Unbabel/wmt23-cometkiwi-da-xxl"
    assert _resolve_xxl_batch_size(model, 8, _XXL_FULL_BATCH_VRAM_GB_THRESHOLD) == 8
    # Just under threshold → downshift.
    assert _resolve_xxl_batch_size(
        model, 8, _XXL_FULL_BATCH_VRAM_GB_THRESHOLD - 0.1
    ) == _XXL_SAFE_BATCH_SIZE


def test_xxl_respects_user_override_below_safe():
    """If the user already configured batch<=4, don't upshift even on A100."""
    model = "Unbabel/wmt23-cometkiwi-da-xxl"
    assert _resolve_xxl_batch_size(model, 4, 80.0) == 4
    assert _resolve_xxl_batch_size(model, 2, 48.0) == 2
    assert _resolve_xxl_batch_size(model, 1, 48.0) == 1


def test_xxl_with_unknown_vram_trusts_config():
    """If VRAM detection fails (CPU run, weird driver), don't surprise the
    user with a silent downshift — respect their configured batch_size.
    """
    model = "Unbabel/wmt23-cometkiwi-da-xxl"
    assert _resolve_xxl_batch_size(model, 8, None) == 8
    assert _resolve_xxl_batch_size(model, 16, None) == 16


def test_xxl_on_24gb_3090_downshifts():
    """3090 (24 GiB) is below threshold — still downshifts to 4 (the run
    may still OOM at batch=4 on a 24 GiB card, but at least we've done
    what the resolver can do in isolation)."""
    model = "Unbabel/XCOMET-XXL"
    assert _resolve_xxl_batch_size(model, 8, 24.0) == _XXL_SAFE_BATCH_SIZE


def test_xxl_case_insensitive_match():
    """Catalogue writes 'xxl' lowercase; HF repo id is 'XXL' uppercase.
    Both paths must activate the downshift."""
    assert _resolve_xxl_batch_size("some/XXL-model", 8, 48.0) == _XXL_SAFE_BATCH_SIZE
    assert _resolve_xxl_batch_size("some/xxl-model", 8, 48.0) == _XXL_SAFE_BATCH_SIZE
    assert _resolve_xxl_batch_size("some/XxL-model", 8, 48.0) == _XXL_SAFE_BATCH_SIZE


def test_xxl_with_none_model_is_passthrough():
    """Defensive: a None model_id shouldn't trip the XXL branch."""
    assert _resolve_xxl_batch_size(None, 8, 48.0) == 8
