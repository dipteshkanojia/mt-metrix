"""Unit tests for COMET's ``needs_reference`` inference heuristic.

The heuristic is only the fallback when no ``needs_reference`` flag is
supplied via config or catalogue, but it used to flip the wrong way for
several Unbabel QE checkpoints whose names contain a ``-qe-da`` / ``-qe-mqm``
suffix — quietly turning them into "needs references" and causing the
2026-04-19 full-matrix run to silently skip them on ref-free QE datasets.

These tests pin the contract so a future refactor doesn't reintroduce the
same failure mode.
"""
from __future__ import annotations

import pytest

from mt_metrix.scorers.comet import _infer_needs_reference


# Each case: (model_id, expected_needs_reference, rationale).
CASES = [
    # --- Reference-free QE (must return False) --------------------------
    ("Unbabel/wmt22-cometkiwi-da", False, "Kiwi — name shortcut"),
    ("Unbabel/wmt23-cometkiwi-da-xl", False, "Kiwi-XL"),
    ("Unbabel/wmt23-cometkiwi-da-xxl", False, "Kiwi-XXL"),
    ("Unbabel/wmt20-comet-qe-da", False, "-qe-da suffix"),
    ("Unbabel/wmt21-comet-qe-da-marian", False, "-qe-da-marian suffix"),
    ("Unbabel/wmt21-comet-qe-mqm-marian", False, "-qe-mqm-marian suffix"),
    # XCOMET in QE mode — names end in -qe, the catalogue also sets
    # needs_reference: false explicitly, but the heuristic should still
    # agree as a defence in depth.
    ("Unbabel/XCOMET-XL-qe", False, "trailing -qe"),
    ("Unbabel/custom-some-qe", False, "trailing -qe"),

    # --- Reference-based DA (must return True) --------------------------
    ("Unbabel/wmt22-comet-da", True, "plain DA ref"),
    ("Unbabel/eamt22-cometinho-da", True, "Cometinho DA"),
    ("Unbabel/wmt20-comet-da", True, "historical DA"),
    ("Unbabel/wmt21-comet-da-marian", True, "Marian DA"),
    # XCOMET in REF mode — the catalogue's explicit needs_reference: true
    # matters in practice, but the heuristic stays on the ref side here.
    ("Unbabel/XCOMET-XL", True, "XCOMET default is ref"),
    ("Unbabel/XCOMET-XXL", True, "XCOMET-XXL default is ref"),
]


@pytest.mark.parametrize("model_id,expected,rationale", CASES)
def test_infer_needs_reference(model_id: str, expected: bool, rationale: str):
    assert _infer_needs_reference(model_id) is expected, (
        f"heuristic flipped for {model_id} ({rationale})"
    )
