"""Tests for GEMBA-DA / GEMBA-MQM prompt construction + response parsing."""
from __future__ import annotations

import pytest

from mt_metrix.prompts.gemba_da import (
    build_gemba_da_prompt,
    parse_gemba_da_score,
)
from mt_metrix.prompts.gemba_mqm import (
    SEVERITY_WEIGHTS,
    build_gemba_mqm_prompt,
    parse_gemba_mqm_score,
)


# ---------------------------------------------------------------------------
# GEMBA-DA prompt building
# ---------------------------------------------------------------------------

def test_gemba_da_prompt_uses_language_names_for_iso_codes():
    p = build_gemba_da_prompt("Hello.", "Bonjour.", "en-fr")
    assert "English" in p
    assert "French" in p
    assert "Score (0 - 100):" in p
    assert "Hello." in p
    assert "Bonjour." in p


def test_gemba_da_prompt_falls_back_for_unknown_codes():
    p = build_gemba_da_prompt("src", "tgt", "xx-yy")
    # unknown codes stay as-is
    assert "xx" in p
    assert "yy" in p


def test_gemba_da_prompt_handles_missing_lang_pair():
    p = build_gemba_da_prompt("src", "tgt", "")
    assert "source" in p
    assert "target" in p


def test_gemba_da_prompt_supports_indic_codes():
    p = build_gemba_da_prompt("Hello.", "હેલો.", "en-gu")
    assert "English" in p
    assert "Gujarati" in p


# ---------------------------------------------------------------------------
# GEMBA-DA score parsing
# ---------------------------------------------------------------------------

def test_parse_gemba_da_integer():
    score, ok = parse_gemba_da_score("85")
    assert ok is True
    assert score == pytest.approx(85.0)


def test_parse_gemba_da_decimal():
    score, ok = parse_gemba_da_score("Score: 73.5")
    assert ok is True
    assert score == pytest.approx(73.5)


def test_parse_gemba_da_with_prose():
    score, ok = parse_gemba_da_score("The translation scores 92 out of 100")
    assert ok is True
    assert score == pytest.approx(92.0)


def test_parse_gemba_da_out_of_range():
    score, ok = parse_gemba_da_score("999")
    assert ok is False
    assert score is None


def test_parse_gemba_da_empty():
    score, ok = parse_gemba_da_score("")
    assert ok is False
    assert score is None


def test_parse_gemba_da_no_number():
    score, ok = parse_gemba_da_score("cannot say")
    assert ok is False
    assert score is None


def test_parse_gemba_da_zero_is_valid():
    score, ok = parse_gemba_da_score("Score: 0")
    assert ok is True
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GEMBA-MQM prompt + parsing
# ---------------------------------------------------------------------------

def test_gemba_mqm_prompt_builds():
    p = build_gemba_mqm_prompt("Hello.", "Bonjour.", "en-fr")
    assert "critical" in p.lower()
    assert "major" in p.lower()
    assert "minor" in p.lower()
    assert "no-error" in p.lower()
    assert "English" in p
    assert "French" in p


def test_severity_weights_table():
    assert SEVERITY_WEIGHTS["critical"] == -25
    assert SEVERITY_WEIGHTS["major"] == -5
    assert SEVERITY_WEIGHTS["minor"] == -1
    assert SEVERITY_WEIGHTS["no-error"] == 0


def test_parse_gemba_mqm_no_error_response():
    score, ok, errors = parse_gemba_mqm_score("no-error")
    assert ok is True
    assert score == pytest.approx(100.0)
    assert errors == []


def test_parse_gemba_mqm_single_minor():
    resp = 'minor - fluency - "the"'
    score, ok, errors = parse_gemba_mqm_score(resp)
    assert ok is True
    assert score == pytest.approx(99.0)  # 100 - 1
    assert len(errors) == 1
    assert errors[0].severity == "minor"
    assert errors[0].category == "fluency"


def test_parse_gemba_mqm_major_plus_minor():
    resp = (
        'major - accuracy - "word1"\n'
        'minor - fluency - "word2"\n'
    )
    score, ok, errors = parse_gemba_mqm_score(resp)
    assert ok is True
    assert score == pytest.approx(94.0)  # 100 - 5 - 1
    assert len(errors) == 2


def test_parse_gemba_mqm_critical_caps_at_zero():
    resp = 'critical - accuracy - "span"\ncritical - accuracy - "span2"\n'
    # 100 - 50 = 50
    score, _, _ = parse_gemba_mqm_score(resp)
    assert score == pytest.approx(50.0)


def test_parse_gemba_mqm_critical_many_clips_to_zero():
    resp = "\n".join([f'critical - accuracy - "x{i}"' for i in range(10)])
    score, ok, _ = parse_gemba_mqm_score(resp)
    assert ok is True
    assert score == pytest.approx(0.0)  # clipped


def test_parse_gemba_mqm_garbage_returns_not_ok():
    score, ok, _ = parse_gemba_mqm_score("I don't know")
    assert ok is False
    assert score is None
