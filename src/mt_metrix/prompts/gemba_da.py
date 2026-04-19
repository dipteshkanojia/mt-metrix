"""GEMBA-DA prompt (Kocmi & Federmann, "Large Language Models Are State-of-the-Art Evaluators of Translation Quality", EAMT 2023).

The prompt asks the LLM to assign a single integer in [0, 100]. Lower end
indicates more errors; higher end indicates fewer. We parse the first
integer in [0, 100] from the response. If parsing fails, the caller
surfaces ``parse_ok=False`` in the segment extras.

Language-pair names are derived from ISO codes where possible.
"""
from __future__ import annotations

import re

_ISO_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "pa": "Punjabi",
    "ur": "Urdu",
    "et": "Estonian",
    "ne": "Nepali",
    "si": "Sinhala",
    "ro": "Romanian",
    "cs": "Czech",
    "km": "Khmer",
    "ps": "Pashto",
    "he": "Hebrew",
    "yo": "Yoruba",
}


def _language_name(code: str) -> str:
    code = code.strip().lower()
    return _ISO_NAMES.get(code, code)


def build_gemba_da_prompt(source: str, target: str, lang_pair: str) -> str:
    """Build the GEMBA-DA prompt for a single (source, target) pair.

    ``lang_pair`` is an ISO-like ``"src-tgt"`` string (e.g. ``"en-de"``).
    """
    if "-" in lang_pair:
        src_code, tgt_code = lang_pair.split("-", 1)
        src_name, tgt_name = _language_name(src_code), _language_name(tgt_code)
    else:
        src_name, tgt_name = "source", "target"

    return (
        f"Score the following translation from {src_name} to {tgt_name} with respect "
        f"to the source sentence on a continuous scale from 0 to 100, where a score of "
        f"zero means \"no meaning preserved\" and score of one hundred means \"perfect "
        f"meaning and grammar\".\n\n"
        f"{src_name} source: \"{source}\"\n"
        f"{tgt_name} translation: \"{target}\"\n\n"
        f"Score (0 - 100):"
    )


_SCORE_RE = re.compile(r"(?<!\d)(\d{1,3})(?:\.(\d+))?(?!\d)")


def parse_gemba_da_score(response: str) -> tuple[float | None, bool]:
    """Parse the first valid score in [0, 100] from the LLM response.

    Returns ``(score_or_None, parse_ok)``. ``parse_ok`` is ``False`` if no
    number was found or it was outside [0, 100].
    """
    for match in _SCORE_RE.finditer(response):
        integer_part = match.group(1)
        decimal_part = match.group(2)
        try:
            value = float(f"{integer_part}.{decimal_part}") if decimal_part else float(integer_part)
        except ValueError:
            continue
        if 0.0 <= value <= 100.0:
            return value, True
    return None, False
