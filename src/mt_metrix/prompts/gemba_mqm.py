"""GEMBA-MQM prompt (Kocmi & Federmann 2023).

GEMBA-MQM asks the LLM to identify error spans labelled by MQM severity:

- ``critical``  (nonsense / unusable; weight −25)
- ``major``     (significant deviation; weight −5)
- ``minor``     (small error; weight −1)
- ``no-error``  (nothing wrong; weight 0)

The final numeric score is computed deductively, clipped into [0, 100]:

    score = max(0, min(100, 100 + sum(weights)))

This matches the formulation used in Kocmi & Federmann (2023) and Freitag
et al. (MQM weighting). The raw LLM output and the parsed error list are
both retained in ``extra`` for downstream analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SEVERITY_WEIGHTS = {
    "critical": -25,
    "major": -5,
    "minor": -1,
    "no-error": 0,
    "no error": 0,
}


@dataclass
class GembaMqmError:
    severity: str
    category: str
    span: str


def build_gemba_mqm_prompt(source: str, target: str, lang_pair: str) -> str:
    """Build a GEMBA-MQM prompt for one (source, target) pair."""
    from mt_metrix.prompts.gemba_da import _language_name

    if "-" in lang_pair:
        src_code, tgt_code = lang_pair.split("-", 1)
        src_name, tgt_name = _language_name(src_code), _language_name(tgt_code)
    else:
        src_name, tgt_name = "source", "target"

    return (
        f"You are an annotator for the quality of machine translation. Your task is "
        f"to identify errors and assess the quality of the translation.\n\n"
        f"Based on the source segment and machine translation surrounded with triple "
        f"backticks, identify error types in the translation and classify them. The "
        f"categories of errors are: accuracy (addition, mistranslation, omission, "
        f"untranslated text), fluency (character encoding, grammar, inconsistency, "
        f"punctuation, register, spelling), style (awkward), terminology "
        f"(inappropriate for context, inconsistent use), non-translation, other, or "
        f"no-error.\nEach error is classified as one of three categories: critical, "
        f"major, and minor. Critical errors inhibit comprehension of the text. Major "
        f"errors disrupt the flow, but what the text is trying to say is still "
        f"understandable. Minor errors are technically errors, but do not disrupt the "
        f"flow or hinder comprehension.\n\n"
        f"{src_name} source:\n```{source}```\n"
        f"{tgt_name} translation:\n```{target}```\n\n"
        f"Class the errors, one per line, in the format: "
        f"<severity> - <category> - \"<error span>\"\n"
        f"If there are no errors, respond with: no-error\n"
    )


_ERROR_LINE_RE = re.compile(
    r"(?P<sev>critical|major|minor|no[- ]?error)"
    r"\s*[-:—]\s*"
    r"(?P<cat>[A-Za-z /\-]+?)"
    r"\s*[-:—]\s*"
    r"['\"]?(?P<span>.+?)['\"]?$",
    re.IGNORECASE,
)


def parse_gemba_mqm_score(response: str) -> tuple[float | None, bool, list[GembaMqmError]]:
    """Parse GEMBA-MQM response.

    Returns ``(score, parse_ok, errors)``. Score is in [0, 100].
    ``parse_ok`` is ``True`` iff at least one recognised line was found OR the
    response explicitly indicates ``no-error``.
    """
    errors: list[GembaMqmError] = []
    any_match = False

    text = response.strip()
    if re.search(r"\bno[- ]?error(s)?\b", text, flags=re.IGNORECASE) and not re.search(
        r"\b(critical|major|minor)\b", text, flags=re.IGNORECASE
    ):
        return 100.0, True, []

    total_penalty = 0
    for line in text.splitlines():
        line = line.strip().strip("- ").strip()
        if not line:
            continue
        m = _ERROR_LINE_RE.search(line)
        if not m:
            # try a looser "severity only" pattern
            m2 = re.match(
                r"^(?P<sev>critical|major|minor|no[- ]?error)\b.*",
                line,
                flags=re.IGNORECASE,
            )
            if m2:
                sev = m2.group("sev").lower().replace("-", " ").strip()
                sev = "no-error" if sev.startswith("no") else sev
                weight = SEVERITY_WEIGHTS.get(sev, 0)
                total_penalty += weight
                any_match = True
                errors.append(
                    GembaMqmError(severity=sev, category="unspecified", span="")
                )
            continue
        sev = m.group("sev").lower().replace("-", " ").strip()
        sev = "no-error" if sev.startswith("no") else sev
        cat = m.group("cat").strip()
        span = m.group("span").strip().strip("'\"")
        weight = SEVERITY_WEIGHTS.get(sev, 0)
        total_penalty += weight
        any_match = True
        errors.append(GembaMqmError(severity=sev, category=cat, span=span))

    if not any_match:
        return None, False, []

    score = max(0.0, min(100.0, 100.0 + float(total_penalty)))
    return score, True, errors
