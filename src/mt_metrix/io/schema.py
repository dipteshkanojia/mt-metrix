"""Unified segment schema used across dataset loaders and scorers.

A :class:`Segment` is the input unit: source, MT output, optional reference,
optional gold score, and metadata.

A :class:`SegmentScore` is the output unit per scorer per segment: a numeric
score plus an open `extra` payload for model-specific fields (XCOMET error
spans, Tower LLM raw output, parse flags, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


PredictionSpace = Literal["raw_da", "z_da"]
"""Space a scorer predicts in.

- ``raw_da`` — raw Direct Assessment 0-100 (GEMBA-DA LLM prompt output,
  GEMBA-MQM derived 100 − error-penalty, sacrebleu reference-based scores).
- ``z_da`` — per-rater z-normalised DA (COMET-QE, CometKiwi, XCOMET
  training targets).

Add new spaces here — tabulate's gold-column resolver reads this alias.
"""


@dataclass
class Segment:
    """A single translation instance to be scored."""

    source: str
    target: str
    reference: Optional[str] = None
    gold_raw: Optional[float] = None
    gold_z: Optional[float] = None
    lang_pair: str = ""
    domain: str = "general"
    segment_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gold(self) -> Optional[float]:
        """Backward-compat accessor. Prefer ``gold_z`` over ``gold_raw``.

        Existing code paths that read ``seg.gold`` keep getting a z-scored
        value when available (the historic default for QE datasets). New
        code should read ``gold_raw`` / ``gold_z`` explicitly.
        """
        return self.gold_z if self.gold_z is not None else self.gold_raw

    def has_reference(self) -> bool:
        return self.reference is not None and len(self.reference) > 0

    def has_gold(self) -> bool:
        return self.gold_raw is not None or self.gold_z is not None


@dataclass
class SegmentScore:
    """A scorer's output for one segment.

    `score` is the canonical numeric output (float). `extra` carries anything
    else the scorer wants to surface (error spans, raw LLM text, parse flags,
    confidence intervals, …). `extra` is written to `segments.jsonl` but not
    the TSV — the TSV stays flat and human-readable.
    """

    segment_id: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)
