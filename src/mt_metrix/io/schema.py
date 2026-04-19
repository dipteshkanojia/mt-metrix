"""Unified segment schema used across dataset loaders and scorers.

A :class:`Segment` is the input unit: source, MT output, optional reference,
optional gold score, and metadata.

A :class:`SegmentScore` is the output unit per scorer per segment: a numeric
score plus an open `extra` payload for model-specific fields (XCOMET error
spans, Tower LLM raw output, parse flags, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Segment:
    """A single translation instance to be scored."""

    source: str
    target: str
    reference: Optional[str] = None
    gold: Optional[float] = None
    lang_pair: str = ""
    domain: str = "general"
    segment_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def has_reference(self) -> bool:
        return self.reference is not None and len(self.reference) > 0

    def has_gold(self) -> bool:
        return self.gold is not None


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
