"""The :class:`Scorer` protocol and :class:`ScorerConfig` shared by every metric family.

A scorer is a stateful object:

1. Constructed from a :class:`ScorerConfig` (family, model id, batch size, …).
2. ``load()`` — pulls weights, initialises engines (e.g. vLLM). Separated
   from construction so the runner can delay loading until just before use.
3. ``score(segments)`` — returns a ``list[SegmentScore]`` with the same order
   and ``segment_id`` as the input.
4. ``unload()`` — releases GPU memory so the next scorer in a sequence can
   load without OOM. Critical when chaining large models (COMET-XXL →
   Tower-13B) in a single SLURM job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mt_metrix.io.schema import Segment, SegmentScore


@dataclass
class ScorerConfig:
    """Configuration for a single scorer instance.

    Parameters
    ----------
    family: str
        Family name matching a registered factory: ``"comet"``, ``"tower"``,
        ``"sacrebleu"``.
    name: str
        Display name for this scorer — used as the column header in
        ``segments.tsv`` and the key in ``summary.json``. Does NOT need to be
        globally unique across families; runners may auto-prefix on collision.
    model: str | None
        HuggingFace repo ID or path to weights. ``None`` for families that
        don't load weights (sacrebleu).
    params: dict[str, Any]
        Family-specific parameters (e.g. ``batch_size`` for COMET,
        ``prompt_mode`` for Tower, ``chrf_word_order`` for sacrebleu).
    """

    family: str
    name: str
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """Runtime protocol every scorer implements."""

    @property
    def config(self) -> ScorerConfig: ...

    @property
    def name(self) -> str: ...

    @property
    def family(self) -> str: ...

    @property
    def needs_reference(self) -> bool:
        """True if this scorer requires ``Segment.reference`` for every segment.

        The runner checks this before calling ``score`` — datasets without a
        reference column cause reference-needing scorers to be skipped with a
        warning log rather than crash.
        """
        ...

    def load(self) -> None:
        """Pull weights / initialise engine. Idempotent."""
        ...

    def score(self, segments: list[Segment]) -> list[SegmentScore]:
        """Score each segment. Return list in the same order with matching ``segment_id``."""
        ...

    def unload(self) -> None:
        """Release GPU memory / close engine. Idempotent."""
        ...
