"""SacreBLEU-backed reference metrics: BLEU, chrF++, TER.

All three are reference-based, so they require ``Segment.reference`` to be set.
Per-segment scores come from ``sacrebleu.sentence_*`` with paper-standard
defaults. A corpus-level score is also computed and exposed on the
``corpus_score`` attribute so the runner can write it into ``summary.json``.

chrF++ defaults (Popović 2017, WMT practice):
    chrf_order=6, chrf_word_order=2, use_effective_order=True, lowercase=False

BLEU defaults: sacrebleu defaults (WMT). If a language-specific tokeniser is
needed (ja / zh / ko), set ``params.tokenize`` accordingly ("ja-mecab",
"zh", "char", etc.). Defaults to ``"13a"`` which matches WMT practice.

TER defaults: sacrebleu defaults (no case folding, no punctuation stripping).
"""
from __future__ import annotations

import logging
from typing import Any

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import register_scorer

log = logging.getLogger(__name__)


class SacreBleuScorer:
    """Scorer for ``family="sacrebleu"``.

    ``params.metric`` selects which: ``"bleu"`` (default), ``"chrf"``, ``"ter"``.
    chrF++ is selected by combining ``metric: chrf`` with ``chrf_word_order: 2``.
    """

    def __init__(self, cfg: ScorerConfig) -> None:
        self._cfg = cfg
        self._metric: str = str(cfg.params.get("metric", "bleu")).lower()
        self.corpus_score: dict[str, Any] | None = None
        if self._metric not in {"bleu", "chrf", "ter"}:
            raise ValueError(
                f"sacrebleu scorer 'metric' must be bleu|chrf|ter, got {self._metric!r}"
            )

    @property
    def config(self) -> ScorerConfig:
        return self._cfg

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def family(self) -> str:
        return "sacrebleu"

    @property
    def needs_reference(self) -> bool:
        return True

    def load(self) -> None:
        """No-op: sacrebleu is a pure CPU library."""
        import sacrebleu  # noqa: F401  — check it's importable

    def score(self, segments: list[Segment]) -> list[SegmentScore]:
        import sacrebleu

        params = self._cfg.params
        targets = [s.target for s in segments]
        refs_list = [[s.reference or "" for s in segments]]  # list-of-lists

        # -- per-segment --
        out: list[SegmentScore] = []
        if self._metric == "bleu":
            tokenize = params.get("tokenize", "13a")
            smooth = params.get("smooth_method", "exp")
            lowercase = bool(params.get("lowercase", False))
            for seg in segments:
                sb = sacrebleu.sentence_bleu(
                    seg.target,
                    [seg.reference or ""],
                    tokenize=tokenize,
                    smooth_method=smooth,
                    lowercase=lowercase,
                )
                out.append(
                    SegmentScore(
                        segment_id=seg.segment_id,
                        score=float(sb.score),
                        extra={"signature": str(sb.signature) if hasattr(sb, "signature") else ""},
                    )
                )
            corpus = sacrebleu.corpus_bleu(
                targets, refs_list,
                tokenize=tokenize, smooth_method=smooth, lowercase=lowercase,
            )
        elif self._metric == "chrf":
            chrf_order = int(params.get("chrf_order", 6))
            chrf_word_order = int(params.get("chrf_word_order", 2))
            beta = int(params.get("beta", 2))
            eps_smoothing = bool(params.get("eps_smoothing", False))
            for seg in segments:
                sb = sacrebleu.sentence_chrf(
                    seg.target,
                    [seg.reference or ""],
                    char_order=chrf_order,
                    word_order=chrf_word_order,
                    beta=beta,
                    eps_smoothing=eps_smoothing,
                )
                out.append(
                    SegmentScore(
                        segment_id=seg.segment_id,
                        score=float(sb.score),
                        extra={},
                    )
                )
            corpus = sacrebleu.corpus_chrf(
                targets, refs_list,
                char_order=chrf_order, word_order=chrf_word_order, beta=beta,
                eps_smoothing=eps_smoothing,
            )
        else:  # ter
            normalized = bool(params.get("normalized", False))
            no_punct = bool(params.get("no_punct", False))
            asian_support = bool(params.get("asian_support", True))
            case_sensitive = bool(params.get("case_sensitive", True))
            for seg in segments:
                sb = sacrebleu.sentence_ter(
                    seg.target,
                    [seg.reference or ""],
                    normalized=normalized,
                    no_punct=no_punct,
                    asian_support=asian_support,
                    case_sensitive=case_sensitive,
                )
                out.append(
                    SegmentScore(
                        segment_id=seg.segment_id,
                        score=float(sb.score),
                        extra={},
                    )
                )
            corpus = sacrebleu.corpus_ter(
                targets, refs_list,
                normalized=normalized, no_punct=no_punct,
                asian_support=asian_support, case_sensitive=case_sensitive,
            )

        self.corpus_score = {
            "score": float(corpus.score),
            "signature": str(corpus.signature) if hasattr(corpus, "signature") else "",
            "metric": self._metric,
        }
        return out

    def unload(self) -> None:
        """No-op."""


def _factory(cfg: ScorerConfig) -> SacreBleuScorer:
    return SacreBleuScorer(cfg)


register_scorer("sacrebleu", _factory)
