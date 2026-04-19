"""Output writers: TSV (human-friendly), JSONL (rich), summary JSON (aggregates)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mt_metrix.io.schema import Segment, SegmentScore

log = logging.getLogger(__name__)


def write_segments_tsv(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
) -> None:
    """Write a flat TSV with one row per segment and one column per metric.

    Columns: ``segment_id, lang_pair, domain, source, target, reference, gold, <metric1>, <metric2>, ...``
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        row: dict[str, Any] = {
            "segment_id": seg.segment_id,
            "lang_pair": seg.lang_pair,
            "domain": seg.domain,
            "source": seg.source,
            "target": seg.target,
            "reference": seg.reference or "",
            "gold": "" if seg.gold is None else seg.gold,
        }
        for name, col in scores_by_name.items():
            if i < len(col):
                val = col[i].score
                row[name] = "" if val is None else val
            else:
                row[name] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def write_segments_jsonl(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
) -> None:
    """Write one JSON object per line with all scorer extras included."""
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            obj: dict[str, Any] = {
                "segment_id": seg.segment_id,
                "lang_pair": seg.lang_pair,
                "domain": seg.domain,
                "source": seg.source,
                "target": seg.target,
                "reference": seg.reference,
                "gold": seg.gold,
                "scores": {},
            }
            for name, col in scores_by_name.items():
                if i < len(col):
                    obj["scores"][name] = {
                        "score": col[i].score,
                        "extra": col[i].extra,
                    }
            f.write(json.dumps(obj, ensure_ascii=False, default=_json_default))
            f.write("\n")


def write_summary(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
    skipped_metrics: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    corpus_scores: dict[str, Any] | None = None,
) -> None:
    """Write a one-file summary JSON with aggregates and correlations.

    Includes:

    - ``metrics``: per-metric mean, std, min, max, Pearson/Spearman/Kendall/SPA
      vs gold where gold is available.
    - ``corpus_scores``: optional per-scorer corpus-level numbers (BLEU, chrF++
      score on the whole dataset, etc.).
    - ``skipped_metrics``: list of {name, reason} for scorers that could not
      run (e.g. needed reference but dataset had none).
    - ``run_metadata``: whatever the runner chose to include (git SHA, torch
      version, timing).
    """
    import numpy as np

    gold = np.array(
        [s.gold for s in segments if s.gold is not None],
        dtype=float,
    )
    gold_mask = np.array([s.gold is not None for s in segments], dtype=bool)

    metrics_summary: dict[str, Any] = {}
    for name, col in scores_by_name.items():
        preds = np.array([c.score if c.score is not None else np.nan for c in col], dtype=float)
        finite = np.isfinite(preds)
        summary = {
            "n": int(finite.sum()),
            "n_nan": int((~finite).sum()),
            "mean": _safe_float(preds[finite].mean()) if finite.any() else None,
            "std": _safe_float(preds[finite].std()) if finite.any() else None,
            "min": _safe_float(preds[finite].min()) if finite.any() else None,
            "max": _safe_float(preds[finite].max()) if finite.any() else None,
        }
        if gold_mask.any():
            usable = finite & gold_mask
            if usable.sum() >= 2:
                summary["correlation_vs_gold"] = _correlations(preds[usable], gold[gold_mask & finite])
            else:
                summary["correlation_vs_gold"] = None
        metrics_summary[name] = summary

    payload: dict[str, Any] = {
        "run_metadata": run_metadata,
        "metrics": metrics_summary,
        "corpus_scores": corpus_scores or {},
        "skipped_metrics": skipped_metrics,
        "n_segments": len(segments),
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _correlations(pred, gold) -> dict[str, float | None]:
    """Pearson, Spearman, Kendall, SPA between two aligned arrays."""
    from scipy import stats

    result: dict[str, float | None] = {
        "pearson": None,
        "spearman": None,
        "kendall": None,
        "spa": None,
        "n": int(len(pred)),
    }
    if len(pred) < 2:
        return result
    try:
        r, _ = stats.pearsonr(pred, gold)
        result["pearson"] = _safe_float(r)
    except Exception as e:  # pragma: no cover — defensive
        log.debug("pearson failed: %s", e)
    try:
        rho, _ = stats.spearmanr(pred, gold)
        result["spearman"] = _safe_float(rho)
    except Exception as e:  # pragma: no cover
        log.debug("spearman failed: %s", e)
    try:
        tau, _ = stats.kendalltau(pred, gold)
        result["kendall"] = _safe_float(tau)
    except Exception as e:  # pragma: no cover
        log.debug("kendall failed: %s", e)
    result["spa"] = _safe_float(_soft_pairwise_accuracy(pred, gold))
    return result


def _soft_pairwise_accuracy(pred, gold) -> float:
    """Soft Pairwise Accuracy.

    SPA = mean over ordered pairs of 1 if sign(pred_i - pred_j) == sign(gold_i - gold_j)
    else 0. Ties (gold_i == gold_j) are weighted 0.5.
    """
    import numpy as np

    pred = np.asarray(pred)
    gold = np.asarray(gold)
    n = len(pred)
    if n < 2:
        return 0.0
    diffs_pred = np.sign(pred[:, None] - pred[None, :])
    diffs_gold = np.sign(gold[:, None] - gold[None, :])
    iu = np.triu_indices(n, k=1)
    dp = diffs_pred[iu]
    dg = diffs_gold[iu]
    if len(dp) == 0:
        return 0.0
    matches = (dp == dg).astype(float)
    ties = (dg == 0).astype(float)
    score = matches + 0.5 * ties * (1.0 - matches)
    return float(score.mean())


def _safe_float(v) -> float | None:
    import math
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def _json_default(obj):
    """JSON encoder for numpy types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
