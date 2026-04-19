"""Dataset loaders for mt-metrix.

Three loader kinds, picked by ``DatasetConfig.kind``:

- ``local``    — TSV/CSV/JSONL/parquet on disk
- ``huggingface`` — any HF hub dataset (e.g. ``surrey-nlp/Legal-QE``)
- ``gyroqe``   — loader that understands gyroQE's unified catalogue

All produce ``list[Segment]`` using the unified schema. Column mapping is
config-driven so we don't impose a canonical raw format on every dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mt_metrix.config import DatasetConfig
from mt_metrix.io.schema import Segment

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-mapping helper
# ---------------------------------------------------------------------------

def _resolve_column(row: dict[str, Any], spec: str | None) -> Any:
    """Resolve one column value from a row, honouring @constant / @from prefixes.

    - ``None`` → field is left as default.
    - ``"@constant:X"`` → literal string ``X`` for every row.
    - ``"@from:colname"`` → value of ``colname`` in the row.
    - bare string → treated as a column name.
    """
    if spec is None:
        return None
    if isinstance(spec, str) and spec.startswith("@constant:"):
        return spec[len("@constant:"):]
    if isinstance(spec, str) and spec.startswith("@from:"):
        col = spec[len("@from:"):]
        return row.get(col)
    # default: treat as column name
    return row.get(spec)


def _row_to_segment(
    row: dict[str, Any],
    columns: dict[str, str],
    idx: int,
    default_lang_pair: str,
    default_domain: str,
) -> Segment:
    source = _resolve_column(row, columns.get("source"))
    target = _resolve_column(row, columns.get("target"))
    if not source or not target:
        raise ValueError(
            f"row {idx} missing source or target (after column mapping): {row}"
        )

    reference = _resolve_column(row, columns.get("reference"))
    gold_raw = _resolve_column(row, columns.get("gold"))
    gold: float | None = None
    if gold_raw is not None:
        try:
            gold = float(gold_raw)
        except (TypeError, ValueError):
            gold = None

    lang_pair = _resolve_column(row, columns.get("lang_pair")) or default_lang_pair
    domain = _resolve_column(row, columns.get("domain")) or default_domain
    segment_id_raw = _resolve_column(row, columns.get("segment_id"))
    segment_id = str(segment_id_raw) if segment_id_raw is not None else f"seg_{idx:08d}"

    return Segment(
        source=str(source),
        target=str(target),
        reference=str(reference) if reference else None,
        gold=gold,
        lang_pair=str(lang_pair),
        domain=str(domain),
        segment_id=segment_id,
        meta=row,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_local(cfg: DatasetConfig) -> list[Segment]:
    import pandas as pd

    path = Path(cfg.params["path"]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported local file suffix {suffix!r}")

    # Optional row limit
    limit = cfg.params.get("limit")
    if limit is not None:
        df = df.head(int(limit))

    default_lang = cfg.params.get("lang_pair", "")
    default_domain = cfg.params.get("domain", "general")
    rows = df.to_dict(orient="records")
    return [
        _row_to_segment(r, cfg.columns, i, default_lang, default_domain)
        for i, r in enumerate(rows)
    ]


def _load_huggingface(cfg: DatasetConfig) -> list[Segment]:
    """Load one or more subsets from a HuggingFace dataset repo.

    Accepts either ``config`` (single subset name) or ``configs`` (list of
    subset names to concatenate). Multi-subset is how we build domain-wide
    matrices from repos like ``surrey-nlp/Legal-QE``, which ship one subset
    per language pair (``en-gujarati``, ``en-tamil``, ``en-telugu``).
    ``limit`` caps the total segment count after concatenation.
    """
    from datasets import load_dataset

    repo = cfg.params["repo"]
    single = cfg.params.get("config")
    multi = cfg.params.get("configs")
    if single is not None and multi is not None:
        raise ValueError(
            "dataset config cannot set both 'config' (single subset) and "
            "'configs' (list of subsets) — pick one"
        )
    if multi is not None and not isinstance(multi, (list, tuple)):
        raise ValueError(
            f"'configs' must be a list of subset names, got {type(multi).__name__}"
        )

    split = cfg.params.get("split", "test")
    cache_dir = cfg.params.get("cache_dir")
    limit = cfg.params.get("limit")
    default_lang = cfg.params.get("lang_pair", "")
    default_domain = cfg.params.get("domain", "general")

    subsets: list[str | None] = list(multi) if multi is not None else [single]

    segs: list[Segment] = []
    idx = 0
    for subset in subsets:
        log.info("loading HF dataset %s (config=%s, split=%s)", repo, subset, split)
        kwargs: dict[str, Any] = {"split": split}
        if subset is not None:
            kwargs["name"] = subset
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir

        ds = load_dataset(repo, **kwargs)
        for row in ds:
            segs.append(
                _row_to_segment(dict(row), cfg.columns, idx, default_lang, default_domain)
            )
            idx += 1
            if limit is not None and idx >= int(limit):
                return segs
    return segs


def _load_gyroqe(cfg: DatasetConfig) -> list[Segment]:
    """Adapter for gyroQE's unified catalogue.

    Expects gyroQE's unified TSV outputs at
    ``{gyroqe_root}/data/processed/{year}/{lang_pair}/{split}.tsv`` — the
    schema documented in gyroQE's ``data/DATA_CATALOGUE.md``.
    """
    gyroqe_root = Path(cfg.params["path"]).expanduser().resolve()
    year = cfg.params.get("year", "mlqe-pe")
    lang_pair = cfg.params["lang_pair"]
    split = cfg.params.get("split", "test")

    candidates = [
        gyroqe_root / "data" / "processed" / year / lang_pair / f"{split}.tsv",
        gyroqe_root / "data" / "processed" / year / f"{lang_pair}_{split}.tsv",
    ]
    for path in candidates:
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            f"could not find gyroQE processed data under any of: "
            f"{[str(c) for c in candidates]}"
        )

    # Default column mapping for gyroQE's unified schema
    defaults = {
        "source": "source",
        "target": "target",
        "reference": "reference",
        "gold": "z_mean",
        "lang_pair": "lang_pair",
        "domain": "domain",
    }
    columns = {**defaults, **(cfg.columns or {})}

    # Reuse the local TSV path with the patched columns
    local_cfg = DatasetConfig(
        kind="local",
        params={
            "path": str(path),
            "lang_pair": lang_pair,
            "domain": cfg.params.get("domain", "general"),
            "limit": cfg.params.get("limit"),
        },
        columns=columns,
    )
    return _load_local(local_cfg)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    "local": _load_local,
    "huggingface": _load_huggingface,
    "gyroqe": _load_gyroqe,
}


def load_dataset_from_config(cfg: DatasetConfig) -> list[Segment]:
    """Dispatch to the right loader for ``cfg.kind``."""
    if cfg.kind not in _LOADERS:
        raise ValueError(
            f"unknown dataset kind {cfg.kind!r}; expected one of {sorted(_LOADERS)}"
        )
    return _LOADERS[cfg.kind](cfg)
