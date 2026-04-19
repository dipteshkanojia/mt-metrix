"""Config loading, validation, and include-resolution.

Config shape (run-level YAML)::

    run:
      id: my_run              # optional; auto-generated if absent

    dataset: !include ../datasets/foo.yaml
    # or inline:
    dataset:
      kind: huggingface
      repo: surrey-nlp/Legal-QE
      split: test
      columns: {...}

    scorers:
      - ref: comet/wmt22-cometkiwi-da       # lookup from configs/models/comet.yaml
        overrides:                           # optional
          batch_size: 32
      - ref: sacrebleu/chrf
      - family: tower                        # inline, no catalogue lookup
        name: tower-gemba-da-7b
        model: Unbabel/TowerInstruct-7B-v0.2
        params:
          prompt_mode: gemba-da
          tensor_parallel_size: 1

    output:
      root: outputs
      formats: [tsv, jsonl, summary]

The config loader resolves ``!include`` directives and ``ref:`` lookups
into a single :class:`RunConfig` object. The resolved config is written
to ``outputs/<run_id>/config.yaml`` so runs are reproducible from their
own output directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mt_metrix.scorers.base import ScorerConfig


# ---------------------------------------------------------------------------
# YAML loader with !include support
# ---------------------------------------------------------------------------

class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader extension that supports ``!include path/to/other.yaml``.

    Includes are resolved relative to the file that contains them. Included
    files may themselves use ``!include``.
    """


def _include_constructor(loader: _IncludeLoader, node: yaml.Node) -> Any:
    raw = loader.construct_scalar(node)
    base = Path(loader.name).parent if getattr(loader, "name", None) else Path.cwd()
    target = (base / raw).resolve()
    return _load_yaml_with_includes(target)


_IncludeLoader.add_constructor("!include", _include_constructor)


def _load_yaml_with_includes(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        loader = _IncludeLoader(f)
        loader.name = str(path)  # type: ignore[attr-defined]
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for a dataset loader.

    ``kind`` picks the adapter: ``local``, ``huggingface``, or ``gyroqe``.
    ``columns`` maps source-column names in the raw data to mt-metrix's
    unified field names. Values starting with ``@constant:`` are treated as
    literal values for every row; values starting with ``@from:`` copy from
    another column.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        columns = data.pop("columns", {}) or {}
        kind = data.pop("kind", None)
        if kind is None:
            raise ValueError("dataset config missing required field 'kind'")
        # Everything else goes into params so adapters can consume their
        # specific flags (repo, path, split, config, name, …) without this
        # loader needing to know them.
        return cls(kind=kind, params=data, columns=columns)


@dataclass
class OutputConfig:
    root: str = "outputs"
    formats: list[str] = field(default_factory=lambda: ["tsv", "jsonl", "summary"])


@dataclass
class RunConfig:
    run_id: str
    dataset: DatasetConfig
    scorers: list[ScorerConfig]
    output: OutputConfig
    raw: dict[str, Any] = field(default_factory=dict)  # original dict for snapshot

    def output_dir(self) -> Path:
        return Path(self.output.root) / self.run_id


# ---------------------------------------------------------------------------
# Model catalogue resolution (ref: lookups)
# ---------------------------------------------------------------------------

def _load_catalogue(path: Path) -> dict[str, dict[str, Any]]:
    """Load a catalogue YAML into a flat dict keyed by ``<family>/<name>``.

    Catalogue files look like::

        family: comet
        models:
          wmt22-comet-da:
            model: Unbabel/wmt22-comet-da
            needs_reference: true
            params: {batch_size: 64}
          wmt22-cometkiwi-da:
            model: Unbabel/wmt22-cometkiwi-da
            needs_reference: false
            params: {batch_size: 64}
    """
    data = _load_yaml_with_includes(path)
    family = data["family"]
    out: dict[str, dict[str, Any]] = {}
    for name, entry in data.get("models", {}).items():
        key = f"{family}/{name}"
        out[key] = {"family": family, "name": name, **entry}
    return out


def _find_catalogues(search_roots: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for root in search_roots:
        models_dir = root / "configs" / "models"
        if not models_dir.is_dir():
            continue
        for path in sorted(models_dir.glob("*.yaml")):
            merged.update(_load_catalogue(path))
    return merged


def _resolve_scorer_entry(
    entry: dict[str, Any],
    catalogues: dict[str, dict[str, Any]],
) -> ScorerConfig:
    """Resolve one scorer entry from a run config into a :class:`ScorerConfig`.

    Entry may be:
    - ``{ref: <family>/<name>, overrides: {...}}`` — catalogue lookup + overrides
    - ``{family: ..., name: ..., model: ..., params: {...}}`` — fully inline
    """
    if "ref" in entry:
        key = entry["ref"]
        if key not in catalogues:
            known = ", ".join(sorted(catalogues)) or "<no catalogues loaded>"
            raise KeyError(f"Unknown scorer ref {key!r}. Known refs: {known}")
        base = dict(catalogues[key])  # shallow copy
        # merge overrides into params
        overrides = entry.get("overrides") or {}
        params = {**(base.get("params") or {}), **overrides}
        # Promote catalogue top-level flags that scorers read from params. Without
        # this, a catalogue entry like `needs_reference: false` at the entry's
        # top level (the documented schema) would be silently dropped, and COMET
        # scorers would fall back to the model-id heuristic — which
        # misclassified `Unbabel/XCOMET-XL` (QE mode) and `wmt20-comet-qe-da`
        # during the 2026-04-19 full-matrix run.
        for flag in ("needs_reference",):
            if flag in base and flag not in overrides:
                params.setdefault(flag, base[flag])
        return ScorerConfig(
            family=base["family"],
            name=base["name"],
            model=base.get("model"),
            params=params,
        )

    # inline
    required = {"family", "name"}
    if not required.issubset(entry.keys()):
        raise ValueError(
            f"Inline scorer entry must have {required}, got {set(entry.keys())}"
        )
    return ScorerConfig(
        family=entry["family"],
        name=entry["name"],
        model=entry.get("model"),
        params=entry.get("params") or {},
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-")


def _auto_run_id(dataset: DatasetConfig, scorers: list[ScorerConfig]) -> str:
    ds_short = _slugify(
        dataset.params.get("name")
        or dataset.params.get("repo")
        or dataset.params.get("path")
        or dataset.kind
    )[:40]
    scorer_short = _slugify(scorers[0].name if scorers else "noop")[:30]
    if len(scorers) > 1:
        scorer_short = f"{scorer_short}-plus{len(scorers) - 1}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ds_short}_{scorer_short}_{ts}"


def load_run_config(
    config_path: str | Path,
    catalogue_roots: list[Path] | None = None,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Load and resolve a run config file into a :class:`RunConfig`.

    Parameters
    ----------
    config_path:
        Path to a run YAML.
    catalogue_roots:
        Roots to search for ``configs/models/*.yaml`` catalogues. Defaults to
        the current working directory and the installed package's parent
        (so both local development and installed-package usage work).
    overrides:
        List of ``dotted.key=value`` strings to splat over the top-level
        config before resolution (e.g. from ``--override`` CLI flags).
    """
    path = Path(config_path).resolve()
    raw = _load_yaml_with_includes(path) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must be key=value")
        key, val = ov.split("=", 1)
        _set_dotted(raw, key, _coerce(val))

    if catalogue_roots is None:
        catalogue_roots = [path.parent.parent.parent, Path.cwd()]
    catalogues = _find_catalogues(catalogue_roots)

    # dataset
    ds_raw = raw.get("dataset")
    if ds_raw is None:
        raise ValueError("run config missing required 'dataset' section")
    dataset = DatasetConfig.from_dict(dict(ds_raw))

    # scorers
    scorer_entries = raw.get("scorers") or []
    if not scorer_entries:
        raise ValueError("run config must list at least one scorer under 'scorers'")
    scorers = [_resolve_scorer_entry(dict(e), catalogues) for e in scorer_entries]

    # output
    out_raw = raw.get("output") or {}
    output = OutputConfig(
        root=out_raw.get("root", "outputs"),
        formats=list(out_raw.get("formats") or ["tsv", "jsonl", "summary"]),
    )

    run_id = (raw.get("run") or {}).get("id") or _auto_run_id(dataset, scorers)

    return RunConfig(
        run_id=run_id,
        dataset=dataset,
        scorers=scorers,
        output=output,
        raw=raw,
    )


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = value


def _coerce(value: str) -> Any:
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def dump_resolved_config(run: RunConfig, path: str | Path) -> None:
    """Write the resolved run config to ``path`` as plain YAML (no includes)."""
    resolved = {
        "run": {"id": run.run_id},
        "dataset": {"kind": run.dataset.kind, **run.dataset.params, "columns": run.dataset.columns},
        "scorers": [
            {
                "family": s.family,
                "name": s.name,
                "model": s.model,
                "params": s.params,
            }
            for s in run.scorers
        ],
        "output": {"root": run.output.root, "formats": run.output.formats},
    }
    Path(path).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
