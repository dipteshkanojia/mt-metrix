"""Aggregate many mt-metrix runs into a paper-ready domain × language matrix.

Inputs
------
One or more ``outputs/<run_id>/`` directories, each containing
``segments.tsv`` (required) and ``summary.json`` (required). The runner
writes these unconditionally; no special run flag is needed.

Outputs
-------
- ``results.csv``    — long-form CSV (one row per (domain, scorer, lang_pair))
- ``paper_table.md`` — GitHub-flavoured markdown, flat header, NA cells
- ``paper_table.tex``— booktabs LaTeX, hierarchical rows via ``\\multirow``

Shape of the output matrix
--------------------------
Rows: domain (Legal / General / Health / Tourism) × scorer-group × scorer name.
Columns: the five Indic language pairs used across the four Surrey datasets
(``En-Gu``, ``En-Hi``, ``En-Mr``, ``En-Ta``, ``En-Te``) plus ``Avg``.

Scorer grouping (affects row order and ``\\midrule`` separators):

- ``COMET-QE``   — ``family=comet`` + ``needs_reference=false``
- ``COMET-ref``  — ``family=comet`` + ``needs_reference=true``
- ``Tower-DA``   — ``family=tower`` + name does NOT end ``-mqm``
- ``Tower-MQM``  — ``family=tower`` + name ends ``-mqm``
- ``sacrebleu``  — ``family=sacrebleu``

NA handling:

- Domain doesn't ship a given language pair (e.g. Legal has no en-hi) → NA
  cell; the corresponding long-form row is omitted from ``results.csv``.
- Scorer ended up in ``skipped_metrics`` (gated weights, no ref column, OOM)
  → NA across every lang column for that scorer, inside that domain only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grouping / labelling
# ---------------------------------------------------------------------------

GROUP_ORDER = ["COMET-QE", "COMET-ref", "Tower-DA", "Tower-MQM", "sacrebleu"]
DOMAIN_ORDER = ["Legal", "General", "Health", "Tourism"]
LANG_ORDER = ["en-hi", "en-mr", "en-ta", "en-te", "en-gu"]

# Pretty-printed labels used as column / row headers.
LANG_LABEL = {
    "en-hi": "En-Hi",
    "en-mr": "En-Mr",
    "en-ta": "En-Ta",
    "en-te": "En-Te",
    "en-gu": "En-Gu",
}
DOMAIN_LABEL = {
    "legal": "Legal",
    "general": "General",
    "health": "Health",
    "tourism": "Tourism",
}

# Canonicalise lang_pair codes that come out of heterogeneous datasets.
# Surrey Legal/General/Health/Tourism publish full-name hyphenated forms
# (``en-gujarati``, ``en-hindi``, ``en-marathi``, ``en-tamil``, ``en-telugu``);
# Low-resource-QE-DA's ``multilingual`` subset publishes contracted glued
# forms (``engu``, ``enhi``, …). Both must land on the paper's canonical
# ISO-639-1 hyphenated form (``en-gu``, ``en-hi``, …) BEFORE pivoting, or the
# per-language columns in paper_table.md / .tex render as NA even though the
# underlying correlations computed just fine. This is the "cross-dataset
# lang_pair codes differ" gotcha flagged under IndicQE paper scope in
# ``docs/INDIC_QE_DATASETS.md`` — fixed here at the tabulate ingestion
# boundary so downstream consumers don't need to remember.
LANG_ALIAS: dict[str, str] = {
    # Full-name forms (Surrey Legal/General/Health/Tourism HF configs).
    "en-gujarati": "en-gu",
    "en-hindi": "en-hi",
    "en-marathi": "en-mr",
    "en-tamil": "en-ta",
    "en-telugu": "en-te",
    # Contracted glued forms (Low-resource-QE-DA multilingual subset).
    "engu": "en-gu",
    "enhi": "en-hi",
    "enmr": "en-mr",
    "enta": "en-ta",
    "ente": "en-te",
}


def canonicalise_lang_pair(lp: str) -> str:
    """Map any known lang_pair spelling onto its canonical ISO form.

    Accepts anything that appears in the Surrey dataset family (full-name
    ``en-gujarati``, contracted ``engu``, canonical ``en-gu``). Unknown
    values pass through lowered but otherwise unchanged — a new pair added
    at the dataset level won't crash tabulation, it just won't land in
    ``LANG_ORDER`` and will only show up in the ``Avg`` column until
    ``LANG_ALIAS`` is taught about it.
    """
    if not lp:
        return lp
    key = str(lp).lower()
    return LANG_ALIAS.get(key, key)


def classify_scorer(family: str, name: str, needs_reference: bool | None) -> str:
    """Return the five-way group bucket for (family, name, needs_reference).

    Families not in the known list fall through to ``"sacrebleu"`` as a
    permissive default — caller should inspect and decide if stricter
    handling is required.
    """
    if family == "comet":
        return "COMET-QE" if not needs_reference else "COMET-ref"
    if family == "tower":
        return "Tower-MQM" if name.endswith("-mqm") else "Tower-DA"
    return "sacrebleu"


# ---------------------------------------------------------------------------
# Run discovery + per-run aggregation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRecord:
    """One scorer × one lang_pair × one domain, with correlations."""

    domain: str
    group: str
    model: str            # scorer name from segments.tsv column
    lang_pair: str
    pearson: float | None
    spearman: float | None
    kendall: float | None
    spa: float | None
    n: int
    skipped_reason: str | None = None  # set when the scorer entered skipped_metrics


def discover_runs(runs_glob: str) -> list[Path]:
    """Expand ``runs_glob`` to a sorted list of run directories.

    The glob is matched against the filesystem directly (not ``Path.glob``'s
    relative semantics) so callers can pass absolute paths containing shell
    wildcards, e.g. ``/mnt/fast/.../outputs/surrey_*_full_matrix``.
    """
    from glob import glob

    hits = [Path(p) for p in glob(runs_glob)]
    return sorted(p for p in hits if p.is_dir() and (p / "summary.json").is_file())


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _per_lang_correlations(run_dir: Path, scorer_names: list[str]) -> dict[tuple[str, str], dict[str, float | None]]:
    """Return {(scorer_name, lang_pair): {pearson, spearman, kendall, spa, n}}.

    Pivots ``segments.tsv`` on ``lang_pair`` and invokes
    :func:`mt_metrix.io.writers._correlations` per group. Scorer columns with
    fewer than two usable (non-NaN, gold-present) rows in a given lang_pair
    produce ``n=0`` and all-None correlations — callers render those as NA.
    """
    import pandas as pd

    from mt_metrix.io.writers import _correlations

    tsv_path = run_dir / "segments.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    out: dict[tuple[str, str], dict[str, float | None]] = {}
    if "gold" not in df.columns or "lang_pair" not in df.columns:
        return out

    for lp, sub_df in df.groupby("lang_pair", sort=False):
        # Canonicalise at ingestion: any LANG_ALIAS entry (en-gujarati,
        # engu, …) lands on en-gu before it's used as a dict key, so
        # downstream pivot/render code sees only canonical ISO codes and
        # the LANG_ORDER lookup in render_{markdown,latex} hits.
        lp_canon = canonicalise_lang_pair(str(lp))
        sub_gold = pd.to_numeric(sub_df["gold"], errors="coerce")
        for col in scorer_names:
            if col not in sub_df.columns:
                continue
            pred = pd.to_numeric(sub_df[col], errors="coerce")
            mask = pred.notna() & sub_gold.notna()
            if mask.sum() < 2:
                out[(col, lp_canon)] = {
                    "pearson": None, "spearman": None, "kendall": None,
                    "spa": None, "n": int(mask.sum()),
                }
                continue
            out[(col, lp_canon)] = _correlations(
                pred[mask].to_numpy(), sub_gold[mask].to_numpy()
            )
    return out


def collect_records(
    run_dirs: Iterable[Path],
    catalogues: dict[str, dict[str, Any]],
) -> list[RunRecord]:
    """Walk each run dir and emit one RunRecord per (scorer, lang_pair).

    Also emits NA-placeholder records for scorers listed in
    ``skipped_metrics`` so the table shows them with a reason rather than
    silently dropping them.
    """
    records: list[RunRecord] = []
    # Build a helper lookup: {scorer_name → (family, needs_reference)} from the
    # catalogues. Catalogue keys are ``family/name`` but scorer columns in
    # segments.tsv / summary.json use ``name`` alone.
    name_to_catalogue: dict[str, dict[str, Any]] = {}
    for key, entry in catalogues.items():
        name_to_catalogue[entry["name"]] = entry

    for run_dir in run_dirs:
        summary = _load_summary(run_dir)
        ds_md = summary.get("run_metadata", {}).get("dataset", {})
        domain = str(ds_md.get("domain") or "").lower()
        if not domain:
            log.warning("run %s has no dataset.domain in summary.json — skipping", run_dir)
            continue
        scorer_md = summary.get("run_metadata", {}).get("scorers", []) or []
        # Map scorer name → family (from the scorer list in metadata, fall back
        # to the catalogue if name collides across families we still pick the
        # one from metadata since that's what actually ran).
        name_to_family: dict[str, str] = {s["name"]: s["family"] for s in scorer_md}

        skipped = {
            s["name"]: s.get("reason", "") for s in summary.get("skipped_metrics", []) or []
        }

        # Scorers that DID run — those in `metrics` and not in `skipped`.
        scored_names = [n for n in (summary.get("metrics") or {}).keys() if n not in skipped]
        per_lang = _per_lang_correlations(run_dir, scored_names) if scored_names else {}

        # Emit records for scored scorers × each lang_pair that showed up.
        present_langs: set[str] = set()
        for (_, lp) in per_lang.keys():
            present_langs.add(lp)

        # For each scorer (scored or skipped), emit records.
        for scorer_name in list(scored_names) + list(skipped.keys()):
            family = name_to_family.get(scorer_name, "sacrebleu")
            needs_ref = (name_to_catalogue.get(scorer_name) or {}).get("needs_reference")
            group = classify_scorer(family, scorer_name, bool(needs_ref))
            reason = skipped.get(scorer_name)

            if scorer_name in skipped:
                # Emit one NA record per lang_pair that was present in the data,
                # so the table has a row to render under this scorer.
                for lp in sorted(present_langs) or [""]:
                    records.append(RunRecord(
                        domain=domain, group=group, model=scorer_name,
                        lang_pair=lp,
                        pearson=None, spearman=None, kendall=None, spa=None, n=0,
                        skipped_reason=reason,
                    ))
            else:
                for lp in sorted(present_langs):
                    corr = per_lang.get((scorer_name, lp))
                    if corr is None:
                        continue
                    records.append(RunRecord(
                        domain=domain, group=group, model=scorer_name,
                        lang_pair=lp,
                        pearson=corr.get("pearson"),
                        spearman=corr.get("spearman"),
                        kendall=corr.get("kendall"),
                        spa=corr.get("spa"),
                        n=int(corr.get("n") or 0),
                        skipped_reason=None,
                    ))
    return records


# ---------------------------------------------------------------------------
# Rendering — CSV (long form)
# ---------------------------------------------------------------------------

def render_csv(records: list[RunRecord], out_path: Path) -> None:
    """Write long-form CSV. NA (domain×lang absent) rows are omitted; skipped
    scorers keep their rows with empty correlation cells + ``skipped_reason``."""
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain", "group", "model", "lang_pair",
            "pearson", "spearman", "kendall", "spa", "n", "skipped_reason",
        ])
        for r in records:
            writer.writerow([
                r.domain, r.group, r.model, r.lang_pair,
                "" if r.pearson is None else f"{r.pearson:.6f}",
                "" if r.spearman is None else f"{r.spearman:.6f}",
                "" if r.kendall is None else f"{r.kendall:.6f}",
                "" if r.spa is None else f"{r.spa:.6f}",
                r.n,
                r.skipped_reason or "",
            ])


# ---------------------------------------------------------------------------
# Pivoting helper shared by markdown + LaTeX renderers
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    value: float | None         # the correlation value for the chosen metric
    is_best: bool = False       # best-in-column for this domain block
    skipped: bool = False       # row comes from skipped_metrics — render NA


def _ordered_rows(records: list[RunRecord]) -> list[tuple[str, str, str]]:
    """Enumerate (domain, group, model) row keys in the table's canonical order."""
    seen: set[tuple[str, str, str]] = set()
    row_keys: list[tuple[str, str, str]] = []
    # First: respect the fixed domain + group order.
    for dom in DOMAIN_ORDER:
        for grp in GROUP_ORDER:
            models_in_bucket = sorted({
                r.model for r in records
                if r.domain.lower() == dom.lower() and r.group == grp
            })
            for m in models_in_bucket:
                key = (dom.lower(), grp, m)
                if key not in seen:
                    seen.add(key)
                    row_keys.append(key)
    # Then: anything with a domain we don't recognise, appended in stable order.
    for r in records:
        key = (r.domain.lower(), r.group, r.model)
        if key not in seen:
            seen.add(key)
            row_keys.append(key)
    return row_keys


def _pivot_cells(
    records: list[RunRecord], metric: str,
) -> dict[tuple[str, str, str], dict[str, Cell]]:
    """Build {(domain, group, model): {lang_pair: Cell}}.

    ``metric`` picks which correlation column to surface in the Cell value;
    must be one of ``pearson`` / ``spearman`` / ``kendall`` / ``spa``.
    """
    if metric not in {"pearson", "spearman", "kendall", "spa"}:
        raise ValueError(f"unknown metric {metric!r}")
    cells: dict[tuple[str, str, str], dict[str, Cell]] = {}
    for r in records:
        key = (r.domain.lower(), r.group, r.model)
        value = getattr(r, metric)
        cells.setdefault(key, {})[r.lang_pair] = Cell(
            value=value,
            skipped=(r.skipped_reason is not None),
        )
    # Best-in-column computation, per domain block.
    # For each (domain, lang_pair), find max(value) over all rows in that block
    # and mark those cells is_best=True (ties → all tied cells marked).
    by_domain: dict[str, list[tuple[str, str, str]]] = {}
    for (dom, grp, mdl) in cells.keys():
        by_domain.setdefault(dom, []).append((dom, grp, mdl))
    for dom, keys in by_domain.items():
        # gather all lang pairs seen in this domain block
        langs = sorted({lp for k in keys for lp in cells[k].keys()})
        for lp in langs:
            best_val: float | None = None
            best_keys: list[tuple[str, str, str]] = []
            for k in keys:
                cell = cells[k].get(lp)
                if cell is None or cell.value is None or cell.skipped:
                    continue
                if best_val is None or cell.value > best_val:
                    best_val = cell.value
                    best_keys = [k]
                elif cell.value == best_val:
                    best_keys.append(k)
            for k in best_keys:
                cells[k][lp].is_best = True
    return cells


def _row_avg(row_cells: dict[str, Cell]) -> float | None:
    """Mean over non-NA values in the row, or None if nothing usable."""
    vals = [c.value for c in row_cells.values() if c.value is not None and not c.skipped]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Rendering — Markdown (flat header)
# ---------------------------------------------------------------------------

def render_markdown(
    records: list[RunRecord], out_path: Path, metric: str = "spearman",
) -> None:
    """GFM markdown table with flat header.

    Columns are ``Domain | Group | Model | En-Hi | En-Mr | En-Ta | En-Te |
    En-Gu | Avg``. Best-ρ cells are rendered ``**x.xx**``; NA cells as
    ``—`` (em-dash). Group separators use an extra blank row since GFM
    doesn't support ``\\midrule``.
    """
    cells = _pivot_cells(records, metric)
    row_keys = _ordered_rows(records)
    if not row_keys:
        out_path.write_text("(no records)\n", encoding="utf-8")
        return

    header = ["Domain", "Group", "Model"] + [LANG_LABEL[l] for l in LANG_ORDER] + ["Avg"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"<!-- metric: {metric} (higher is better); bold = best-in-column per domain -->\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        prev_domain: str | None = None
        prev_group: str | None = None
        for (dom, grp, mdl) in row_keys:
            if prev_domain is not None and dom != prev_domain:
                f.write("|" + "|".join([""] * len(header)) + "|\n")
            row_cells = cells.get((dom, grp, mdl), {})
            avg = _row_avg(row_cells)
            dom_lbl = DOMAIN_LABEL.get(dom, dom.title()) if dom != prev_domain else ""
            grp_lbl = grp if (dom, grp) != (prev_domain, prev_group) else ""
            f.write("| " + " | ".join([
                dom_lbl, grp_lbl, mdl,
                *[_fmt_md_cell(row_cells.get(lp)) for lp in LANG_ORDER],
                _fmt_md_avg(avg),
            ]) + " |\n")
            prev_domain, prev_group = dom, grp


def _fmt_md_cell(cell: Cell | None) -> str:
    if cell is None or cell.value is None or cell.skipped:
        return "—"
    v = f"{cell.value:.2f}"
    return f"**{v}**" if cell.is_best else v


def _fmt_md_avg(avg: float | None) -> str:
    return "—" if avg is None else f"{avg:.2f}"


# ---------------------------------------------------------------------------
# Rendering — LaTeX (booktabs, \multirow, \midrule)
# ---------------------------------------------------------------------------

def render_latex(
    records: list[RunRecord], out_path: Path, metric: str = "spearman",
) -> None:
    """booktabs LaTeX table with hierarchical rows via ``\\multirow``.

    Requires the ``booktabs`` and ``multirow`` packages in the preamble.
    Best-ρ cells wrapped in ``\\textbf{}``; NA cells render as ``--``.
    Rule policy: ``\\midrule`` between groups within a domain; ``\\midrule``
    between domains; ``\\cmidrule(lr)`` not used here — kept deliberately
    simple for the paper template.
    """
    cells = _pivot_cells(records, metric)
    row_keys = _ordered_rows(records)
    lang_headers = [LANG_LABEL[l] for l in LANG_ORDER]
    col_spec = "l l l " + " ".join(["c"] * (len(LANG_ORDER) + 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-count rows per (domain, group) so \multirow row-spans come out right.
    dom_count: dict[str, int] = {}
    grp_count: dict[tuple[str, str], int] = {}
    for (dom, grp, _mdl) in row_keys:
        dom_count[dom] = dom_count.get(dom, 0) + 1
        grp_count[(dom, grp)] = grp_count.get((dom, grp), 0) + 1

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"% Auto-generated by mt-metrix tabulate (metric = {metric}, higher is better).\n")
        f.write("% Bold cells are per-domain, per-language best of the chosen metric.\n")
        f.write("% Requires \\usepackage{booktabs,multirow} in your preamble.\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
        f.write(" & ".join(
            ["Domain", "Group", "Model"] + lang_headers + ["Avg"]
        ) + " \\\\\n\\midrule\n")

        prev_domain: str | None = None
        prev_group: str | None = None
        for i, (dom, grp, mdl) in enumerate(row_keys):
            row_cells = cells.get((dom, grp, mdl), {})
            avg = _row_avg(row_cells)
            # Domain cell (only on first row of domain block).
            if dom != prev_domain:
                if prev_domain is not None:
                    f.write("\\midrule\n")
                dom_cell = f"\\multirow{{{dom_count[dom]}}}{{*}}{{\\rotatebox{{90}}{{{DOMAIN_LABEL.get(dom, dom.title())}}}}}"
            else:
                dom_cell = ""
            # Group cell (only on first row of (domain, group) block).
            if (dom, grp) != (prev_domain, prev_group):
                # Separator between groups within same domain.
                if dom == prev_domain:
                    f.write("\\cmidrule(l){2-" + str(3 + len(LANG_ORDER) + 1) + "}\n")
                grp_cell = f"\\multirow{{{grp_count[(dom, grp)]}}}{{*}}{{{grp}}}"
            else:
                grp_cell = ""
            row_tex = [
                dom_cell, grp_cell, _tex_escape(mdl),
                *[_fmt_tex_cell(row_cells.get(lp)) for lp in LANG_ORDER],
                _fmt_tex_avg(avg),
            ]
            f.write(" & ".join(row_tex) + " \\\\\n")
            prev_domain, prev_group = dom, grp

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{Segment-level correlation ({metric}) of QE metrics against human z-scores across Surrey domain-specific Indic QE datasets.}}\n")
        f.write("\\label{tab:surrey_domain_matrix}\n\\end{table}\n")


def _fmt_tex_cell(cell: Cell | None) -> str:
    if cell is None or cell.value is None or cell.skipped:
        return "--"
    v = f"{cell.value:.2f}"
    return f"\\textbf{{{v}}}" if cell.is_best else v


def _fmt_tex_avg(avg: float | None) -> str:
    return "--" if avg is None else f"{avg:.2f}"


def _tex_escape(s: str) -> str:
    """Minimal LaTeX escape for scorer names (underscores / percent / ampersand)."""
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}"))


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def tabulate(
    runs_glob: str,
    out_dir: Path,
    metric: str = "spearman",
    catalogue_roots: list[Path] | None = None,
) -> dict[str, Path]:
    """Walk ``runs_glob``, write ``results.csv``, ``paper_table.md`` and
    ``paper_table.tex`` under ``out_dir``. Returns the three written paths."""
    from mt_metrix.config import _find_catalogues

    if catalogue_roots is None:
        catalogue_roots = [Path.cwd(), Path(__file__).resolve().parents[3]]
    catalogues = _find_catalogues(catalogue_roots)

    run_dirs = discover_runs(runs_glob)
    if not run_dirs:
        raise FileNotFoundError(
            f"no run directories matched {runs_glob!r} "
            "(each dir must contain a summary.json)"
        )
    log.info("tabulate: %d runs matched %s", len(run_dirs), runs_glob)

    records = collect_records(run_dirs, catalogues)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "paper_table.md"
    tex_path = out_dir / "paper_table.tex"
    render_csv(records, csv_path)
    render_markdown(records, md_path, metric=metric)
    render_latex(records, tex_path, metric=metric)
    return {"csv": csv_path, "md": md_path, "tex": tex_path}
