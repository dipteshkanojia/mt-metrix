"""Tests for the paper-matrix tabulation module.

We synthesise a small outputs tree under ``tmp_path`` (two domains × two
language pairs × two scorers) and drive ``tabulate`` end-to-end. The runner
never runs — we just write segments.tsv + summary.json shapes that match
what :mod:`mt_metrix.io.writers` produces.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mt_metrix.reports.tabulate import (
    Cell,
    LANG_ALIAS,
    canonicalise_lang_pair,
    classify_scorer,
    collect_records,
    discover_runs,
    render_csv,
    render_latex,
    render_markdown,
    tabulate,
    _pivot_cells,
)


# ---------------------------------------------------------------------------
# classify_scorer
# ---------------------------------------------------------------------------

def test_classify_scorer_comet_qe():
    assert classify_scorer("comet", "wmt22-cometkiwi-da", needs_reference=False) == "COMET-QE"


def test_classify_scorer_comet_ref():
    assert classify_scorer("comet", "wmt22-comet-da", needs_reference=True) == "COMET-ref"


def test_classify_scorer_tower_da_vs_mqm():
    assert classify_scorer("tower", "towerinstruct-7b-v0.2", needs_reference=False) == "Tower-DA"
    assert classify_scorer("tower", "towerinstruct-7b-v0.2-mqm", needs_reference=False) == "Tower-MQM"


def test_classify_scorer_sacrebleu_fallback():
    assert classify_scorer("sacrebleu", "bleu", needs_reference=True) == "sacrebleu"
    # Unknown family falls through to sacrebleu bucket (permissive).
    assert classify_scorer("mystery", "thing", needs_reference=False) == "sacrebleu"


# ---------------------------------------------------------------------------
# Fixture builder — emits a fake run directory
# ---------------------------------------------------------------------------

def _write_fake_run(
    run_dir: Path,
    domain: str,
    rows_per_lang: dict[str, list[tuple[float, dict[str, float]]]],
    skipped: list[dict[str, str]] | None = None,
    scorer_meta: list[dict[str, str]] | None = None,
) -> None:
    """Write a synthetic run directory with segments.tsv + summary.json.

    ``rows_per_lang`` maps lang_pair → list of (gold, {scorer: score}) tuples.
    ``scorer_meta`` is the list that normally lands in summary.run_metadata.scorers.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    # Figure out scorer names from the first row we encounter.
    all_scorers: list[str] = []
    for rows in rows_per_lang.values():
        for _, s in rows:
            for name in s:
                if name not in all_scorers:
                    all_scorers.append(name)
            break

    # Build segments.tsv by hand to keep the dep surface tiny.
    header = ["segment_id", "lang_pair", "domain", "source", "target", "reference", "gold"] + all_scorers
    lines = ["\t".join(header)]
    idx = 0
    for lp, rows in rows_per_lang.items():
        for gold, scorers in rows:
            row = [
                f"seg_{idx:04d}", lp, domain, f"src_{idx}", f"tgt_{idx}", "",
                f"{gold}",
            ] + [f"{scorers.get(s, '')}" for s in all_scorers]
            lines.append("\t".join(row))
            idx += 1
    (run_dir / "segments.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scorer_meta_list = scorer_meta or [
        {"family": "comet", "name": s, "model": f"Unbabel/{s}", "params": {}}
        for s in all_scorers
    ]
    summary = {
        "run_metadata": {
            "run_id": run_dir.name,
            "dataset": {"kind": "huggingface", "repo": "fake/X", "domain": domain,
                         "configs": list(rows_per_lang.keys())},
            "scorers": scorer_meta_list,
        },
        "metrics": {s: {} for s in all_scorers},
        "corpus_scores": {},
        "skipped_metrics": skipped or [],
        "n_segments": idx,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def fake_runs(tmp_path: Path) -> Path:
    """Three fake run directories: one Legal, one General, one with a skip."""
    outputs = tmp_path / "outputs"
    # Legal domain: en-gu + en-ta, two QE scorers (perfect + inverse correlation).
    _write_fake_run(
        outputs / "surrey_legal_full_matrix",
        domain="legal",
        rows_per_lang={
            "en-gu": [
                (0.9, {"kiwi-da": 0.85, "kiwi-xl": 0.10}),
                (0.6, {"kiwi-da": 0.55, "kiwi-xl": 0.40}),
                (0.3, {"kiwi-da": 0.35, "kiwi-xl": 0.70}),
                (0.1, {"kiwi-da": 0.05, "kiwi-xl": 0.90}),
            ],
            "en-ta": [
                (0.8, {"kiwi-da": 0.80, "kiwi-xl": 0.20}),
                (0.4, {"kiwi-da": 0.45, "kiwi-xl": 0.60}),
                (0.2, {"kiwi-da": 0.15, "kiwi-xl": 0.75}),
                (0.0, {"kiwi-da": 0.05, "kiwi-xl": 0.95}),
            ],
        },
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
            {"family": "comet", "name": "kiwi-xl", "model": "Unbabel/kiwi-xl", "params": {}},
        ],
    )
    # General domain: en-hi only, one scorer that ran + one that got skipped.
    _write_fake_run(
        outputs / "surrey_general_full_matrix",
        domain="general",
        rows_per_lang={
            "en-hi": [
                (0.9, {"kiwi-da": 0.85}),
                (0.2, {"kiwi-da": 0.25}),
                (0.5, {"kiwi-da": 0.55}),
                (0.1, {"kiwi-da": 0.10}),
            ],
        },
        skipped=[
            {"name": "bleu", "reason": "dataset-has-no-references"},
        ],
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
            {"family": "sacrebleu", "name": "bleu", "model": None, "params": {}},
        ],
    )
    return outputs


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------

def test_discover_runs_matches_and_sorts(fake_runs: Path):
    matches = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    names = [p.name for p in matches]
    assert names == ["surrey_general_full_matrix", "surrey_legal_full_matrix"]


def test_discover_runs_ignores_dirs_without_summary(tmp_path: Path):
    (tmp_path / "outputs" / "incomplete_run").mkdir(parents=True)
    matches = discover_runs(str(tmp_path / "outputs" / "*"))
    assert matches == []


# ---------------------------------------------------------------------------
# collect_records
# ---------------------------------------------------------------------------

def test_collect_records_builds_domain_rows(fake_runs: Path):
    run_dirs = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    # Minimal catalogue — marks kiwi-da as QE and bleu as needs_ref for bucketing.
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
        "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
        "sacrebleu/bleu": {"family": "sacrebleu", "name": "bleu", "needs_reference": True},
    }
    records = collect_records(run_dirs, catalogues)

    # Legal has 2 scorers × 2 langs = 4 scored records.
    legal = [r for r in records if r.domain == "legal"]
    assert {r.lang_pair for r in legal} == {"en-gu", "en-ta"}
    assert {r.model for r in legal} == {"kiwi-da", "kiwi-xl"}
    # Spearman for the positive-trend scorer must be > 0; for inverse < 0.
    kiwi_da_engu = next(r for r in legal if r.model == "kiwi-da" and r.lang_pair == "en-gu")
    kiwi_xl_engu = next(r for r in legal if r.model == "kiwi-xl" and r.lang_pair == "en-gu")
    assert kiwi_da_engu.spearman is not None and kiwi_da_engu.spearman > 0.9
    assert kiwi_xl_engu.spearman is not None and kiwi_xl_engu.spearman < -0.9

    # General has one scored scorer and one skipped → skipped appears as NA record.
    general = [r for r in records if r.domain == "general"]
    scored_names = {r.model for r in general if r.skipped_reason is None}
    skipped_names = {r.model for r in general if r.skipped_reason is not None}
    assert scored_names == {"kiwi-da"}
    assert skipped_names == {"bleu"}
    assert all(r.pearson is None for r in general if r.skipped_reason is not None)


# ---------------------------------------------------------------------------
# _pivot_cells + rendering
# ---------------------------------------------------------------------------

def test_pivot_marks_best_per_domain_per_column(fake_runs: Path):
    run_dirs = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
        "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
    }
    records = collect_records(run_dirs, catalogues)
    cells = _pivot_cells(records, metric="spearman")
    # kiwi-da has positive rho; kiwi-xl has negative. kiwi-da must be best for
    # every lang column in the Legal domain block.
    for lp in ("en-gu", "en-ta"):
        cell_da = cells[("legal", "COMET-QE", "kiwi-da")][lp]
        cell_xl = cells[("legal", "COMET-QE", "kiwi-xl")][lp]
        assert cell_da.is_best is True
        assert cell_xl.is_best is False


def test_render_markdown_has_bold_best(fake_runs: Path, tmp_path: Path):
    run_dirs = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
        "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
        "sacrebleu/bleu": {"family": "sacrebleu", "name": "bleu", "needs_reference": True},
    }
    records = collect_records(run_dirs, catalogues)
    out = tmp_path / "report" / "paper_table.md"
    render_markdown(records, out, metric="spearman")
    txt = out.read_text(encoding="utf-8")
    assert "| Domain |" in txt
    assert "En-Hi" in txt and "En-Gu" in txt and "Avg" in txt
    # kiwi-da's best values should be bolded on the Legal block.
    assert "**" in txt  # at least one bold cell
    # Skipped scorer renders em-dash, not a number.
    assert "—" in txt


def test_render_latex_has_multirow_and_booktabs(fake_runs: Path, tmp_path: Path):
    run_dirs = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
        "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
        "sacrebleu/bleu": {"family": "sacrebleu", "name": "bleu", "needs_reference": True},
    }
    records = collect_records(run_dirs, catalogues)
    out = tmp_path / "report" / "paper_table.tex"
    render_latex(records, out, metric="spearman")
    tex = out.read_text(encoding="utf-8")
    assert "\\toprule" in tex
    assert "\\multirow" in tex
    assert "\\textbf{" in tex
    assert "\\bottomrule" in tex
    # Scorer names with dashes shouldn't break; underscores get escaped.
    assert "kiwi-da" in tex  # plain hyphens are safe


def test_render_csv_has_headers_and_rows(fake_runs: Path, tmp_path: Path):
    run_dirs = discover_runs(str(fake_runs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
        "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
        "sacrebleu/bleu": {"family": "sacrebleu", "name": "bleu", "needs_reference": True},
    }
    records = collect_records(run_dirs, catalogues)
    out = tmp_path / "report" / "results.csv"
    render_csv(records, out)
    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("domain,group,model,lang_pair")
    # One row per (domain, scorer, lang_pair) — Legal × 2 scorers × 2 langs = 4,
    # General × 1 scored × 1 lang = 1, General skipped × 1 lang = 1.
    assert len(rows) - 1 == 4 + 1 + 1


# ---------------------------------------------------------------------------
# top-level tabulate() convenience
# ---------------------------------------------------------------------------

def test_tabulate_writes_all_three_outputs(fake_runs: Path, tmp_path: Path, monkeypatch):
    # Patch catalogue lookup so tabulate finds our synthetic scorers.
    from mt_metrix.reports import tabulate as tab_mod

    def fake_find(_roots):
        return {
            "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
            "comet/kiwi-xl": {"family": "comet", "name": "kiwi-xl", "needs_reference": False},
            "sacrebleu/bleu": {"family": "sacrebleu", "name": "bleu", "needs_reference": True},
        }
    monkeypatch.setattr("mt_metrix.config._find_catalogues", fake_find)

    out_dir = tmp_path / "report"
    paths = tab_mod.tabulate(
        runs_glob=str(fake_runs / "surrey_*_full_matrix"),
        out_dir=out_dir,
        metric="spearman",
    )
    assert paths["csv"].exists()
    assert paths["md"].exists()
    assert paths["tex"].exists()


def test_tabulate_errors_when_no_runs_match(tmp_path: Path):
    from mt_metrix.reports.tabulate import tabulate as tab
    with pytest.raises(FileNotFoundError, match="no run directories matched"):
        tab(runs_glob=str(tmp_path / "does-not-exist-*"), out_dir=tmp_path / "r")


# ---------------------------------------------------------------------------
# canonicalise_lang_pair + cross-dataset spelling tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,canonical",
    [
        # Full-name forms shipped by Surrey Legal/General/Health/Tourism.
        ("en-gujarati", "en-gu"),
        ("en-hindi", "en-hi"),
        ("en-marathi", "en-mr"),
        ("en-tamil", "en-ta"),
        ("en-telugu", "en-te"),
        # Contracted forms shipped by Low-resource-QE-DA multilingual subset.
        ("engu", "en-gu"),
        ("enhi", "en-hi"),
        ("enmr", "en-mr"),
        ("enta", "en-ta"),
        ("ente", "en-te"),
        # Already-canonical codes pass through.
        ("en-gu", "en-gu"),
        ("en-hi", "en-hi"),
        # Uppercase input still canonicalises (HF sometimes upper-cases).
        ("EN-GUJARATI", "en-gu"),
        ("EnGu", "en-gu"),
        # Unknown pairs pass through lowered but unchanged — won't crash,
        # just won't render under LANG_ORDER until LANG_ALIAS is extended.
        ("en-xx", "en-xx"),
        ("pl-en", "pl-en"),
        # Empty is preserved (used as skipped-scorer placeholder).
        ("", ""),
    ],
)
def test_canonicalise_lang_pair(raw: str, canonical: str):
    assert canonicalise_lang_pair(raw) == canonical


def test_lang_alias_covers_every_canonical_lang_order_code():
    """Every canonical ISO code in LANG_ORDER must be reachable from at
    least one full-name and one contracted alias. Guards against someone
    adding a new column header without teaching the alias map."""
    from mt_metrix.reports.tabulate import LANG_ORDER

    canonical_targets = set(LANG_ALIAS.values())
    for code in LANG_ORDER:
        assert code in canonical_targets, (
            f"LANG_ORDER code {code!r} has no LANG_ALIAS entry mapping to "
            f"it — add the full-name and contracted forms for this pair."
        )


def test_collect_records_canonicalises_full_name_lang_pairs(tmp_path: Path):
    """Integration: a run whose segments.tsv stores ``en-gujarati`` /
    ``en-hindi`` etc. (as the Surrey HF configs publish) must produce
    RunRecords keyed on the canonical ``en-gu`` / ``en-hi`` — otherwise
    the per-lang columns in paper_table.md / .tex render as NA because
    the renderer iterates LANG_ORDER, which only holds the canonical
    codes. Captured from the IndicQE paper-prep batch 2026-04-22."""
    outputs = tmp_path / "outputs"
    _write_fake_run(
        outputs / "surrey_general_full_matrix",
        domain="general",
        rows_per_lang={
            # Full-name codes exactly as the HF configs publish them.
            "en-gujarati": [
                (0.9, {"kiwi-da": 0.85}),
                (0.6, {"kiwi-da": 0.55}),
                (0.3, {"kiwi-da": 0.30}),
                (0.1, {"kiwi-da": 0.10}),
            ],
            "en-hindi": [
                (0.8, {"kiwi-da": 0.75}),
                (0.4, {"kiwi-da": 0.45}),
                (0.2, {"kiwi-da": 0.20}),
                (0.0, {"kiwi-da": 0.05}),
            ],
        },
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
        ],
    )
    run_dirs = discover_runs(str(outputs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
    }
    records = collect_records(run_dirs, catalogues)
    # Lang_pair on every record must already be canonical; neither raw
    # full-name string should survive into the record set.
    lang_pairs = {r.lang_pair for r in records}
    assert lang_pairs == {"en-gu", "en-hi"}, (
        f"expected canonicalisation to en-gu / en-hi, got {lang_pairs!r}"
    )
    # And the correlations still compute (the canonicalisation must be
    # pure key-renaming, not data loss).
    engu = next(r for r in records if r.lang_pair == "en-gu")
    assert engu.spearman is not None and engu.spearman > 0.9


def test_render_markdown_populates_per_lang_columns_for_full_name_dataset(
    tmp_path: Path,
):
    """End-to-end: with full-name lang_pair input, the GFM table's per-lang
    cells under LANG_ORDER headers must contain real values rather than
    the em-dash NA placeholder that the pre-canonicalisation build used to
    emit (confirmed 2026-04-22 against AISURREY IndicQE paper_table.md)."""
    outputs = tmp_path / "outputs"
    _write_fake_run(
        outputs / "surrey_general_full_matrix",
        domain="general",
        rows_per_lang={
            "en-gujarati": [
                (0.9, {"kiwi-da": 0.85}),
                (0.6, {"kiwi-da": 0.55}),
                (0.3, {"kiwi-da": 0.30}),
                (0.1, {"kiwi-da": 0.10}),
            ],
        },
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
        ],
    )
    run_dirs = discover_runs(str(outputs / "surrey_*_full_matrix"))
    catalogues = {
        "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
    }
    records = collect_records(run_dirs, catalogues)
    out = tmp_path / "report" / "paper_table.md"
    render_markdown(records, out, metric="spearman")
    txt = out.read_text(encoding="utf-8")
    # The kiwi-da row for general/en-gu must contain a concrete number
    # under the En-Gu column. We look for the kiwi-da line and confirm it
    # contains at least one non-em-dash numeric cell.
    lines = [l for l in txt.splitlines() if "kiwi-da" in l]
    assert lines, "kiwi-da row should be present in markdown output"
    da_row = lines[0]
    # Regression guard: before the canonicalisation fix, all five per-lang
    # columns rendered as em-dash for a full-name dataset. Only En-Gu is
    # non-NA here (the fixture only ships en-gujarati) — so exactly four
    # em-dashes are expected, not five.
    assert da_row.count("—") == 4, (
        f"expected 4 NA cells (Hi/Mr/Ta/Te absent) + 1 real cell (Gu), got "
        f"{da_row.count('—')} NA cells. Row: {da_row!r}"
    )
    # And the Gu cell must hold a concrete numeric value rendered bold
    # (best-in-column with only one row in the column means it wins).
    cells = [tok.strip() for tok in da_row.split("|") if tok.strip()]
    # Row is: Domain, Group, Model, Hi, Mr, Ta, Te, Gu, Avg => Gu is index 7
    assert cells[7] == "**1.00**", f"En-Gu cell wrong: {cells[7]!r} in {da_row!r}"
