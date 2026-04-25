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


# ---------------------------------------------------------------------------
# Bug 1 — duplicate (domain, group, model, lang_pair) across runs must not
# let a later skipped/errored retry stomp the earlier successful record.
# ---------------------------------------------------------------------------

def test_pivot_cells_successful_record_wins_over_skipped_retry():
    """Observed 2026-04-25: a retry run emitted skipped_reason=runtime: [Errno 22]
    for the 4 COMET-QE scorers that had run successfully in a prior run. The
    final paper_table.md showed em-dashes for those rows because _pivot_cells
    did setdefault(...)[lang] = Cell(...), which let the later skipped record
    overwrite the good one. Regardless of record order, a Cell with a real
    value must beat a skipped placeholder."""
    from mt_metrix.reports.tabulate import RunRecord, _pivot_cells

    good = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=0.62, spearman=0.58, kendall=0.41, spa=0.70, n=1000,
        skipped_reason=None,
    )
    bad_retry = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=None, spearman=None, kendall=None, spa=None, n=0,
        skipped_reason="runtime: [Errno 22] Invalid argument",
    )

    # Order 1: success first, then skipped retry → the bug case.
    cells = _pivot_cells([good, bad_retry], metric="spearman")
    cell = cells[("lowres_qe", "COMET-QE", "wmt22-cometkiwi-da")]["en-gu"]
    assert cell.value == 0.58, (
        f"success-then-skipped: good value must survive, got value={cell.value}, "
        f"skipped={cell.skipped}"
    )
    assert cell.skipped is False

    # Order 2: skipped first, then success → must still resolve to success.
    # (defensive — run discovery order is filesystem-sorted, so either order
    # is possible depending on the retry's timestamp suffix.)
    cells = _pivot_cells([bad_retry, good], metric="spearman")
    cell = cells[("lowres_qe", "COMET-QE", "wmt22-cometkiwi-da")]["en-gu"]
    assert cell.value == 0.58
    assert cell.skipped is False


def test_pivot_cells_two_successful_records_last_wins():
    """If BOTH records have real values (e.g. two legitimate runs of the same
    scorer on the same lang_pair), the later one wins — ``_pivot_cells`` has
    no way to know which run is "correct" and overwriting in insertion order
    keeps behaviour deterministic for callers who sort run dirs by timestamp."""
    from mt_metrix.reports.tabulate import RunRecord, _pivot_cells

    earlier = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=0.50, spearman=0.45, kendall=0.30, spa=0.65, n=800,
        skipped_reason=None,
    )
    later = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=0.62, spearman=0.58, kendall=0.41, spa=0.70, n=1000,
        skipped_reason=None,
    )
    cells = _pivot_cells([earlier, later], metric="spearman")
    cell = cells[("lowres_qe", "COMET-QE", "wmt22-cometkiwi-da")]["en-gu"]
    assert cell.value == 0.58, "later successful record must overwrite earlier"


def test_pivot_cells_two_skipped_last_wins():
    """When both records are skipped, there's no 'success' to protect —
    standard last-wins applies so the caller sees the most recent reason."""
    from mt_metrix.reports.tabulate import RunRecord, _pivot_cells

    first_skip = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=None, spearman=None, kendall=None, spa=None, n=0,
        skipped_reason="gated-weights-need-hf-accept",
    )
    second_skip = RunRecord(
        domain="lowres_qe", group="COMET-QE", model="wmt22-cometkiwi-da",
        lang_pair="en-gu",
        pearson=None, spearman=None, kendall=None, spa=None, n=0,
        skipped_reason="runtime: [Errno 22] Invalid argument",
    )
    cells = _pivot_cells([first_skip, second_skip], metric="spearman")
    cell = cells[("lowres_qe", "COMET-QE", "wmt22-cometkiwi-da")]["en-gu"]
    assert cell.skipped is True


# ---------------------------------------------------------------------------
# Bug 2 — _ordered_rows fallback for unknown domains must still respect
# GROUP_ORDER. Otherwise the paper_table renders the same group (e.g.
# COMET-QE) in two separate blocks interleaved with other groups.
# ---------------------------------------------------------------------------

def test_ordered_rows_groups_unknown_domain_by_group_order():
    """Observed 2026-04-25: with domain='lowres_qe' (not in DOMAIN_ORDER),
    the fallback branch of _ordered_rows appended records in pure insertion
    order, so COMET-QE rows from RUN1 rendered first, Tower-DA rows from
    RUN1 rendered next, then a second COMET-QE block from RUN3. The table
    showed two disjoint COMET-QE sections. Rows within an unknown domain
    must still cluster by GROUP_ORDER so each group appears in exactly one
    contiguous block."""
    from mt_metrix.reports.tabulate import RunRecord, _ordered_rows

    # Insertion order deliberately interleaves groups: COMET-QE, Tower-DA,
    # COMET-QE again (as happens when RUN3 contributes new COMET-QE models
    # AFTER RUN1 already contributed Tower-DA models).
    def _rec(group: str, model: str) -> RunRecord:
        return RunRecord(
            domain="lowres_qe", group=group, model=model, lang_pair="en-gu",
            pearson=0.5, spearman=0.5, kendall=0.3, spa=0.6, n=100,
            skipped_reason=None,
        )
    records = [
        _rec("COMET-QE", "wmt22-cometkiwi-da"),
        _rec("COMET-QE", "wmt23-cometkiwi-da-xl"),
        _rec("Tower-DA", "tower-plus-2b"),
        _rec("Tower-DA", "tower-plus-9b"),
        _rec("COMET-QE", "wmt23-cometkiwi-da-xxl"),  # came in late
        _rec("sacrebleu", "bleu"),
    ]
    row_keys = _ordered_rows(records)
    # Extract the group column for each key, in order.
    groups_in_order = [grp for (_dom, grp, _mdl) in row_keys]
    # All COMET-QE rows must be adjacent — in a single contiguous block,
    # not two separated by Tower-DA.
    first_comet = groups_in_order.index("COMET-QE")
    last_comet = len(groups_in_order) - 1 - list(reversed(groups_in_order)).index("COMET-QE")
    comet_slice = groups_in_order[first_comet:last_comet + 1]
    assert all(g == "COMET-QE" for g in comet_slice), (
        f"COMET-QE rows must be contiguous; got group sequence {groups_in_order}"
    )
    # And the whole sequence must follow GROUP_ORDER: COMET-QE before
    # Tower-DA before sacrebleu.
    unique_groups_in_order = []
    for g in groups_in_order:
        if g not in unique_groups_in_order:
            unique_groups_in_order.append(g)
    assert unique_groups_in_order == ["COMET-QE", "Tower-DA", "sacrebleu"], (
        f"expected GROUP_ORDER-respecting sequence, got {unique_groups_in_order}"
    )


def test_ordered_rows_sorts_models_alphabetically_within_unknown_domain_group():
    """Models within a (domain, group) block must be sorted for stability —
    the same property the DOMAIN_ORDER fast path already enforces. Without
    sorting, two CI runs processing the same records in different discovery
    orders would emit tables with different row orders."""
    from mt_metrix.reports.tabulate import RunRecord, _ordered_rows

    def _rec(model: str) -> RunRecord:
        return RunRecord(
            domain="lowres_qe", group="Tower-DA", model=model, lang_pair="en-gu",
            pearson=0.5, spearman=0.5, kendall=0.3, spa=0.6, n=100,
            skipped_reason=None,
        )
    # Insertion order: z first, then a, then m — must come out a, m, z.
    records = [_rec("z-model"), _rec("a-model"), _rec("m-model")]
    row_keys = _ordered_rows(records)
    models = [m for (_d, _g, m) in row_keys]
    assert models == ["a-model", "m-model", "z-model"], (
        f"expected alphabetical model order, got {models}"
    )


def test_ordered_rows_known_domain_still_respects_domain_and_group_order():
    """Regression guard: the Bug-2 fix must not regress the DOMAIN_ORDER
    fast path. A mix of 'legal' (known) and 'lowres_qe' (unknown) must
    still put 'legal' first, and within each domain groups must follow
    GROUP_ORDER."""
    from mt_metrix.reports.tabulate import RunRecord, _ordered_rows

    records = [
        # lowres_qe first in insertion order, but legal must come first in output.
        RunRecord(
            domain="lowres_qe", group="COMET-QE", model="m1",
            lang_pair="en-gu", pearson=0.5, spearman=0.5, kendall=0.3,
            spa=0.6, n=100, skipped_reason=None,
        ),
        RunRecord(
            domain="legal", group="Tower-DA", model="m2",
            lang_pair="en-gu", pearson=0.5, spearman=0.5, kendall=0.3,
            spa=0.6, n=100, skipped_reason=None,
        ),
        RunRecord(
            domain="legal", group="COMET-QE", model="m3",
            lang_pair="en-gu", pearson=0.5, spearman=0.5, kendall=0.3,
            spa=0.6, n=100, skipped_reason=None,
        ),
    ]
    row_keys = _ordered_rows(records)
    # Legal block must come before lowres_qe block.
    domains = [d for (d, _g, _m) in row_keys]
    first_lowres = domains.index("lowres_qe")
    last_legal = len(domains) - 1 - list(reversed(domains)).index("legal")
    assert last_legal < first_lowres, (
        f"legal block must precede lowres_qe block; got {domains}"
    )
    # Within the legal block, COMET-QE must precede Tower-DA.
    legal_groups = [
        g for (d, g, _m) in row_keys if d == "legal"
    ]
    assert legal_groups.index("COMET-QE") < legal_groups.index("Tower-DA")


# ---------------------------------------------------------------------------
# Bug 3 — the Avg column must average over the SAME language pairs that
# appear as columns in the rendered table (LANG_ORDER). Observed 2026-04-25:
# when a dataset ships extra pairs outside LANG_ORDER (eten / neen / sien
# from Low-resource-QE-DA), Avg silently included them, so a row whose five
# displayed Indic cells mean-to 0.23 would show Avg=0.30.
# ---------------------------------------------------------------------------

def test_row_avg_only_averages_over_lang_order_columns():
    """Avg must mirror the visible column set, not the union of language
    pairs across records. Otherwise a row reads as '0.18 0.19 0.33 0.23
    0.23 | 0.29' where the final cell looks like it should be the mean of
    the preceding five."""
    from mt_metrix.reports.tabulate import Cell, LANG_ORDER, _row_avg

    # Build a row with ALL LANG_ORDER pairs plus three extras. The five
    # LANG_ORDER values average to 0.20. The three extras are much higher;
    # if the bug is live, their contribution drags the reported Avg up.
    row_cells: dict[str, Cell] = {lp: Cell(value=0.20) for lp in LANG_ORDER}
    # Extras outside LANG_ORDER must not affect the Avg.
    row_cells["eten"] = Cell(value=0.90)
    row_cells["neen"] = Cell(value=0.80)
    row_cells["sien"] = Cell(value=0.70)
    avg = _row_avg(row_cells)
    assert avg is not None
    assert abs(avg - 0.20) < 1e-9, (
        f"Avg must equal mean of LANG_ORDER cells (0.20); got {avg}. "
        f"If bug is live, Avg will be higher because eten/neen/sien are "
        f"being included."
    )


def test_row_avg_ignores_skipped_lang_order_cells():
    """Within LANG_ORDER, skipped cells still shouldn't count — matches
    the pre-bug-3 behaviour that was already right for skipped handling."""
    from mt_metrix.reports.tabulate import Cell, _row_avg

    row_cells: dict[str, Cell] = {
        "en-hi": Cell(value=0.10),
        "en-gu": Cell(value=0.30),
        # en-mr skipped — must not be averaged in.
        "en-mr": Cell(value=None, skipped=True),
    }
    avg = _row_avg(row_cells)
    assert avg is not None
    assert abs(avg - 0.20) < 1e-9  # (0.10 + 0.30) / 2


def test_row_avg_returns_none_when_no_lang_order_values():
    """If the row has values only in pairs outside LANG_ORDER (pathological
    case — the domain shipped no Indic pairs at all), Avg must return None
    rather than hiding the gap behind an average over unrelated pairs."""
    from mt_metrix.reports.tabulate import Cell, _row_avg

    row_cells: dict[str, Cell] = {
        "eten": Cell(value=0.50),
        "neen": Cell(value=0.60),
    }
    assert _row_avg(row_cells) is None


def test_render_markdown_avg_matches_visible_columns():
    """End-to-end guard: in the rendered markdown, the Avg cell must equal
    the mean of the five visible lang cells (computed from the same rounded
    display values). Reproduces the 2026-04-25 surprise where a row with
    displayed cells '0.18 0.19 0.33 0.23 0.23' showed Avg=0.29."""
    from mt_metrix.reports.tabulate import Cell, LANG_ORDER, _fmt_md_avg, _row_avg

    # Row with values that would average differently if extras are included.
    row_cells: dict[str, Cell] = {
        "en-hi": Cell(value=0.18),
        "en-mr": Cell(value=0.19),
        "en-ta": Cell(value=0.33),
        "en-te": Cell(value=0.23),
        "en-gu": Cell(value=0.23),
        # Low-resource-QE-DA also ships these; they must not pollute Avg.
        "eten": Cell(value=0.51),
        "neen": Cell(value=0.36),
        "sien": Cell(value=0.32),
    }
    # Under the fix, avg should be (0.18+0.19+0.33+0.23+0.23)/5 = 0.232
    avg = _row_avg(row_cells)
    assert avg is not None
    assert abs(avg - 0.232) < 1e-9, (
        f"Avg over LANG_ORDER only should be 0.232; got {avg}"
    )
    # And the two-decimal display rounds to 0.23, not 0.29 (the buggy value).
    assert _fmt_md_avg(avg) == "0.23"


# ---------------------------------------------------------------------------
# Integration guard — the live 2026-04-25 scenario: an original successful
# run + a retry run that re-submitted the same scorers and failed. Before
# the three fixes, paper_table.md showed em-dashes for the retried scorers,
# two disjoint COMET-QE blocks, and Avg values inflated by Low-resource-QE
# extras.
# ---------------------------------------------------------------------------

def test_tabulate_end_to_end_2026_04_25_scenario(
    tmp_path: Path, monkeypatch,
):
    outputs = tmp_path / "outputs"
    # RUN1: original successful run — kiwi-da on en-gu, en-hi, eten.
    _write_fake_run(
        outputs / "surrey_lowres_qe_20260420",
        domain="lowres_qe",
        rows_per_lang={
            "en-gu": [
                (0.9, {"kiwi-da": 0.85}), (0.6, {"kiwi-da": 0.55}),
                (0.3, {"kiwi-da": 0.30}), (0.1, {"kiwi-da": 0.10}),
            ],
            "en-hi": [
                (0.8, {"kiwi-da": 0.75}), (0.4, {"kiwi-da": 0.45}),
                (0.2, {"kiwi-da": 0.20}), (0.0, {"kiwi-da": 0.05}),
            ],
            "eten": [
                (0.9, {"kiwi-da": 0.90}), (0.5, {"kiwi-da": 0.50}),
                (0.3, {"kiwi-da": 0.30}), (0.1, {"kiwi-da": 0.10}),
            ],
        },
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
        ],
    )
    # RUN2: retry that RE-submitted kiwi-da but errored. The scorer
    # appears in skipped_metrics with a runtime reason. Without the
    # fix, this stomps RUN1's real values.
    _write_fake_run(
        outputs / "surrey_lowres_qe_20260425_retry",
        domain="lowres_qe",
        rows_per_lang={
            # The retry still writes segments.tsv but with no pred values
            # (empty cells); _per_lang_correlations will see n<2 and skip.
            "en-gu": [(0.9, {}), (0.6, {}), (0.3, {}), (0.1, {})],
            "en-hi": [(0.8, {}), (0.4, {}), (0.2, {}), (0.0, {})],
        },
        skipped=[
            {"name": "kiwi-da", "reason": "runtime: [Errno 22] Invalid argument"},
        ],
        scorer_meta=[
            {"family": "comet", "name": "kiwi-da", "model": "Unbabel/kiwi-da", "params": {}},
        ],
    )
    # RUN3: a third run that adds a second COMET-QE scorer (kiwi-xxl),
    # mirroring the real scenario where the paper-prep batch ran the
    # XXL variant separately. Before Bug 2's fix, this would render in
    # its own disjoint COMET-QE block beneath the unrelated Tower-DA
    # rows.
    _write_fake_run(
        outputs / "surrey_lowres_qe_xxl",
        domain="lowres_qe",
        rows_per_lang={
            "en-gu": [
                (0.9, {"kiwi-xxl": 0.88}), (0.6, {"kiwi-xxl": 0.60}),
                (0.3, {"kiwi-xxl": 0.35}), (0.1, {"kiwi-xxl": 0.15}),
            ],
        },
        scorer_meta=[
            {"family": "comet", "name": "kiwi-xxl", "model": "Unbabel/kiwi-xxl", "params": {}},
        ],
    )

    def fake_find(_roots):
        return {
            "comet/kiwi-da": {"family": "comet", "name": "kiwi-da", "needs_reference": False},
            "comet/kiwi-xxl": {"family": "comet", "name": "kiwi-xxl", "needs_reference": False},
        }
    monkeypatch.setattr("mt_metrix.config._find_catalogues", fake_find)

    out_dir = tmp_path / "report"
    tabulate(
        runs_glob=str(outputs / "surrey_lowres_qe_*"),
        out_dir=out_dir,
        metric="spearman",
    )

    md = (out_dir / "paper_table.md").read_text(encoding="utf-8")

    # Bug 1: kiwi-da's row must NOT be all em-dashes — RUN1's successes
    # survive RUN2's skipped retry.
    kiwi_da_lines = [l for l in md.splitlines() if "kiwi-da" in l]
    assert len(kiwi_da_lines) == 1, (
        f"kiwi-da must render exactly once, not in two blocks; got "
        f"{len(kiwi_da_lines)} lines:\n" + "\n".join(kiwi_da_lines)
    )
    kiwi_da_row = kiwi_da_lines[0]
    assert "1.00" in kiwi_da_row or "0.90" in kiwi_da_row or "**" in kiwi_da_row, (
        f"kiwi-da row must hold real values (RUN1 data), not all em-dashes. "
        f"Row: {kiwi_da_row!r}"
    )

    # Bug 2: COMET-QE group must appear only once. The word "COMET-QE"
    # shows up in the kiwi-da row (group column = "COMET-QE") and then
    # gets elided (empty string) for kiwi-xxl's row since they share a
    # (domain, group) bucket. Two distinct "COMET-QE" labels would mean
    # the group rendered in two disjoint blocks.
    comet_qe_label_count = sum(
        1 for line in md.splitlines()
        if "| COMET-QE |" in line
    )
    assert comet_qe_label_count == 1, (
        f"COMET-QE group label must appear exactly once (subsequent models "
        f"in the same group inherit via elision); got {comet_qe_label_count} "
        f"occurrences — table:\n{md}"
    )

    # Bug 3: Avg column must match the mean of the visible lang cells
    # (LANG_ORDER), not inflated by the eten pair in RUN1. kiwi-da has
    # values in en-gu (1.00 spearman on a 4-point perfect trend) and
    # en-hi (1.00 spearman) only; Avg over visible LANG_ORDER should be
    # 1.00 — NOT the mean of en-gu/en-hi/eten.
    # Parse the kiwi-da row's cells.
    cells = [tok.strip() for tok in kiwi_da_row.split("|") if tok.strip()]
    # Row is: Domain, Group, Model, Hi, Mr, Ta, Te, Gu, Avg — Avg at index 8.
    avg_cell = cells[8]
    # en-hi (0.75..0.05 trend) and en-gu (0.85..0.10) are both perfect
    # monotonic with the gold, so their spearman = 1.00 each. Mean = 1.00.
    # With the bug live, eten (also perfect → 1.0) would be included and
    # the result would still read 1.00 — so in this fixture the numeric
    # equality doesn't discriminate. What DOES discriminate is the count
    # of NA cells: en-mr/en-ta/en-te are genuinely absent and must be
    # em-dashes, NOT suppressed because Avg pretended coverage was wider.
    assert avg_cell in {"1.00", "**1.00**"}, (
        f"kiwi-da Avg cell must be 1.00 (mean over LANG_ORDER-only); got "
        f"{avg_cell!r} in row {kiwi_da_row!r}"
    )
    # And the three missing Indic pairs must render as em-dash.
    na_count = kiwi_da_row.count("—")
    assert na_count == 3, (
        f"kiwi-da row should show em-dashes for the three missing Indic "
        f"pairs (Mr/Ta/Te); got {na_count} — row: {kiwi_da_row!r}"
    )
