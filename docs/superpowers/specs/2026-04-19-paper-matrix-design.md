# Paper-style Matrix Tabulation — Design

**Status:** Draft  |  **Date:** 2026-04-19  |  **Target:** mt-metrix main

## Summary

Extend mt-metrix so that one matrix job per domain scores every scorer in the
catalogue across all the domain's language pairs in a single pass (loading each
model exactly once), and a new `mt-metrix tabulate` subcommand aggregates those
runs into a single paper-ready table — Pearson (r) and Spearman (ρ) with NA
cells for missing (domain, lang_pair) combinations, matching the layout of
Table 8 in the ALOPE paper (arXiv:2603.07372).

## Motivation

The intended downstream use is an extension of ALOPE: comparing every QE
scorer in the catalogue (CometKiwi-{DA,XL,XXL}, XCOMET-XL/XXL-QE,
COMET-QE family, Tower-GEMBA-{DA,MQM}) against LoRA-adapted ALOPE layers
across four domains (Legal, General, Healthcare, Tourism) and five Indic
language pairs (en-gu, en-hi, en-mr, en-ta, en-te). Running one model per
job across one lang pair at a time would compound per-model load time by
~15× on the large Tower and XCOMET checkpoints; fusing the subsets keeps
loads amortised.

## Scope

**In scope:**

- Dataset loader: accept a list of HF subsets and concatenate into one
  segments list, preserving the per-row `lang_pair` column.
- `mt-metrix tabulate` subcommand producing LaTeX (booktabs), Markdown,
  and long-form CSV from one or more run directories.
- Row grouping by scorer family (COMET-QE / COMET-ref / Tower-DA /
  Tower-MQM / sacrebleu) with horizontal rules.
- Per-(domain, scorer, lang_pair) Pearson (r) and Spearman (ρ), plus
  row-wise Avg across available lang pairs.
- Best-ρ-per-column bolding within each domain block.
- NA cells for (domain, lang_pair) combinations that don't exist in the
  dataset, or scorers that failed during a run.

**Out of scope:**

- Layer-wise ALOPE adaptation (that belongs to the ALOPE-QE project).
- Bootstrap CIs on correlations (can be added later; not in the paper
  reference table).
- Kendall's τ in the table (still computed per segment, but not rendered
  — keeps column count manageable).
- New dataset families beyond the four Surrey domains already present.

## Architecture

Three changes, each isolated:

```
+----------------------+     +--------------------+     +------------------+
| multi-subset loader  | --> | existing runner    | --> | tabulate command |
| (configs: [list])    |     | (one model at a    |     | (reads run       |
|                      |     |  time, unchanged)  |     |  outputs)        |
+----------------------+     +--------------------+     +------------------+
        |                              |                          |
        v                              v                          v
   configs/datasets/*.yaml      segments.tsv                paper_table.tex
   (opt. `configs:` list)       + summary.json             paper_table.md
                                 per run                   results.csv
```

Each piece is independently testable; the runner is unchanged. The existing
`scripts/submit_aisurrey.sh` keeps submitting four jobs (one per domain)
and need not change.

## Detailed design

### 1. Multi-subset dataset loader

**Config surface.** `configs/datasets/surrey_legal.yaml` becomes:

```yaml
kind: huggingface
repo: surrey-nlp/Legal-QE
configs: [en-gujarati, en-tamil, en-telugu]   # new: list
# config: en-gujarati                          # old: single string still works
split: test
domain: legal
columns:
  source: source_text
  target: target_text
  gold: z_mean
  lang_pair: "@from:language_pair"
  domain: "@constant:legal"
```

**Loader behaviour** (`src/mt_metrix/io/datasets.py::_load_huggingface`):

- If `configs` is a list: call `load_dataset(repo, name=c, split=split)` for
  each entry, concatenate row-wise into one segments list. The order in the
  list determines the row order in `segments.tsv`.
- If `config` (singular) is still present: keep current single-subset behaviour.
- Having both is an error ("use `configs:` list form only").
- `limit`, if set, is applied AFTER concatenation (so you get a cross-lang
  sample, not just the first lang).
- `cache_dir` is passed through unchanged.

**Per-row lang_pair** comes from `@from:language_pair`, which works as-is:
each HF subset's rows carry their lang pair in that column.

### 2. Matrix runs (unchanged)

All four `configs/runs/surrey_<domain>_full_matrix.yaml` stay structurally
identical — they just `!include` the updated dataset YAMLs. `runner.py`'s
existing load/score/unload-per-scorer loop is already correct: each of the
~25 scorers is loaded once, scored against the full fused dataset, then
unloaded before the next one. The output is a single
`$SCRATCH/outputs/surrey_<domain>_full_matrix/` directory per domain.

### 3. `mt-metrix tabulate` subcommand

**CLI:**

```
mt-metrix tabulate \
    --runs-glob '$SCRATCH/outputs/surrey_*_full_matrix' \
    --out results/ \
    [--metrics r,rho]             # default
    [--bold best|off]              # default: best (bold best ρ per column within each domain)
    [--families auto|<list>]       # default: auto (COMET-QE,COMET-ref,Tower-DA,Tower-MQM,sacrebleu)
    [--langs auto|<list>]          # default: auto (union of observed lang_pairs)
    [--domain-order legal,general,healthcare,tourism]
```

Runs in the conda env; no GPU needed; seconds on full data.

**Input discovery.** For each run directory, read:

- `summary.json` — domain (via `dataset.domain`), scorer list (family, name),
  skipped_metrics.
- `segments.tsv` — per-segment data with columns
  `segment_id, lang_pair, domain, source, target, reference, gold, <scorer_1>, <scorer_2>, …`.

**Aggregation.** Group by `(domain, scorer_name, lang_pair)`; for each group
compute Pearson's r and Spearman's ρ against the `gold` column (drop NaN
rows per-pair). Store counts `n` alongside. Row-wise Avg is the mean of
non-NA correlations for that (domain, scorer) row.

**Scorer family / group inference.**

| family     | group         | rule                                             |
|------------|---------------|--------------------------------------------------|
| `comet`    | `COMET-QE`    | `needs_reference == false`                       |
| `comet`    | `COMET-ref`   | `needs_reference == true`                        |
| `tower`    | `Tower-MQM`   | scorer `name` ends in `-mqm`                     |
| `tower`    | `Tower-DA`    | otherwise                                        |
| `sacrebleu`| `sacrebleu`   | always                                           |

`needs_reference` is already on the scorer instance; `name` is in the
summary. An optional `group: <str>` field on each scorer in
`configs/models/*.yaml` overrides this inference if the user wants custom
grouping later.

**Default column order:** `En-Hi, En-Mr, En-Ta, En-Te, En-Gu, Avg` (matching
Table 8). The implementation sorts observed lang pairs in this canonical
order; unknown lang pairs are appended alphabetically before Avg.

**Default row order:** by domain in the `--domain-order` above, then by
group in the family order above, then by scorer name alphabetically.

### Output files

`results/paper_table.tex` — booktabs LaTeX:

```latex
\begin{tabular}{ll*{12}{r}}
\toprule
\textbf{Domain} & \textbf{Model}
    & \multicolumn{2}{c}{En-Hi} & \multicolumn{2}{c}{En-Mr}
    & \multicolumn{2}{c}{En-Ta} & \multicolumn{2}{c}{En-Te}
    & \multicolumn{2}{c}{En-Gu} & \multicolumn{2}{c}{Avg} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}
    & & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ \\
\midrule
\multirow{12}{*}{\rotatebox{90}{General}}
    & wmt22-cometkiwi-da   & 0.xxx & 0.xxx & …  & … & \textbf{0.xxx} & … \\
    & wmt23-cometkiwi-xl   & …                                                         \\
    \cmidrule(l){2-14}
    & tower-plus-2b        & …                                                         \\
\midrule
\multirow{…}{*}{\rotatebox{90}{Healthcare}}
    & …                                                                                \\
\bottomrule
\end{tabular}
```

`results/paper_table.md` — GitHub-flavoured Markdown with two header rows
(GFM supports only flat headers, so lang and metric collapse into
`En-Hi(r) | En-Hi(ρ) | …` column names); model-group separator rows are
inserted as `**COMET-QE**`, `**Tower-DA**`, etc. in the Model column to
signal grouping visually.

`results/results.csv` — long form, one row per
`(domain, scorer, lang_pair)` combination:

```
domain,scorer,family,group,lang_pair,n,pearson,spearman,kendall
legal,wmt22-cometkiwi-da,comet,COMET-QE,en-gu,270,0.643,0.442,0.316
legal,wmt22-cometkiwi-da,comet,COMET-QE,en-ta,260,…
…
```

Long-form CSV is the source of truth — the LaTeX and Markdown tables are
reshaped from this. Downstream analysis (per-lang ablation, bootstrap,
plotting) reads the CSV.

### Missing-data handling

- **Missing lang pair in the dataset** (e.g., Tourism has no en-gu):
  no rows with `lang_pair == "en-gu"` appear in `segments.tsv` for that
  domain → aggregator emits `NA` for that cell in the LaTeX/MD table and
  omits the row from the CSV.
- **Scorer failed during run** (OOM, API error): scorer appears in
  `summary.json::skipped_metrics`, not in `segments.tsv` columns →
  aggregator emits `NA` across that scorer's entire row for that domain.
- **Zero-variance in gold or prediction for a subset**: correlation is
  undefined → `NA`, and a warning is logged.

### Best-ρ bolding

Within each domain block (not across domains), for each lang-pair column,
find the scorer with the highest ρ (ignoring NAs), bold it in the LaTeX
table, and mark with `**…**` in Markdown. The CSV is untouched — bolding
is a presentation concern.

## Testing

- **Unit:**
  - `test_datasets_multi_subset` — loader fuses two fake HF subsets,
    verifies `lang_pair` column distinguishes them, checks
    `limit` applies post-concat.
  - `test_tabulate_aggregation` — synthetic segments.tsv + summary.json
    → assert per-group correlations match a hand-computed baseline.
  - `test_tabulate_missing_data` — one domain×lang pair absent, one
    scorer skipped; assert NA cells appear in the LaTeX/MD output and
    the scorer is excluded from the CSV rows for that combination.
  - `test_tabulate_family_inference` — assert the group mapping
    table in the design section renders each scorer into the correct
    family bucket.
- **Integration:**
  - New fixture-based e2e test that runs the existing mini-dataset
    fixture through two synthetic "scorers" across two lang pairs and
    verifies the resulting LaTeX compiles (shell out to `pdflatex` if
    available, otherwise a syntax-only parser check).
- **Smoke:** `example_quick` run already covers the single-subset form;
  add a `tests/fixtures/runs/` snapshot with two lang pairs to exercise
  the aggregator end-to-end without a GPU.

## Backwards compatibility

- Single-subset `config: <str>` form still works (loader checks `configs`
  first, falls back to `config`).
- Existing matrix configs don't change; they benefit automatically once
  the dataset YAMLs are updated to `configs: [...]`.
- `mt-metrix correlate` (existing per-run correlation subcommand) is
  untouched; `tabulate` is additive.

## Open questions

1. **Table 8 uses En-Hi / En-Mr / En-Ta / En-Te / En-Gu order.** Our
   existing code normalises to ISO form `en-gu`, `en-hi`, `en-mr`,
   `en-ta`, `en-te`. The tabulator will display as Title-Case `En-Gu`
   etc. to match the paper; the column-order override in the CLI lets
   you swap for different sort preferences.
2. **Avg across NAs.** The mean is taken over non-NA lang pairs in the
   row. The paper appears to do the same (Table 8 Legal rows show Avg
   over 3 available pairs). Worth confirming on a paper side-by-side.
3. **Kendall's τ.** Computed and written to CSV; not rendered in the
   LaTeX/MD tables by default. Easy to turn on with `--metrics r,rho,tau`
   later.

## Files touched

- `src/mt_metrix/io/datasets.py` — multi-subset loader.
- `src/mt_metrix/cli.py` — new `tabulate` subcommand.
- `src/mt_metrix/reports/tabulate.py` — new module (aggregation + writers).
- `configs/datasets/surrey_legal.yaml` — switch to `configs: [...]`.
- `configs/datasets/surrey_general.yaml` — add `configs: [...]`.
- `configs/datasets/surrey_health.yaml` — add `configs: [...]`.
- `configs/datasets/surrey_tourism.yaml` — add `configs: [...]`.
- `tests/test_datasets.py` — new multi-subset tests.
- `tests/test_tabulate.py` — new aggregator tests.
- `tests/fixtures/runs/mini_matrix/` — fixture run data.
- `examples/05_paper_matrix.md` — end-to-end walkthrough of the matrix
  → tabulate loop.

No new top-level dependencies; `scipy.stats.pearsonr` / `spearmanr` and
`kendalltau` are already transitive deps via pandas / sacrebleu.
