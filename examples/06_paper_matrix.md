# 06 — Paper matrix: all domains × all language pairs × all scorers

This walkthrough reproduces the ALOPE-style Table 8 — hierarchical rows
(Domain → scorer group → model), columns per language pair with an `Avg`
column, bold best-in-column-per-domain, NA cells for missing pairs. Four
domains (Legal, General, Health, Tourism) × five Indic language pairs
(En-Gu, En-Hi, En-Mr, En-Ta, En-Te), run every QE-capable metric.

The workflow is two steps: **submit four matrix jobs** (one per domain,
each loads every scorer serially and concatenates all language subsets),
then **tabulate** the outputs into CSV + Markdown + LaTeX.

## 0. On AISURREY — pull and activate

```bash
cd $HOME/workspace/mt-metrix            # or wherever you cloned on the cluster
git pull
conda activate /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/conda_env
```

## 1. Submit one matrix per domain

Each `surrey_<domain>_full_matrix.yaml` already pins the right dataset
(all language subsets concatenated) and the full scorer list. Default
plan is `a100 / 1 GPU / 24h` — plenty for any single COMET run; override
for Tower-13B+.

```bash
# Legal — 3 langs (en-gu, en-ta, en-te)
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml

# General — 5 langs (en-gu, en-hi, en-mr, en-ta, en-te)
scripts/submit.sh configs/runs/surrey_general_full_matrix.yaml

# Health — 4 langs (en-gu, en-hi, en-mr, en-ta)
scripts/submit.sh configs/runs/surrey_health_full_matrix.yaml

# Tourism — 3 langs (en-hi, en-mr, en-te)
scripts/submit.sh configs/runs/surrey_tourism_full_matrix.yaml
```

Or, one-shot all four:

```bash
scripts/submit_aisurrey.sh
```

The runner loads scorer 1, scores every segment across every language
pair, unloads, then loads scorer 2, and so on. One sbatch job per domain
— no model-reload overhead between language pairs.

### Right-size GPU per domain

CometKiwi-DA fits on 24 GB; XL/XXL need 48 GB+; Tower-13B wants 2× A100.
Split any stressful entries into their own run configs rather than
bloating one job's resource envelope:

```bash
scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml \
    -p 3090_risk --gres=gpu:1                  # 24 GB, fast queue

scripts/submit.sh configs/runs/surrey_legal_all_tower.yaml \
    --gres=gpu:2 --mem=128G --time=08:00:00
```

## 2. Wait for all four to finish

```bash
squeue -u $USER
# When queue clears:
ls $SCRATCH/outputs/
# surrey_legal_full_matrix/   surrey_general_full_matrix/
# surrey_health_full_matrix/  surrey_tourism_full_matrix/
```

Each output dir has `summary.json` (aggregate correlations + skip reasons),
`segments.tsv` (per-segment scores with the `lang_pair` column for pivoting),
`segments.jsonl` (rich form with scorer extras) and `run.log`.

## 3. Tabulate

```bash
mt-metrix tabulate \
    --runs-glob "$SCRATCH/outputs/surrey_*_full_matrix" \
    --out results/ \
    --metric spearman
```

Output:
- `results/results.csv`    — long form: one row per (domain, scorer, lang_pair).
- `results/paper_table.md` — GitHub-flavoured markdown, flat header, NA as `—`.
- `results/paper_table.tex`— booktabs LaTeX, hierarchical rows via `\multirow`,
  bold best-per-column-per-domain.

The Markdown table is fine to paste into a draft; the LaTeX table drops
straight into a paper — it expects `\usepackage{booktabs,multirow}` in
the preamble.

### Pick a different correlation

```bash
# Pearson r instead of Spearman rho
mt-metrix tabulate --runs-glob "..." --out results/ --metric pearson

# Kendall tau
mt-metrix tabulate --runs-glob "..." --out results/ --metric kendall

# Soft Pairwise Accuracy
mt-metrix tabulate --runs-glob "..." --out results/ --metric spa
```

The long-form CSV always contains all four metrics — `--metric` only
picks which one to surface in the rendered tables.

## 4. Reading the table

- Row groups are ordered: **COMET-QE → COMET-ref → Tower-DA → Tower-MQM
  → sacrebleu**. `\midrule` between groups, between domains.
- `COMET-ref` + `sacrebleu` rows will be NA for the Surrey QE datasets
  (no reference column), but they stay in the table so the reviewer sees
  what was considered, not just what was selected.
- Bold cells: best value in that `(domain, lang_pair)` cell among all
  scorers. Ties are all bolded.
- NA cells (`—` in MD, `--` in LaTeX) mean either:
  (a) the domain doesn't ship that language pair (e.g. Legal has no en-hi), or
  (b) the scorer landed in `skipped_metrics` (gated weights not accepted,
      OOM, dataset missing the column the scorer needs).

Check `$SCRATCH/outputs/<run>/summary.json::skipped_metrics` to see why a
row came out empty.

## 5. Running on a single language pair

The shared `configs/datasets/*.yaml` now default to `configs: [all pairs]`.
To scope a run to one pair, copy the YAML and swap `configs:` for
`config:` (singular):

```yaml
# configs/datasets/surrey_legal_gu_only.yaml
kind: huggingface
repo: surrey-nlp/Legal-QE
config: en-gujarati     # was `configs: [en-gujarati, en-tamil, en-telugu]`
split: test
domain: legal
columns:
  source: source_text
  target: target_text
  gold: z_mean
  lang_pair: "@from:language_pair"
  domain: "@constant:legal"
```

Then point a run config at the scoped dataset instead of the canonical one.
