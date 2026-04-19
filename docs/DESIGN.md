# mt-metrix — Design Document

**Status:** v0.1 — first implementation.
**Author:** Diptesh Kanojia (with Claude).
**Date:** 2026-04-18.

## 1. Purpose

A single tool to evaluate machine-translation quality using any combination of:

- **COMET family** (reference-based and reference-free Quality Estimation)
- **Tower family** (LLM-based scoring via GEMBA-style prompts)
- **Reference-based metrics** (BLEU, chrF++, TER via `sacrebleu`)

on any dataset (local TSV/CSV/JSONL, HuggingFace hub, or the unified gyroQE
catalogue), with one config file, one CLI command, and an output format that is
consistent regardless of which metrics ran.

This is the MT-evaluation counterpart to gyroQE — gyroQE *trains* new QE
metrics, mt-metrix *runs* them (and their competitors, and reference metrics)
as an evaluation suite.

## 2. Scope

### In scope (v0.1)

- Plugin registry for metric families.
- Scorers for: COMET family (all HF Unbabel variants), Tower family
  (TowerBase, TowerInstruct, Tower-Plus, TowerInstruct-Mistral) via GEMBA-DA
  and GEMBA-MQM prompts, sacrebleu (BLEU, chrF++, TER).
- Dataset loaders: local file (TSV/CSV/JSONL/parquet), HuggingFace hub, gyroQE
  catalogue adapter.
- Output writers: TSV, JSONL, summary JSON.
- Correlation evaluation against gold scores (Pearson, Spearman, Kendall, SPA).
- YAML config system with model- / dataset- / run-level reuse.
- CLI: `score`, `list-models`, `list-datasets`, `submit`, `download`.
- SLURM submission for AISURREY (a100 partition, scratch-aware paths, HF cache
  redirect, `--exclude=aisurrey26`, torch-2.4.0+cu121 pin).
- Configs for all 4 Surrey NLP domain-specific QE datasets (Legal, General,
  Tourism, Health), all COMET variants, all Tower variants.

### Out of scope (v0.1)

- Training of new metrics — use gyroQE.
- Reranking / N-best tasks — this scores given translations, no re-decoding.
- BLEURT, MetricX (Google), YiSi-2 — architecture supports them (add a plugin)
  but no scorers shipped in v0.1. Tracked in `docs/MODELS.md` as "planned".
- Web UI or API service — library + CLI only.
- `push_to_hub` publishing of scored datasets — separate concern.

## 3. High-level architecture

```
                       ┌──────────────────────┐
                       │      CLI (cli.py)    │
                       │  score / submit / …  │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   Runner (runner.py) │
                       │  orchestrates a run  │
                       └──────────┬───────────┘
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │ Dataset loader │      │   Scorers      │      │   Writers      │
 │  (io/datasets) │      │ (plugins via   │      │  (io/writers)  │
 │ HF / local /   │      │  registry)     │      │  TSV / JSONL / │
 │ gyroQE catalogue│     │                │      │  summary.json  │
 └────────────────┘      └───┬────────┬──┘       └────────────────┘
                              │        │
                    ┌─────────┘        └──────────┐
                    ▼                             ▼
          ┌──────────────────┐          ┌──────────────────┐
          │  COMETScorer     │          │  TowerScorer     │
          │  unbabel-comet   │          │  vllm + prompts  │
          └──────────────────┘          └──────────────────┘

                    ┌──────────────────┐
                    │ SacreBLEUScorer  │
                    │  sacrebleu       │
                    └──────────────────┘
```

Every scorer implements the same `Scorer` protocol. Adding a new family
(MetricX, BLEURT, custom) is a new file under `src/mt_metrix/scorers/` and
registry registration — no core changes.

## 4. Data model

### Input schema (unified)

All datasets are normalised into this structure:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `source` | str | yes | Source sentence |
| `target` | str | yes | MT output (the thing being scored) |
| `reference` | str \| None | no | Gold translation, when available |
| `gold` | float \| None | no | Human score (z_mean, DA, MQM, …) for correlation |
| `lang_pair` | str | yes | ISO pair, e.g. `en-de`, `en-gu` |
| `domain` | str | no | `general`, `legal`, `tourism`, `health`, … |
| `segment_id` | str | no | Stable identifier; auto-generated if absent |
| `meta` | dict | no | Passthrough: annotator scores, year, split, … |

Loaders produce `list[Segment]` (typed dataclass). The runner passes segments
to each scorer; scorers return `list[SegmentScore]`.

### Output schema

Per run, `outputs/<run_id>/` contains:

- **`config.yaml`** — exact config used (resolved, with defaults expanded).
- **`segments.tsv`** — one row per segment, columns: `segment_id`, `lang_pair`,
  `domain`, `source`, `target`, `reference`, `gold`, plus one column per
  active metric.
- **`segments.jsonl`** — same data with rich fields: XCOMET error spans, Tower
  raw LLM output, prompt parse flag, etc.
- **`summary.json`** — per-metric: `mean`, `std`, and if `gold` is present,
  Pearson / Spearman / Kendall / SPA vs gold. Also: `run_metadata`
  (git SHA, pip freeze, torch/cuda, wall time), `skipped_metrics` with reasons.
- **`run.log`** — stdout + warnings (skipped metrics, parse failures, OOM).

`<run_id>` format: `{dataset_short}_{models_short}_{YYYYMMDD-HHMMSS}`, e.g.
`surrey_legal_cometkiwi-da_20260418-192301`.

## 5. Scorer interface

```python
# src/mt_metrix/scorers/base.py
class Scorer(Protocol):
    name: str              # "cometkiwi-da", "gemba-da:TowerInstruct-7B-v0.2", …
    family: str            # "comet", "tower", "sacrebleu"
    needs_reference: bool  # False for QE metrics, True for BLEU/chrF++/comet-da

    def load(self) -> None: ...
    def score(self, segments: list[Segment]) -> list[SegmentScore]: ...
    def unload(self) -> None: ...  # release GPU memory
```

`SegmentScore` has `score: float` plus an open `extra: dict[str, Any]` for
model-specific payload (XCOMET spans, Tower rationale, etc.).

Runners call `load → score → unload` serially per scorer so GPU memory is
released between metrics. This matters on the cluster when we run COMET-XXL
→ Tower-13B back-to-back.

## 6. Metric family details

### 6.1 COMET

- Backend: `unbabel-comet` (pinned ≥ 2.2.2).
- Variants supported (enumerated in `configs/models/comet.yaml`):
  - `wmt22-comet-da` — reference-based, XLM-R-large
  - `wmt22-cometkiwi-da` — QE, InfoXLM-large (gated)
  - `wmt23-cometkiwi-da-xl` — QE, XLM-R-XL 3.5B (gated)
  - `wmt23-cometkiwi-da-xxl` — QE, XLM-R-XXL 10.7B (gated)
  - `XCOMET-XL` — reference-based + QE-mode, span-level errors (gated)
  - `XCOMET-XXL` — same, larger (gated)
  - `eamt22-cometinho-da` — lightweight reference-based
  - `wmt20-comet-da`, `wmt20-comet-qe-da`, `wmt21-comet-da`,
    `wmt21-comet-qe-da`, `wmt21-comet-qe-mqm` — historical checkpoints
- Default inference params (from COMET paper, Rei et al. 2020/2022/2023):
  - `batch_size`: 16 (XL/XXL), 64 (base/large), 128 (small/cometinho)
  - `gpus`: 1 (single-GPU default; multi-GPU via `DEVICES`)
  - `num_workers`: 2
  - `progress_bar`: True
  - XCOMET: `output_seg_err_spans`: True (populates `extra.error_spans`)
- Reference requirement: **`wmt*-comet-da`, `XCOMET-*`** can use reference;
  **`wmt*-cometkiwi-da`** are QE (no reference needed). The runner consults
  `needs_reference` per scorer and skips if the dataset lacks refs (warning
  logged).

### 6.2 Tower (GEMBA-DA / GEMBA-MQM prompting)

- Backend: `vllm` (preferred) or `transformers` (fallback for small models,
  local-dev smoke tests).
- Variants supported (`configs/models/tower.yaml`):
  - `Unbabel/TowerBase-7B-v0.1`
  - `Unbabel/TowerBase-13B-v0.1`
  - `Unbabel/TowerInstruct-7B-v0.1`
  - `Unbabel/TowerInstruct-7B-v0.2`
  - `Unbabel/TowerInstruct-13B-v0.1`
  - `Unbabel/TowerInstruct-Mistral-7B-v0.2`
  - `Unbabel/TowerInstruct-WMT24-Chat-7B`
  - `Unbabel/Tower-Plus-2B` (Gemma2)
  - `Unbabel/Tower-Plus-9B` (Gemma2)
  - `Unbabel/Tower-Plus-72B` (Qwen2)
- Prompting modes:
  - **GEMBA-DA** (Kocmi & Federmann 2023): "Score this translation from 0 to
    100…" — single float per segment. `temperature=0.0`, `max_tokens=50`,
    `top_p=1.0`. Parsed via regex `\b(\d{1,3})\b` with range check [0, 100].
  - **GEMBA-MQM**: ask for MQM error categories + severities, parse into a
    numeric score (deductive: `-25 * critical − 5 * major − 1 * minor`, clipped
    to [0, 100]). `temperature=0.0`, `max_tokens=256`. Rich `extra` with raw
    response and parsed errors.
  - **Tower-native**: the Tower paper (Alves et al. 2024) shows a specific
    chat template for MT-eval tasks. Implemented in `prompts/tower_native.py`
    (opt-in via config).
- vLLM config: `tensor_parallel_size` auto-set from visible GPUs; defaults
  optimised for a100-40GB: 7B → 1 GPU, 13B → 2 GPUs, 72B → 4 GPUs (tensor
  parallel) or offload. Settable via config.
- Parse failures logged in `extra.parse_ok=False` with raw response retained.

### 6.3 Reference metrics (sacrebleu)

- `sacrebleu ≥ 2.4`.
- Scorers: `bleu`, `chrf` (chrF++), `ter`.
- Signatures stored in `summary.json` for reproducibility.
- Per-segment scores computed via sentence-level `sacrebleu.sentence_bleu`,
  `sentence_chrf`, `sentence_ter`.
- Corpus-level scores also computed and included in `summary.json`.
- chrF++ default: `chrf_order=6`, `chrf_word_order=2`, `use_effective_order=True`,
  `lowercase=False`. Matches WMT defaults.

### 6.4 Adding a new family

Any plugin implementing `Scorer` and registering itself with
`scorers.registry` is automatically callable via config. See
`docs/ADDING_METRICS.md` for a walk-through.

## 7. Dataset loaders

### 7.1 Local file loader

TSV / CSV / JSONL / parquet. Column mapping in config:

```yaml
dataset:
  kind: local
  path: path/to/file.tsv
  columns:
    source: original
    target: translation
    reference: post_edit       # optional
    gold: z_mean                # optional
    lang_pair: "@constant:en-de"   # literal, not a column
```

### 7.2 HuggingFace loader

```yaml
dataset:
  kind: huggingface
  repo: surrey-nlp/Legal-QE
  config: en-gu            # optional, HF subset config
  split: test              # default: test
  columns:
    source: source_text
    target: target_text
    gold: z_mean
    domain: "@constant:legal"
    lang_pair: "@from:language_pair"
```

`@constant:X` = literal value for every row. `@from:col` = copy column.
Missing optional columns → field is `None`.

### 7.3 gyroQE catalogue loader

Reads gyroQE's unified TSV output (produced by
`gyroqe/data/loaders.py`). Shortcut:

```yaml
dataset:
  kind: gyroqe
  path: /path/to/gyroQE
  name: mlqe-pe             # selects one slice of the catalogue
  lang_pair: en-de
  split: test
```

## 8. Config system

Three layers, composable via `!include`-style references:

```
configs/
├── models/
│   ├── comet.yaml        # catalogue of all COMET variants with defaults
│   └── tower.yaml        # catalogue of all Tower variants with defaults
├── datasets/
│   ├── surrey_legal.yaml # loader config, paths, schema mapping
│   └── …
└── runs/
    └── surrey_legal_cometkiwi.yaml
       # picks: dataset + list of metrics + overrides + output path
```

A run config looks like:

```yaml
run:
  id: surrey_legal_cometkiwi_da

dataset: !include ../datasets/surrey_legal.yaml

# Either a named ref from configs/models/*.yaml, or an inline definition.
scorers:
  - ref: comet/wmt22-cometkiwi-da
    overrides:
      batch_size: 32

output:
  root: outputs
  formats: [tsv, jsonl, summary]
```

The runner resolves `ref:` against the model catalogues, applies overrides,
snapshots the fully-resolved config to `outputs/<run_id>/config.yaml`.

## 9. CLI surface

```
mt-metrix score --config <path>
                [--output-root <dir>]
                [--override key=value ...]

mt-metrix list-models [--family comet|tower|sacrebleu]
mt-metrix list-datasets
mt-metrix download --family comet --to /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/models
mt-metrix submit --config <path>
                 [--partition a100] [--gpus 1] [--time 04:00:00]
                 [--dry-run]
mt-metrix correlate --run <run_id>     # re-compute correlations from segments.tsv
```

`mt-metrix submit` is a thin Python wrapper around `scripts/submit.sh`,
which is the canonical path on AISURREY. For interactive cluster use,
prefer the shell script directly.

## 10. SLURM submission (AISURREY)

**The only supported submit path is `scripts/submit.sh`.** It runs five
pre-flight checks (partition exists and is not the nonexistent `gpu`;
conda env `mt-metrix` present; no duplicate in queue; `sbatch --test-only`
accepts the plan) and submits with `--exclude=aisurrey26`. Direct
`sbatch` invocation is deprecated.

The parameterised sbatch template lives at `scripts/run_mt_metrix.slurm`:

- Default `--partition=a100 --gres=gpu:1 --cpus-per-task=8 --mem=64G
  --time=24:00:00`. Override on the CLI:
  `scripts/submit.sh <cfg> -p rtx_a6000_risk --gres=gpu:1 --time=02:00:00`.
- Exports `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` to
  `$SCRATCH/hf_cache/...`, and `COMET_CACHE` to `$SCRATCH/models/comet`.
- Sources `~/.hf_token` into `HF_TOKEN` if present (gated models).
- Activates the `mt-metrix` conda env.
- Runs `mt-metrix score --config $CONFIG --output-root $SCRATCH/outputs`.

One-time cluster setup is `scripts/setup_cluster.sh` — idempotent; clones
the repo, creates the env with `torch==2.4.0+cu121`, installs
`[comet,tower]` extras, runs the smoke tests.

**Right-sizing.** COMET-base / CometKiwi-DA / BLEU all fit on 24 GB.
CometKiwi-XL needs 40–48 GB. Tower-7B (vLLM) fits on 24 GB; Tower-13B on
2× 48 GB or 1× A100. Only Tower-72B and COMET-XXL actually need A100.
Check `outputs/<run_id>/summary.json::peak_gpu_memory_gb` after a first
run and move to the cheapest partition that fits:

| Peak memory | Partition                                            |
|-------------|------------------------------------------------------|
| ≤ 10 GB     | `2080ti`                                             |
| ≤ 22 GB     | `3090` / `3090_risk`                                 |
| ≤ 44 GB     | `rtx8000` / `rtx_a6000_risk` / `l40s_risk`           |
| ≤ 76 GB     | `a100`                                               |

## 11. Testing strategy

- **Unit**: config parsing, scorer registry, writer schemas, dataset loaders
  with fixture TSVs.
- **Integration (slow)**: actual COMET/Tower inference on a 10-row fixture.
  Skipped unless `MT_METRIX_RUN_SLOW=1`.
- **No mocks of model inference** — per gyroQE discipline, slow tests use
  real models on tiny inputs.

## 12. Error handling philosophy

- Explicit at boundaries: config validation fails loudly with the offending
  key path.
- Graceful in-flight: if a scorer OOMs on a batch, retry at `batch_size //= 2`
  down to 1; if it still fails, mark the batch as `score=NaN` with the error
  in `extra.error` and continue with other segments / scorers.
- Always write *something* to `outputs/<run_id>/` even if the run fails
  partway — users inspect `run.log` for crash context.

## 13. Reproducibility

- `config.yaml` in every output directory is the *resolved* config — all
  includes expanded, all defaults materialised. `mt-metrix score --config
  outputs/<run_id>/config.yaml` reproduces the run.
- `summary.json` includes: git SHA (if repo clean), `pip freeze` hash,
  torch+cuda version, HF model revision SHAs, random seed.
- Model caches pinned to a known scratch path — no "which node did the weights
  download to?" surprises.

## 14. Non-goals and explicit trade-offs

- **No async / concurrent scoring across models**: simpler serial runner,
  easier memory accounting. Multi-model runs submit as separate SLURM jobs.
- **No on-the-fly fine-tuning**: evaluation only. Use gyroQE / bea2026 for
  training.
- **No custom training data schema**: we map *into* our unified schema at
  load time, we don't impose a canonical raw format.
