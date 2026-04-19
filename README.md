# mt-metrix

A comprehensive MT evaluation suite. Single config → one command → per-segment
and corpus-level scores on a dataset using any combination of:

- **COMET** family (Unbabel): `wmt22-comet-da`, `wmt22-cometkiwi-da`,
  `wmt23-cometkiwi-da-xl/xxl`, `eamt22-cometinho-da`, `XCOMET-XL/XXL`, plus the
  historical WMT20 / WMT21 checkpoints.
- **Tower** family via LLM-prompted scoring (GEMBA-DA, GEMBA-MQM,
  Tower-native): `TowerBase-7B/13B`, `TowerInstruct-7B/13B` + variants,
  `Tower-Plus-2B/9B/72B`.
- **Reference-based lexical metrics** via sacrebleu: BLEU, chrF, chrF++, TER
  (auto-skipped when the dataset has no reference column).

Library-first, batch-first, with a thin CLI and an AISURREY SLURM submission
path built in.

## Quick start

```bash
# install (core + COMET + Tower optional extras)
pip install -e ".[comet,tower]"

# smoke test: cometinho + CometKiwi on 20 rows of Surrey Legal-QE
mt-metrix score --config configs/runs/example_quick.yaml

# a single CometKiwi run on the full Legal-QE split
mt-metrix score --config configs/runs/surrey_legal_cometkiwi.yaml

# submit one SLURM job per scorer to AISURREY
mt-metrix submit --config configs/runs/surrey_legal_full_matrix.yaml
```

Outputs always land under `outputs/<run_id>/`:

```
outputs/<run_id>/
  config.yaml        # fully-resolved snapshot (no !include, no refs)
  segments.tsv       # flat, one row per segment, one column per metric
  segments.jsonl     # rich, includes per-metric `extra` payload
  summary.json       # metric aggregates + correlations vs gold + skipped list
  run.log
```

## Why mt-metrix

Running `wmt22-cometkiwi-da` once is fine. Running 12 COMET variants × 16 Tower
variants × 4 domains with consistent outputs and correlations is the actual
job. mt-metrix:

1. **Plugin scorers** behind a single `Scorer` protocol. Adding a new metric
   family is a new module, not a fork of the runner.
2. **Config-first.** Every run is reproducible from `outputs/<run_id>/config.yaml`
   — model IDs, parameters, dataset path, everything.
3. **Auto-detect references.** If the dataset has no `reference` column, ref-
   based metrics are skipped with a visible `skipped_metrics` entry in
   `summary.json`.
4. **Uniform outputs.** Downstream analysis never branches on "which model was
   this" — segment.tsv / segments.jsonl / summary.json have the same shape
   regardless of scorer.
5. **Cluster + local on the same code path.** `score` runs locally; `submit`
   writes an sbatch script that runs the same `score` command on AISURREY.

## Repository layout

```
mt-metrix/
├── src/mt_metrix/       # library code
│   ├── scorers/         # plugin scorers (comet.py, tower.py, sacrebleu_scorer.py)
│   ├── io/              # dataset loaders, segment schema, output writers
│   ├── prompts/         # GEMBA-DA / GEMBA-MQM / Tower-native prompts
│   ├── submit/          # SLURM rendering + submission
│   ├── config.py        # YAML loader (!include, ref: lookups)
│   ├── runner.py        # one run end-to-end
│   └── cli.py
├── configs/
│   ├── models/          # catalogues per family (comet.yaml, tower.yaml, ...)
│   ├── datasets/        # dataset configs
│   └── runs/            # run configs (recipe = dataset + scorers)
├── scripts/slurm_templates/   # standalone sbatch templates by model size
├── docs/                # DESIGN.md, AISURREY.md, MODELS.md, PARAMETERS.md
├── examples/            # walkthroughs
└── tests/               # unit + integration (slow tests gated on env var)
```

## CLI

```bash
mt-metrix score          --config <run.yaml> [--override key=value ...]
mt-metrix submit         --config <run.yaml> [--partition a100 --gpus 1 ...]
mt-metrix list-models    [--family comet|tower|sacrebleu]
mt-metrix list-datasets
mt-metrix correlate      --run <outputs/run_id>
mt-metrix download       --family comet --to <scratch-dir>
```

## Writing a run config

A run is a dataset + an ordered list of scorers. Anything in the catalogues
is reachable by `ref:`; inline config also works.

```yaml
run:
  id: my_run

dataset: !include ../datasets/surrey_legal.yaml

scorers:
  - ref: comet/wmt22-cometkiwi-da
  - ref: comet/wmt23-cometkiwi-da-xl
    overrides:
      batch_size: 16
  - ref: tower/towerinstruct-7b-v0.2
  - ref: sacrebleu/chrf++            # auto-skipped if no reference column

output:
  root: outputs
  formats: [tsv, jsonl, summary]
```

See `configs/runs/` for ready-made recipes and `examples/` for walkthroughs.

## Tests

```bash
pytest                     # fast tests only (~seconds, no GPU, no downloads)
MT_METRIX_RUN_SLOW=1 pytest  # also run COMET/Tower integration tests
```

## Licence

MIT for mt-metrix itself. Model checkpoints carry their own licences —
the Tower family is **CC-BY-NC-4.0** (non-commercial research). Several COMET
checkpoints are **gated** on HuggingFace and require licence acceptance.
See `LICENSE` and `docs/MODELS.md`.

## Citations

mt-metrix is plumbing around other people's work. If you publish results,
cite the underlying metrics and models:

- Rei et al. 2020, 2022 — COMET / COMET-22 / CometKiwi
- Rei et al. 2023 — CometKiwi-XL/XXL
- Guerreiro et al. 2023 — XCOMET
- Alves et al. 2024 — Tower / TowerInstruct
- Pombal et al. 2025 — Tower-Plus
- Kocmi & Federmann 2023 — GEMBA-DA, GEMBA-MQM
- Popović 2017 — chrF / chrF++
- Papineni et al. 2002 — BLEU
- Snover et al. 2006 — TER
- Post 2018 — sacrebleu (reproducibility signatures)

See `docs/PARAMETERS.md` for exact paper-sourced defaults.
