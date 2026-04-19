# mt-metrix — Project Instructions

## What this is
A comprehensive MT evaluation suite. Single config → one command → scores on a
dataset using any combination of COMET variants, Tower (LLM-prompted scoring via
GEMBA-DA / GEMBA-MQM), and reference-based metrics (BLEU, chrF++, TER).
Library-first with a thin CLI; batch is the first-class path, training-loop hooks
and SLURM submission are built in.

## Architecture reference
Read `docs/DESIGN.md` for full rationale, `docs/MODELS.md` for the supported-model
catalogue, `docs/PARAMETERS.md` for paper-sourced defaults, and `docs/AISURREY.md`
for cluster-specific ops before making changes.

## Code conventions
- British English in documentation and user-facing text (log messages, CLI help)
- Type hints on every public function
- Scorers are plugins implementing a single `Scorer` protocol in
  `src/mt_metrix/scorers/base.py` — no cross-family coupling
- Configs are YAML files under `configs/`; never hardcode model IDs,
  hyperparameters, or dataset paths in Python
- No API keys, tokens, or HF credentials in source files — use `HF_TOKEN`
  environment variable (loaded from `.env` locally, from `~/.hf_token` on cluster)
- Outputs always land under `outputs/<run_id>/` and include `config.yaml`,
  `segments.tsv`, `segments.jsonl`, `summary.json`, `run.log`

## Testing
```
pytest tests/ -v
```
Tests use tiny fixture datasets (10 rows) under `tests/fixtures/`. COMET and
Tower tests are marked `slow` and skipped unless `MT_METRIX_RUN_SLOW=1`.

## Running a scoring job
```
mt-metrix score --config configs/runs/surrey_legal_cometkiwi.yaml
```

Default output directory: `outputs/<dataset>_<model>_<YYYYMMDD-HHMMSS>/`.
Override with `--output-root <path>`.

## Cluster deployment (AISURREY)

Don't improvise cluster ops — read these first:
- `~/Documents/Claude/agent-context/aisurrey-cluster.md` — filesystem topology
  (/vol/research login-only, /mnt/fast/nobackup scratch compute-visible), torch
  2.4.1+cu121 pin, HF cache redirect, a100 partition, aisurrey26 is flaky (use
  `--exclude=aisurrey26`).
- `docs/AISURREY.md` in this repo — mt-metrix-specific conventions (scratch
  paths, model cache layout, SLURM template variants by model size).

Scratch layout on AISURREY:
```
/mnt/fast/nobackup/scratch4weeks/dk0023/mt-metrix/
  models/     # HF cache for model weights (COMET + Tower)
  hf-cache/   # HF hub cache (HF_HOME)
  outputs/    # run outputs
```

Gated-model gotcha: COMET models (`wmt22-cometkiwi-da`, `XCOMET-*`,
`cometkiwi-da-xl/xxl`) are gated on HuggingFace. Accept the licence on the web UI
for each one, put your token in `~/.hf_token`, and ensure SLURM jobs export
`HF_TOKEN=$(cat ~/.hf_token)`.

## Key design rules
1. **Plugin scorers** — every metric family is a plugin behind `Scorer`. No
   special cases in the runner for "COMET vs Tower".
2. **Config is the source of truth** — you can always reconstruct a run from its
   `config.yaml`. Random seeds, model revisions, parameter values all snapshot.
3. **Auto-detect references** — if the dataset has a `reference` column,
   reference-based metrics run; otherwise they skip with a visible warning.
4. **Output is the same shape regardless of model** — every run produces
   `segments.tsv`, `segments.jsonl`, `summary.json`. Downstream analysis never
   branches on "which model was this".
5. **Don't wrap what's maintained upstream** — use `unbabel-comet` for COMET,
   `sacrebleu` for BLEU/chrF++, `vllm` for Tower. Plugin authors glue APIs, they
   don't reimplement.
