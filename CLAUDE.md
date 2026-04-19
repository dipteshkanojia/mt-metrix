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

Don't improvise cluster ops — read these first, in order:
- `docs/SESSION_HANDOFF.md` — fast onboarding for a fresh session.
- `docs/AISURREY.md` — mt-metrix-specific runbook.
- `~/Documents/Claude/agent-context/aisurrey-cluster.md` — cluster
  topology, partition specs, reliability history.
- `~/Documents/Claude/agent-context/aisurrey-deploy.md` — the five
  pre-flight checks every submission runs through.

**Canonical submit path: `scripts/submit.sh` — no exceptions.** It runs
five pre-flight checks (partition sanity, conda env present, no
duplicate job, `sbatch --test-only` validation) and adds
`--exclude=aisurrey26` automatically. Direct `sbatch` invocation is
deprecated.

```
scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
```

One-time cluster setup is `scripts/setup_cluster.sh` — idempotent,
clones the repo, creates the env, installs torch 2.4.1+cu121, runs the
smoke tests.

Scratch layout on AISURREY:
```
/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/
  repo/       # this repo (cloned by setup_cluster.sh)
  models/     # pre-downloaded HF weights (optional)
  hf_cache/   # HF_HOME (datasets + on-the-fly downloads)
  outputs/    # run outputs (--output-root)
```

Gotchas (do not improvise around these):
- **No `gpu` partition on AISURREY.** Default is `a100`; cheaper
  alternatives are `rtx_a6000_risk` / `l40s_risk` / `rtx8000` (48 GB),
  `3090` / `3090_risk` (24 GB), `2080ti` (11 GB), `debug` (4h cap).
- **`aisurrey26` is flaky** (silent `1:0` exits, 2026-04); the wrapper
  always excludes it.
- **Conda env is a prefix path on scratch** (`$SCRATCH/conda_env`), not a named env. Activate with `conda activate /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/conda_env`. The user volume can't hold torch + vllm + comet deps.
- **torch pin: `torch==2.4.1+cu121`.** Newer/older builds cause silent
  NCCL mismatches.
- **HF cache** must be redirected to scratch via `HF_HOME`,
  `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` in the sbatch header (not
  only in `.bashrc`); `scripts/run_mt_metrix.slurm` does this.
- **Gated models** (`wmt22-cometkiwi-da`, `XCOMET-*`, all Tower) need
  licence acceptance on the HF web UI for the account whose token sits
  in `~/.hf_token`.

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
