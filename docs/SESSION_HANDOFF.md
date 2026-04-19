# mt-metrix — Session Handoff

**Written:** 2026-04-19. **For:** a fresh Claude session picking up
mt-metrix development (or a human picking it up after a break) without the
prior conversation context.

Read this first. Everything else flows from here.

## Who / what / why in two lines

mt-metrix is a comprehensive MT-evaluation suite: single config → one
command → scores on a dataset using any combination of COMET variants,
Tower-as-evaluator (GEMBA-DA / GEMBA-MQM / Tower-native), and reference
metrics (BLEU, chrF++, TER). Library-first, batch-first, with a thin CLI
and an AISURREY SLURM submission path.

## Current state (one screen)

- **Architecture:** plugin scorers behind a `Scorer` protocol; families
  are `comet`, `tower`, `sacrebleu`. Adding a metric family is a new file,
  not a runner fork.
- **Tests:** 82 passing, 1 slow-skipped (COMET download). `pytest tests/`
  runs in ~3 s.
- **Cluster deploy:** `scripts/submit.sh` is the only path. It wraps
  `sbatch` with five pre-flight checks (partition exists and is not the
  nonexistent `gpu`, conda env present, no duplicate job, `sbatch
  --test-only` accepts the plan) and adds `--exclude=aisurrey26`. Direct
  `sbatch` invocation is deprecated — every failure mode we've hit before
  is re-caught by the wrapper.
- **Surrey QE datasets:** `configs/datasets/surrey_{legal,general,tourism,health}.yaml`
  point at `surrey-nlp/<Domain>-QE` on HF; `configs/runs/surrey_<domain>_full_matrix.yaml`
  run every supported scorer against a domain.

## What to do next (three priorities, in order)

### Priority 1: First live run on AISURREY

On the submit node, after the one-time setup:

```bash
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
scripts/submit.sh configs/runs/example_quick.yaml --dry-run
scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
```

Verify `outputs/<run_id>/summary.json` looks sane. Then scale up to one
of the real configs (`surrey_legal_cometkiwi.yaml`,
`surrey_legal_recommended.yaml`).

### Priority 2: Flagship matrices on a100

```bash
scripts/submit_aisurrey.sh                 # all four domains, full matrix
# or just one:
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml
```

Each matrix is 6–12 scorers × one domain. A100 wall time is bounded by
Tower-72B if you include it; COMET-only matrices run in <1 h.

### Priority 3: Right-size subsequent runs

After the first run, check `outputs/<run_id>/summary.json::peak_gpu_memory_gb`
and drop to a cheaper partition for anything <44 GB:

- ≤ 10 GB → `2080ti`
- ≤ 22 GB → `3090` / `3090_risk`
- ≤ 44 GB → `rtx8000` / `rtx_a6000_risk` / `l40s_risk`
- ≤ 76 GB → `a100`

`scripts/submit.sh <config> -p rtx_a6000_risk --gres=gpu:1` is the pattern.
Stop fighting for `a100` on small models.

## ONE tiny first step

On the cluster, fresh session:

```bash
ssh aisurrey
bash /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo/scripts/setup_cluster.sh
```

If the repo doesn't exist yet, the setup script clones it. If it does,
the script just updates + re-runs the smoke tests. Idempotent.

## Cluster submission discipline — DO NOT SKIP

**Preferred path: `scripts/submit.sh`.** It runs five pre-flight checks,
calls `sbatch --test-only`, and adds `--exclude=aisurrey26` automatically.
Fails fast on the common mistakes:

- partition = `gpu` (doesn't exist on AISURREY)
- env casing (the env is `mt-metrix` — check `conda env list`)
- duplicate job already in queue
- bad gres / mem / time (caught by `sbatch --test-only`)

```bash
scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
```

If you must call sbatch directly, run these first in order:

1. `sinfo` — **no `gpu` partition on AISURREY.** a100 / rtx8000 /
   rtx_a6000_risk / l40s_risk / 3090 / 3090_risk / 2080ti / debug.
2. `squeue -u $USER` — don't stack duplicates.
3. `conda env list` — env is `mt-metrix` (check casing).
4. `head -20 scripts/run_mt_metrix.slurm` — confirm `#SBATCH --partition=`
   matches a real partition.
5. `sbatch --test-only ...` — dry-run to catch gres / mem / time typos.

The slurm script defaults to `--partition=a100 --gres=gpu:1`. Override on
CLI: `scripts/submit.sh <cfg> -p rtx_a6000_risk --gres=gpu:1`.

**4-GPU soft cap.** Nodes expose up to 8 GPUs, but getting all 8 on one
node is effectively impossible. `submit.sh` warns if you request >4.

Full cluster topology and all partition specs live in
`~/Documents/Claude/agent-context/aisurrey-cluster.md` (topology) and
`~/Documents/Claude/agent-context/aisurrey-deploy.md` (deploy SOP). Read
them whenever the cluster surface is in scope.

## Must-know gotchas

1. **There is no `gpu` partition on AISURREY.** `#SBATCH --partition=gpu`
   fails instantly. Default is `a100`; cheaper alternatives are listed
   above. `scripts/submit.sh` catches the typo before sbatch sees it.

2. **`aisurrey26` is flaky** (silent `1:0` exits observed 2026-04).
   `submit.sh` always adds `--exclude=aisurrey26`.

3. **Conda env is `mt-metrix`**. Match the case shown in `conda env list`
   — `MT-metrix` / `mtmetrix` fail.

4. **Filesystem topology matters.** `/vol/research/...` is login-only,
   invisible to compute nodes. Everything a job needs must live under
   `/mnt/fast/nobackup/scratch4weeks/$USER/` (code, data, HF cache,
   weights). `setup_cluster.sh` clones into
   `/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo`.

5. **torch pin: `torch==2.4.1+cu121`.** Newer / older builds cause silent
   NCCL/SM mismatches. `setup_cluster.sh` installs exactly this.

6. **HF cache redirect.** `HF_HOME`, `TRANSFORMERS_CACHE`,
   `HF_DATASETS_CACHE` are exported in the sbatch header, not only in
   `.bashrc`. `$HOME` is too small and not always compute-visible.

7. **Gated models.** CometKiwi-DA/XL/XXL, XCOMET-XL/XXL, Tower variants
   (CC-BY-NC-4.0) need licence acceptance on huggingface.co under the
   same account whose token lives in `~/.hf_token`. A 401 at download
   time means either the token is missing or a licence hasn't been
   accepted.

8. **Auto-skip reference-based metrics.** If the dataset has no
   `reference` column, BLEU/chrF/chrF++/TER are skipped with a visible
   `skipped_metrics` list in `summary.json`. Don't "fix" this — it's
   the feature.

## Key files

| Purpose | Path |
|---------|------|
| Project intro | [README.md](../README.md) |
| Architecture + rationale | [docs/DESIGN.md](DESIGN.md) |
| Supported models | [docs/MODELS.md](MODELS.md) |
| Paper-sourced defaults | [docs/PARAMETERS.md](PARAMETERS.md) |
| Cluster runbook | [docs/AISURREY.md](AISURREY.md) |
| This file | [docs/SESSION_HANDOFF.md](SESSION_HANDOFF.md) |
| Project-agent instructions | [CLAUDE.md](../CLAUDE.md) |
| Pre-flight wrapper | [scripts/submit.sh](../scripts/submit.sh) |
| Parameterised sbatch | [scripts/run_mt_metrix.slurm](../scripts/run_mt_metrix.slurm) |
| One-time cluster setup | [scripts/setup_cluster.sh](../scripts/setup_cluster.sh) |
| Multi-domain submit | [scripts/submit_aisurrey.sh](../scripts/submit_aisurrey.sh) |
| Runner entrypoint | [src/mt_metrix/runner.py](../src/mt_metrix/runner.py) |
| CLI | [src/mt_metrix/cli.py](../src/mt_metrix/cli.py) |
| Scorer plugins | [src/mt_metrix/scorers/](../src/mt_metrix/scorers/) |

## Persistent memory

Global (applies to any AISURREY project):

- `~/Documents/Claude/agent-context/aisurrey-cluster.md` — cluster-wide
  ops (filesystem, torch pin, node reliability, SLURM patterns).
- `~/Documents/Claude/agent-context/aisurrey-deploy.md` — deploy SOP
  (five pre-flight checks, submit.sh template, right-sizing).

Project-specific (read first before non-trivial changes):

- `CLAUDE.md` in the repo root — code conventions + cluster gotchas.
- `docs/DESIGN.md` — plugin scorer protocol, YAML schema, output format.

## Rules of engagement

From the user's global preferences:

- British English in all comments and writing.
- Max 3 priorities per response. Never give long lists.
- Every task needs ONE tiny first step.
- Wednesday evening is sacred rest; no work scheduled then.

From the project's `CLAUDE.md`:

- Plugin scorers — every metric family is a plugin behind the `Scorer`
  protocol; no special cases in the runner for "COMET vs Tower".
- Config is the source of truth — every run is reproducible from its
  `config.yaml` snapshot.
- Uniform outputs — every run produces
  `segments.tsv` + `segments.jsonl` + `summary.json` + `run.log`.
  Downstream never branches on scorer family.
- Don't reimplement what's maintained upstream. Use `unbabel-comet` for
  COMET, `sacrebleu` for BLEU/chrF, `vllm` for Tower.

## If something breaks

- **`sbatch: error: invalid partition specified: gpu`** → you called
  sbatch directly. Use `scripts/submit.sh`; it catches this before
  submission.
- **`EnvironmentNameNotFound: mt-metrix`** → env casing mismatch. Run
  `conda env list` and use the exact name.
- **Job exits `1:0` with no traceback** → suspect node, usually
  `aisurrey26`. `submit.sh` excludes it; if it still happens, check
  `sacct -j <id> --format=JobID,State,ExitCode,NodeList`.
- **HF `401 Unauthorized`** → missing `~/.hf_token` or unaccepted
  licence on that specific model page.
- **`torch.cuda.OutOfMemory` on CometKiwi-XL/XXL** → add
  `overrides: {batch_size: 4}` to that scorer entry in the run config.
- **`ImportError: vllm`** → installed without `[tower]` extra. Run
  `pip install -e ".[tower]"` or remove Tower scorers from the run.
- **Test failures locally** → `pytest tests/ -v`; sacrebleu tests skip
  cleanly if sacrebleu isn't installed. COMET tests need
  `MT_METRIX_RUN_SLOW=1` and download weights.

## Quick reference — minimal commands

One-time setup on AISURREY:
```bash
ssh aisurrey
# first time only — clones repo to scratch, builds env, runs tests:
bash /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo/scripts/setup_cluster.sh || {
    # if the repo isn't there yet, fetch the script alone and let it clone:
    mkdir -p "/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix"
    cd "/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix"
    git clone git@github.com:dipteshkanojia/mt-metrix.git repo
    bash repo/scripts/setup_cluster.sh
}
```

Every subsequent session:
```bash
ssh aisurrey
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
conda activate mt-metrix
scripts/submit.sh configs/runs/<some-config>.yaml
```

End of handoff.
