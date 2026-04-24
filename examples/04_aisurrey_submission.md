# 04 — Submitting to AISURREY

The canonical submit path is `scripts/submit.sh` — a bash wrapper that
runs six pre-flight checks (including `scripts/cluster_probe.py` which
queries `scontrol` for live GPU availability per partition and infers
the job's VRAM need from the config's scorers), calls `sbatch
--test-only` to validate the plan against the live cluster, and submits
with `--exclude=aisurrey26`.

Read `docs/AISURREY.md` for the full cluster runbook, `docs/SESSION_HANDOFF.md`
for the fresh-session briefing. This example is the short version.

## 0. One-time cluster setup (idempotent)

```bash
ssh aisurrey
# Clones the repo into scratch, creates conda env with torch 2.4.0+cu121,
# installs mt-metrix with [comet,tower] extras, runs the smoke tests.
mkdir -p /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
git clone https://github.com/dipteshkanojia/mt-metrix.git repo    # SSH requires key setup; HTTPS works out of the box on AISURREY
bash repo/scripts/setup_cluster.sh
```

After the first run this script just updates the repo and re-runs tests.
Re-run it whenever you pull new dependencies.

Don't forget the HF token for gated models (CometKiwi-DA/XL/XXL,
XCOMET-XL/XXL, all Tower):

```bash
echo "hf_xxx..." > ~/.hf_token && chmod 600 ~/.hf_token
# Accept licences on huggingface.co under the same account.
```

## 1. Dry-run the submission (pre-flight only)

`--dry-run` runs the six pre-flight checks (including the cluster
probe) and `sbatch --test-only`, but does NOT queue the job. Use this
whenever you're unsure about a config change or a partition override.

```bash
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
```

The wrapper prints its checks as it goes:
```
[1/6] config file check...     ok: configs/runs/surrey_legal_full_matrix.yaml
[2/6] partition sanity check...ok: partition 'nice-project' exists
[3/6] conda env check...       ok: conda env present at .../conda_env
[4/6] duplicate job check...   ok: no duplicates of 'surrey_legal_full_matrix' in queue
[5/6] cluster probe (live capacity + VRAM fit)...

  AISURREY partition survey (target: nice-project × 1 GPU, need ≥48 GB VRAM)
    partition           status         gpus (free/total)   type                    vram    next free   pending
    ------------------------------------------------------------------------------------------------------
   *nice-project        READY          2/2                 nvidia_l40s             48G     now         0
    rtx_a6000_risk      READY          4/8                 nvidia_rtx_a6000        48G     now         2
    a100                READY          0/4                 nvidia_a100             80G     2h14m       3
    cogvis-project      [not ours]     2/2                 nvidia_rtx_a6000+       48G     now         0

  inferred VRAM need: 48 GB (from 3/4 scorers runnable at --gres=gpu:1; peak = 48 GB because scorers run sequentially)
  skipped at tp>--gres=gpu:1: tower/tower-plus-72b — runner will skip these at load time; re-submit with more GPUs if you need them.

  → nice-project: READY — proceed with pre-flight.
[6/6] sbatch --test-only...    ok: dry-run accepted
dry-run mode: pre-flight OK, not submitting.
```

If the recommender prefers a different partition, you get an interactive prompt:

```
[5/6] cluster probe (live capacity + VRAM fit)...
  ...survey table...
  → a100: CONTESTED — shape fits but 0 free GPUs right now.
  recommender prefers 'nice-project' over 'a100'.
Your target: a100
Recommender's ranking:
  1) nice-project (wait: now; tier=1; free now, group partition)
  2) rtx_a6000_risk (wait: now; tier=2; free now)
  3) a100 (wait: 2h14m; tier=5; no immediate capacity; est. wait 134 min)
Pick 1-3 or c to cancel (15s, no default):
```

Pick `1` / `2` / `3` to re-route, or `c` to cancel. No keypress within 15 s also cancels — there's no silent default.

Escape hatches:

- `--stay-on-target` — keep your original partition even if the probe prefers another. Good for deliberate "I want a100 for headline timing" runs.
- `SUBMIT_AUTO_ROUTE=1 scripts/submit.sh ...` — accept the #1 recommendation without prompting. Use in cron / unattended scripts.
- `SUBMIT_PROMPT_TIMEOUT=<seconds>` — override the 15 s default prompt timeout. Useful in CI where you want a shorter wait, or interactively when you need longer to decide.

## 2. Actually submit

```bash
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml
```

Output ends with `Submitted batch job 1234567`. Logs go to
`logs/slurm_<jobid>_<config>.{out,err}` in the repo root.

## 3. Override partition / GPU count / wall time

The default is `--partition=a100 --gres=gpu:1 --time=24:00:00`. Overrides
pass straight through to sbatch:

```bash
# 48 GB card (good for CometKiwi-XL, Tower-13B)
scripts/submit.sh configs/runs/surrey_legal_all_comet.yaml \
    -p rtx_a6000_risk --gres=gpu:1

# 2 GPUs on a100 for bigger Tower runs
scripts/submit.sh configs/runs/surrey_legal_tower_plus.yaml \
    --gres=gpu:2 --mem=128G --time=08:00:00

# Cheap smoke on 24 GB card
scripts/submit.sh configs/runs/example_quick.yaml -p 3090_risk
```

**Right-size after the first run.** Check
`outputs/<run_id>/summary.json::peak_gpu_memory_gb` and move to the
cheapest partition that covers it — see the table in `docs/AISURREY.md`.

## 4. All four Surrey domains at once

```bash
scripts/submit_aisurrey.sh                   # legal, general, tourism, health
scripts/submit_aisurrey.sh legal general     # just these two
scripts/submit_aisurrey.sh legal -p rtx_a6000_risk  # override partition
```

Each matrix submits independently. Results land under
`$SCRATCH/outputs/surrey_<domain>_full_matrix/`.

## 5. Monitor + post-mortem

```bash
squeue -u $USER
tail -f logs/slurm_<jobid>_<config>.err
sacct -j <jobid> --format=JobID,State,ExitCode,NodeList,Elapsed
```

`ExitCode=1:0` with no traceback usually means a flaky node. `submit.sh`
already excludes `aisurrey26`; if another node misbehaves, scancel and
resubmit.

## 6. Recomputing correlations

Useful if you want to try different correlation metrics or analyse a
subset (e.g. per language pair) without re-running inference:

```bash
mt-metrix correlate --run $SCRATCH/outputs/surrey_legal_full_matrix
```

## 7. Python API (optional)

If you need to submit from Python (e.g. a training loop that launches
evaluation jobs), use the wrapper directly — same pre-flight, same
excludes:

```python
from mt_metrix.submit.slurm import submit_via_wrapper

rc, job_id = submit_via_wrapper(
    "configs/runs/surrey_legal_cometkiwi.yaml",
    sbatch_args=["-p", "rtx_a6000_risk", "--gres=gpu:1"],
)
```

Internally this shells out to `scripts/submit.sh` so you inherit the full
pre-flight.

## 8. Split a matrix into per-model jobs

For very large runs, split per-scorer into separate configs so each job
gets its own resource plan:

```yaml
# configs/runs/surrey_legal_kiwi_only.yaml
run: {id: surrey_legal_kiwi_only}
dataset: !include ../datasets/surrey_legal.yaml
scorers:
  - ref: comet/wmt22-cometkiwi-da
  - ref: comet/wmt23-cometkiwi-da-xl
  - ref: comet/wmt23-cometkiwi-da-xxl

# configs/runs/surrey_legal_tower_plus.yaml
run: {id: surrey_legal_tower_plus}
dataset: !include ../datasets/surrey_legal.yaml
scorers:
  - ref: tower/tower-plus-2b
  - ref: tower/tower-plus-9b
  - ref: tower/tower-plus-72b
```

Submit each separately — different partitions / GPU counts, parallel jobs
on the queue.
