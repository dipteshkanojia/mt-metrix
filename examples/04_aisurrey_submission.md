# 04 — Submitting to AISURREY

The canonical submit path is `scripts/submit.sh` — a bash wrapper that
runs five pre-flight checks, calls `sbatch --test-only` to validate the
plan against the live cluster, and submits with `--exclude=aisurrey26`.

Read `docs/AISURREY.md` for the full cluster runbook, `docs/SESSION_HANDOFF.md`
for the fresh-session briefing. This example is the short version.

## 0. One-time cluster setup (idempotent)

```bash
ssh aisurrey
# Clones the repo into scratch, creates conda env with torch 2.4.1+cu121,
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

`--dry-run` runs the five pre-flight checks and `sbatch --test-only`, but
does NOT queue the job. Use this whenever you're unsure about a config
change or a partition override.

```bash
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
```

The wrapper prints its checks as it goes:
```
[1/5] config file check...     ok: configs/runs/surrey_legal_full_matrix.yaml
[2/5] partition sanity check...ok: partition 'a100' exists
[3/5] conda env check...       ok: conda env 'mt-metrix' present
[4/5] duplicate job check...   ok: no duplicates of 'surrey_legal_full_matrix' in queue
[5/5] sbatch --test-only...    ok: dry-run accepted
dry-run mode: pre-flight OK, not submitting.
```

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
