# 04 — Submitting to AISURREY

`mt-metrix submit` renders an sbatch script that wraps `mt-metrix score`. The
same code path runs locally and on the cluster — submission is just SLURM
glue around it.

Read `docs/AISURREY.md` for the full cluster runbook. This example is the
short version.

## 0. One-time setup (on the login node)

```bash
ssh aisurrey

# env + install
conda create -n mt-metrix python=3.10 -y && conda activate mt-metrix
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
cd ~ && git clone https://github.com/dipteshkanojia/mt-metrix.git
cd mt-metrix && pip install -e ".[comet,tower]"

# scratch layout
export SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
mkdir -p $SCRATCH/{models,hf-cache,outputs}

# HF token (gated COMET + all Tower)
echo "hf_xxx..." > ~/.hf_token
chmod 600 ~/.hf_token
```

## 1. Pre-download weights (one-off)

```bash
conda activate mt-metrix

mt-metrix download --family comet --to $SCRATCH/models
mt-metrix download --family tower --to $SCRATCH/models     # large — ~400GB for 72B
```

## 2. Dry-run the submission

`--dry-run` writes the sbatch file under `outputs/submitted/<run_id>.sbatch`
without calling `sbatch`. Review it to confirm resources make sense.

```bash
mt-metrix submit \
    --config configs/runs/surrey_legal_full_matrix.yaml \
    --dry-run

cat outputs/submitted/surrey_legal_full_matrix.sbatch
```

Look for:
- `#SBATCH --partition=a100`
- `#SBATCH --exclude=aisurrey26`
- Correct `--gres=gpu:N` and `--time=HH:MM:SS`
- `export HF_HOME="$SCRATCH/hf-cache"` and `export COMET_CACHE=...`

## 3. Actually submit

```bash
mt-metrix submit --config configs/runs/surrey_legal_full_matrix.yaml
```

Output:
```
SLURM script: /home/you/mt-metrix/outputs/submitted/surrey_legal_full_matrix.sbatch
sbatch return code: 0; job id: 1234567
```

Monitor:

```bash
squeue -u $USER
tail -f outputs/slurm-logs/slurm-1234567.out
```

## 4. Override resources

The heuristic picks resources from the biggest model in the run. Override:

```bash
mt-metrix submit \
    --config configs/runs/surrey_legal_full_matrix.yaml \
    --gpus 4 \
    --time 12:00:00 \
    --partition a100_long
```

## 5. All four domains in one go

```bash
for domain in legal general tourism health; do
    mt-metrix submit --config configs/runs/surrey_${domain}_full_matrix.yaml
done
```

Each job runs independently. Results land under
`$SCRATCH/outputs/surrey_<domain>_full_matrix/`.

## 6. Recomputing correlations

Useful if you want to try different correlation metrics, or analyse a
subset (e.g. per language pair) without re-running inference:

```bash
mt-metrix correlate --run $SCRATCH/outputs/surrey_legal_full_matrix
```

## Split the matrix across jobs

For very large runs, split per-scorer or per-language-pair into separate
files so each job can be tuned:

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

Submit each separately — different resource plans, parallel jobs on the
queue.
