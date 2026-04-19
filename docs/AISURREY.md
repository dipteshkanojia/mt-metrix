# Running mt-metrix on AISURREY

AISURREY cluster specifics layered on top of
`~/Documents/Claude/agent-context/aisurrey-cluster.md` (global cluster ops).

## Cluster conventions we rely on

- **Partition:** `a100` (also `a100_long` for ≥8h jobs).
- **Node exclusion:** `--exclude=aisurrey26` — it has a flaky GPU.
- **Filesystem:** `/vol/research/<group>` is login-only, NOT visible on
  compute nodes. Use `/mnt/fast/nobackup/scratch4weeks/$USER/` for scratch.
- **Torch pin:** 2.4.1+cu121 in the `mt-metrix` conda env. Later torch
  builds have a known incompatibility with this cluster's driver.
- **Conda:** `~/miniconda3/` is the default; submit scripts source it.

## One-time setup

```bash
ssh aisurrey

# 1. create scratch root
export SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
mkdir -p $SCRATCH/{models,hf-cache,outputs}

# 2. conda env
conda create -n mt-metrix python=3.10 -y
conda activate mt-metrix

# 3. install torch FIRST (pinned for this cluster)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 4. clone + install mt-metrix
git clone https://github.com/dipteshkanojia/mt-metrix.git
cd mt-metrix
pip install -e ".[comet,tower]"

# 5. HuggingFace token (for gated COMET models)
echo "hf_xxx..." > ~/.hf_token
chmod 600 ~/.hf_token
```

## Accepting licences (one-time, on the web)

Several models are gated and will fail to load without licence acceptance
under the same HF account whose token sits in `~/.hf_token`:

- `Unbabel/wmt22-cometkiwi-da`
- `Unbabel/wmt23-cometkiwi-da-xl`
- `Unbabel/wmt23-cometkiwi-da-xxl`
- `Unbabel/XCOMET-XL`, `Unbabel/XCOMET-XXL`
- All Tower models (TowerInstruct, TowerBase, Tower-Plus) — CC-BY-NC-4.0

Visit each model page on HuggingFace Hub and click "Agree and access".

## Pre-downloading model weights

Doing this once up-front prevents every sbatch job from racing the HF hub
cache on first use.

```bash
# all COMET weights into $SCRATCH/models
mt-metrix download --family comet --to $SCRATCH/models

# all Tower weights (this is a lot of disk — ~400GB for the 72B family)
mt-metrix download --family tower --to $SCRATCH/models

# or just one model
mt-metrix download --ref comet/wmt22-cometkiwi-da --to $SCRATCH/models
```

## Submitting a run

`mt-metrix submit` renders an sbatch script under
`outputs/submitted/<run_id>.sbatch`, sizes it based on the largest scorer in
the run, and submits it.

```bash
# dry-run: write the script, don't submit — useful for reviewing resources
mt-metrix submit --config configs/runs/surrey_legal_cometkiwi.yaml --dry-run

# actually submit
mt-metrix submit --config configs/runs/surrey_legal_cometkiwi.yaml

# override resources (e.g. bump wall time)
mt-metrix submit --config configs/runs/surrey_legal_full_matrix.yaml \
    --gpus 4 --time 12:00:00
```

### Size heuristic

`submit/slurm.py`'s `plan_for` picks resources by the largest model in the
scorers list:

| Trigger substring in `model_id`        | Template                 | GPUs | Time     |
|----------------------------------------|--------------------------|------|----------|
| `72b`                                  | `tower_72b.sbatch`       | 4    | 08:00:00 |
| `13b`, `9b`                            | `tower_13b.sbatch`       | 2    | 04:00:00 |
| `xxl`                                  | `comet_xxl.sbatch`       | 1    | 04:00:00 |
| `xl`                                   | `comet_xl.sbatch`        | 1    | 02:00:00 |
| `7b`, `2b`                             | `tower_7b.sbatch`        | 1    | 02:00:00 |
| everything else                        | `comet_small.sbatch`     | 1    | 01:00:00 |

If the heuristic picks wrong, override with `--gpus` / `--time` / `--partition`.

## Scratch layout

```
/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/
├── models/           # pre-downloaded HF weights
├── hf-cache/         # HF_HOME (datasets + any on-the-fly downloads)
│   └── huggingface/
└── outputs/          # --output-root on the cluster
    └── <run_id>/     # config.yaml, segments.tsv, segments.jsonl, summary.json
```

Submitted sbatch jobs export:
```
SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
HF_HOME=$SCRATCH/hf-cache
HUGGINGFACE_HUB_CACHE=$SCRATCH/hf-cache
TRANSFORMERS_CACHE=$SCRATCH/hf-cache
COMET_CACHE=$SCRATCH/models/comet
HF_TOKEN=$(cat ~/.hf_token)   # if the file exists
```

…so model weights land in scratch, not in `$HOME` (which has a 20GB quota
you will blow through immediately).

## Running the flagship matrix

End-to-end, from a fresh login to results:

```bash
ssh aisurrey
cd mt-metrix
git pull                     # in case of updates

conda activate mt-metrix

# submit all four domain matrices
for domain in legal general tourism health; do
    mt-metrix submit \
        --config configs/runs/surrey_${domain}_full_matrix.yaml
done

# monitor
squeue -u $USER
```

Outputs land under `$SCRATCH/outputs/<run_id>/`. Use `mt-metrix correlate
--run <path>` to regenerate correlations without re-running inference.

## When things go wrong

- **`401 Unauthorized` on model download** → HF token missing or licence not
  accepted for that specific model. `cat ~/.hf_token`, then visit the model
  page on HF and accept.
- **`torch.cuda.OutOfMemory` on cometkiwi-xxl** → add `overrides: {batch_size: 4}`
  to that scorer entry in the run config.
- **`ImportError: vllm`** → you installed without the `[tower]` extra. Run
  `pip install -e ".[tower]"` or disable Tower scorers in the run config.
- **sbatch job stuck in `PD` forever** → a100 queue is full; check
  `sinfo -p a100`.
- **aisurrey26 is assigned anyway** → make sure your script has
  `#SBATCH --exclude=aisurrey26` (all mt-metrix-rendered scripts do).
- **`HF_HOME` ends up in `$HOME/.cache`** → your conda activate hook is
  overwriting the env vars. Put `export HF_HOME=$SCRATCH/hf-cache` AFTER
  `conda activate`.
