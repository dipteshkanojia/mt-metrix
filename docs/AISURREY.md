# Running mt-metrix on AISURREY

AISURREY-specific runbook for mt-metrix. Layered on top of the cluster-wide
notes in `~/Documents/Claude/agent-context/aisurrey-cluster.md` (topology)
and `~/Documents/Claude/agent-context/aisurrey-deploy.md` (deploy SOP).

## 30-second version

```bash
# one-time, from aisurrey-submit01:
mkdir -p /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
cd       /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
git clone https://github.com/dipteshkanojia/mt-metrix.git repo    # SSH requires key setup; HTTPS works out of the box on AISURREY
bash repo/scripts/setup_cluster.sh

# every subsequent session:
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
scripts/submit.sh configs/runs/<some-config>.yaml
```

Everything below is explanation of what those commands do and why.

## Cluster conventions we rely on

From `aisurrey-cluster.md` / `aisurrey-deploy.md`:

- **No `gpu` partition exists.** Every sbatch line must name a real
  partition. Defaults to `a100`; see the cheat-sheet in
  `scripts/run_mt_metrix.slurm`.
- **`aisurrey26` is flaky** (silent `1:0` exits, 2026-04).
  `scripts/submit.sh` adds `--exclude=aisurrey26` to every submission.
- **`/vol/research/...` is login-only, invisible to compute nodes.**
  All compute-visible work lives under
  `/mnt/fast/nobackup/scratch4weeks/$USER/` or
  `/mnt/fast/nobackup/users/$USER/`.
- **torch pin:** `torch==2.4.1+cu121`. Newer / older builds cause
  silent NCCL/SM mismatches.
- **Conda env name is `mt-metrix`** (check casing with `conda env list`).
- **HF cache** redirected in the sbatch header, not only in `.bashrc`.

## One-time setup (idempotent)

`scripts/setup_cluster.sh` is the single entry point for first-time setup.
It will:

1. Create `/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/{models,hf_cache,outputs}`.
2. Clone the repo to `$SCRATCH/mt-metrix/repo` (or `git pull` if it exists).
3. Create the `mt-metrix` conda env with Python 3.10 and `torch==2.4.1+cu121`.
4. `pip install -e ".[comet,tower]"`.
5. Warn if `~/.hf_token` is missing (needed for gated COMET / Tower models).
6. Run the fast pytest suite as a smoke test.

Run it interactively on the submit node — NOT as a SLURM job:

```bash
ssh aisurrey
bash /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo/scripts/setup_cluster.sh
```

## Accepting model licences (one-time, on the web)

Several COMET and Tower checkpoints are gated and will return 401 without
licence acceptance under the same HF account whose token lives in
`~/.hf_token`:

- `Unbabel/wmt22-cometkiwi-da`
- `Unbabel/wmt23-cometkiwi-da-xl`
- `Unbabel/wmt23-cometkiwi-da-xxl`
- `Unbabel/XCOMET-XL`, `Unbabel/XCOMET-XXL`
- All Tower models (TowerInstruct, TowerBase, Tower-Plus) — CC-BY-NC-4.0

Visit each model page on HuggingFace Hub and click "Agree and access".

## Submitting jobs — the canonical path

**`scripts/submit.sh` is the only supported submit path.** It wraps
`sbatch` with five pre-flight checks (see
`aisurrey-deploy.md#pre-flight-checklist`):

```bash
scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
```

The first argument is always the config path. Any remaining arguments pass
verbatim to `sbatch`, so `-p`, `--gres`, `--time`, `--mem` all work as
overrides.

The wrapper auto-adds `--exclude=aisurrey26` and sets `--job-name` to the
config's basename (so logs end up at `logs/slurm_<jobid>_<config>.out`).

## Right-sizing: pick the smallest GPU that fits

`a100` is the default, but often overkill. After your first run, check
`outputs/<run_id>/summary.json::peak_gpu_memory_gb` and use the cheapest
partition that covers it:

| Peak memory | Smallest partition that fits      | `-p` override        |
|-------------|-----------------------------------|----------------------|
| ≤ 10 GB     | `2080ti` (11 GB)                  | `-p 2080ti`          |
| ≤ 22 GB     | `3090` / `3090_risk` (24 GB)      | `-p 3090_risk`       |
| ≤ 44 GB     | `rtx8000` / `rtx_a6000_risk` / `l40s_risk` (48 GB) | `-p rtx_a6000_risk` |
| ≤ 76 GB     | `a100` (80 GB)                    | default              |

Approximate mt-metrix sizing (verify with `summary.json::peak_gpu_memory_gb`):

- BLEU / chrF++ / TER: CPU-only — use any partition.
- COMET-base / CometKiwi-DA: <10 GB — fits on 2080ti, but queue is slow;
  3090 is usually faster end-to-end.
- CometKiwi-XL / XCOMET-XL: ~20 GB — 3090 works, 48 GB card is cosier.
- CometKiwi-XXL / XCOMET-XXL: ~40 GB — 48 GB card minimum.
- Tower-7B (vLLM): ~18 GB — 3090 works.
- Tower-13B (vLLM): ~28 GB — 48 GB card.
- Tower-Plus-72B: needs A100 with `--gres=gpu:2+` and a tensor-parallel vLLM config.

**4-GPU soft cap.** `submit.sh` warns if you request >4 GPUs — getting 8
on one node is effectively impossible due to contention.

## Scratch layout

```
/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/
├── repo/                        # this repo (cloned by setup_cluster.sh)
├── models/                      # pre-downloaded HF weights (optional)
│   └── comet/                   # COMET_CACHE lives here
├── hf_cache/                    # HF_HOME (datasets + on-the-fly downloads)
│   ├── hub/
│   ├── transformers/            # TRANSFORMERS_CACHE
│   └── datasets/                # HF_DATASETS_CACHE
└── outputs/                     # scoring run outputs (--output-root)
    └── <run_id>/
        ├── config.yaml
        ├── segments.tsv
        ├── segments.jsonl
        ├── summary.json
        └── run.log
```

`scripts/run_mt_metrix.slurm` exports these paths in the sbatch header:

```
SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
HF_HOME=$SCRATCH/hf_cache
HUGGINGFACE_HUB_CACHE=$HF_HOME
TRANSFORMERS_CACHE=$HF_HOME/transformers
HF_DATASETS_CACHE=$HF_HOME/datasets
COMET_CACHE=$SCRATCH/models/comet
HF_TOKEN=$(cat ~/.hf_token)     # if the file exists
```

## Pre-downloading weights (optional)

`scripts/submit.sh` leaves downloads to the first job that uses each
model. That's fine once `HF_HOME` is on scratch (first-use latency is a
one-off). If you want to warm the cache deliberately:

```bash
mt-metrix download --family comet --to $SCRATCH/models
mt-metrix download --ref comet/wmt22-cometkiwi-da --to $SCRATCH/models
```

Tower-family downloads are large (Tower-Plus-72B is >130 GB); pull only
what you plan to run.

## Running the flagship matrices

End-to-end, from scratch:

```bash
ssh aisurrey
cd /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/repo
git pull
conda activate /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/conda_env

# all four Surrey domain matrices (legal / general / tourism / health)
scripts/submit_aisurrey.sh

# or one at a time, with a partition override:
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml -p a100
```

Monitor:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,ExitCode,NodeList,Elapsed
```

Outputs land under `$SCRATCH/outputs/<run_id>/`. Regenerate correlations
without re-scoring:

```bash
mt-metrix correlate --run $SCRATCH/outputs/<run_id>
```

## When things go wrong

- **`sbatch: error: invalid partition specified: gpu`** → you called
  `sbatch` directly. Use `scripts/submit.sh`; it catches this before
  submission.
- **`EnvironmentNameNotFound: mt-metrix`** → env casing mismatch. Run
  `conda env list` and use the exact name.
- **`401 Unauthorized` on model download** → HF token missing or licence
  not accepted for that model. `cat ~/.hf_token`, then visit the model
  page on HF and accept.
- **`torch.cuda.OutOfMemory` on CometKiwi-XXL** → add
  `overrides: {batch_size: 4}` (or 2) to that scorer entry in the run
  config.
- **`ImportError: vllm`** → installed without `[tower]` extra. Run
  `pip install -e ".[tower]"` or remove Tower scorers from the run.
- **Job stuck in `PD` forever** → `a100` queue is full. `sinfo -p a100`
  and consider a 48 GB alternative (`-p rtx_a6000_risk`).
- **Job exits `1:0` with no traceback** → suspect node. `submit.sh`
  excludes `aisurrey26`; if it still happens, check
  `sacct -j <id> --format=JobID,State,ExitCode,NodeList`.
- **`HF_HOME` ends up in `$HOME/.cache`** → your shell hook overrides the
  env var after `conda activate`. The sbatch script exports `HF_HOME`
  *after* `conda activate` already, so this should not happen inside
  SLURM jobs.
