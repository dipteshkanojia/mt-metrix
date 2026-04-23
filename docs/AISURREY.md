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
  partition. Defaults to `nice-project` (NICE-group dedicated 2× L40s
  48 GB, 128 GB CPU RAM, effectively zero queue contention); see the
  cheat-sheet in `scripts/run_mt_metrix.slurm`. `a100` is reserved for
  the Tower-72B follow-up (`configs/runs/surrey_legal_tower72b.yaml`)
  and headline / paper-quality runs — override with `-p a100 --gres=gpu:N`.
- **`aisurrey26` is flaky** (silent `1:0` exits, 2026-04).
  `scripts/submit.sh` adds `--exclude=aisurrey26` to every submission.
- **`/vol/research/...` is login-only, invisible to compute nodes.**
  All compute-visible work lives under
  `/mnt/fast/nobackup/scratch4weeks/$USER/` or
  `/mnt/fast/nobackup/users/$USER/`.
- **torch pin:** `torch==2.4.0+cu121`. vllm 0.6.x, torchvision 0.19.0
  and xformers 0.0.27.post2 all hard-pin torch==2.4.0; 2.4.1 triggers a
  pip-resolver downgrade mid-install.
- **Conda env name is `mt-metrix`** (check casing with `conda env list`).
- **HF cache** redirected in the sbatch header, not only in `.bashrc`.

## One-time setup (idempotent)

`scripts/setup_cluster.sh` is the single entry point for first-time setup.
It will:

1. Create `/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/{models,hf_cache,outputs}`.
2. Clone the repo to `$SCRATCH/mt-metrix/repo` (or `git pull` if it exists).
3. Create the `mt-metrix` conda env with Python 3.10 and `torch==2.4.0+cu121`.
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
`sbatch` with six pre-flight checks (see
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

Pre-flight step [5/6] is a **cluster probe** (`scripts/cluster_probe.py`).
It parses `scontrol show node -o` to build a live per-partition picture
of free GPU capacity, infers VRAM need from the config's scorers, and
flags the target partition as:

- **READY** — free GPUs right now, proceed.
- **CONTESTED** — shape fits but no free GPUs this instant; the probe
  recommends a wide-open alternative and submit.sh gives you a 5s grace
  to Ctrl-C and re-submit with `-p <alternative>`. On AISURREY, `sbatch`
  rejects (does NOT queue) when the partition has zero free GPUs, so
  this warning is load-bearing: proceeding anyway risks a hard refusal
  at the scheduler.
- **NO-FIT** — no node in the partition can ever run this (e.g. asking
  for 80 GB VRAM on `nice-project`'s 48 GB L40s). Hard fail.

The probe is standalone (stdlib only) and can be run ad-hoc to survey
the cluster:

```bash
# Survey everything using the full-matrix config's inferred VRAM need:
python3 scripts/cluster_probe.py \
    --config configs/runs/surrey_legal_full_matrix.yaml

# JSON for scripting:
python3 scripts/cluster_probe.py \
    --config configs/runs/surrey_legal_tower72b.yaml \
    --partition a100 --gpus 4 --json
```

## Right-sizing: pick the smallest GPU that fits

`nice-project` is the default because it's our group's dedicated
2× L40s 48 GB node (zero queue wait) and covers every full-matrix
scorer except Tower-72B. `a100` is reserved for the 72B follow-up and
publication runs. After your first run, check
`outputs/<run_id>/summary.json::peak_gpu_memory_gb` and use the cheapest
partition that covers it:

| Peak memory | Smallest partition that fits      | `-p` override        |
|-------------|-----------------------------------|----------------------|
| ≤ 10 GB     | `2080ti` (11 GB)                  | `-p 2080ti`          |
| ≤ 22 GB     | `3090` / `3090_risk` (24 GB)      | `-p 3090_risk`       |
| ≤ 44 GB     | `nice-project` (2× L40s 48 GB, default) / `rtx8000` / `rtx_a6000_risk` / `l40s_risk` | default / `-p rtx_a6000_risk` |
| ≤ 76 GB     | `a100` (80 GB)                    | `-p a100`            |

Approximate mt-metrix sizing (verify with `summary.json::peak_gpu_memory_gb`):

- BLEU / chrF++ / TER: CPU-only — use any partition.
- COMET-base / CometKiwi-DA: <10 GB — fits on 2080ti, but queue is slow;
  3090 or nice-project are usually faster end-to-end.
- CometKiwi-XL / XCOMET-XL: ~20 GB — 3090 works, 48 GB card is cosier.
- CometKiwi-XXL / XCOMET-XXL: ~40 GB peak; catalogue default is
  `batch_size: 8` which OOMs on 48 GB L40s. The COMET scorer
  auto-downshifts to `batch_size: 4` on <60 GB cards at runtime
  (`_resolve_xxl_batch_size` in `scorers/comet.py`) and leaves
  `batch_size: 8` intact on A100 80 GB. No config change needed.
- Tower-2B / 7B / Mistral-7B (vLLM or HF): fits nice-project at tp=1.
- Tower-9B / 13B (vLLM): ~18 GB / ~26 GB fp16 — fits nice-project at tp=1.
- Tower-Plus-72B: 144 GB fp16 — genuinely needs `-p a100 --gres=gpu:4`
  with tp=4. Use `configs/runs/surrey_legal_tower72b.yaml` as the
  follow-up run once the nice-project full-matrix job has landed.

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

# or one at a time (default partition is nice-project; the in-file
# --mem=128G is already the node ceiling, don't override it upward —
# aisurrey35 only has 128 GB host RAM and SLURM will reject --mem=256G
# with "Requested node configuration is not available"):
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml

# Tower-72B follow-up (the ONLY partition where it fits, tp=4).
# a100 nodes have >=256 GB host RAM, so --mem=256G is safe HERE:
scripts/submit.sh configs/runs/surrey_legal_tower72b.yaml \
    -p a100 --gres=gpu:4 --mem=256G
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
- **`torch.cuda.OutOfMemory` on CometKiwi-XXL** → the scorer should
  auto-downshift to `batch_size: 4` on <60 GB cards. If you're still
  OOMing (e.g. on a 24 GB 3090 even at batch=4), add
  `overrides: {batch_size: 2}` (or 1) to that scorer entry in the run
  config. On A100 80 GB no override is needed — batch=8 runs as-is.
- **`slurmstepd: Detected 1 oom_kill event` with no CUDA traceback** →
  cgroup / host-RAM OOM. On `nice-project` the node ceiling is 128 GB so
  you can't just bump `--mem` — pivot the OOMing scorer to a100 via a
  dedicated run config (the a100 nodes have >=256 GB and you can pass
  `-p a100 --mem=256G`).
- **`allocation failure: Requested node configuration is not available`**
  at `sbatch --test-only` time → you asked for more resources than the
  target partition's largest node can schedule. On `nice-project`
  (aisurrey35): RealMemory=128 GB but MemSpecLimit reserves 8 GB for
  the OS, so `CfgTRES=mem=125G` is the hard schedulable ceiling.
  Anything over 125G — including the old `--mem=128G` default — is
  rejected. The in-file default is now `--mem=120G`. Check a node's
  live ceiling with `scontrol show node <name> | grep CfgTRES`. For
  workloads that genuinely need >125 GB host RAM, switch partition
  with `-p a100 --mem=256G`.
- **`ImportError: vllm`** → installed without `[tower]` extra. Run
  `pip install -e ".[tower]"` or remove Tower scorers from the run.
- **Job stuck in `PD` forever** → `a100` queue is full. `sinfo -p a100`
  and consider a 48 GB alternative (`-p rtx_a6000_risk`).
- **`sbatch: error: Batch job submission failed: Requested node
  configuration is not available`** → the target partition has zero free
  GPUs right now AND AISURREY rejects rather than queues (unlike most
  SLURM clusters). Run the cluster probe to pick an alternative:
  `python3 scripts/cluster_probe.py --config <your config>`. It prints
  every partition's free GPU count and recommends a wide-open one.
  `scripts/submit.sh` now runs this as pre-flight [5/6] so you see the
  recommendation before the hard refusal.
- **Job exits `1:0` with no traceback** → suspect node. `submit.sh`
  excludes `aisurrey26`; if it still happens, check
  `sacct -j <id> --format=JobID,State,ExitCode,NodeList`.
- **`HF_HOME` ends up in `$HOME/.cache`** → your shell hook overrides the
  env var after `conda activate`. The sbatch script exports `HF_HOME`
  *after* `conda activate` already, so this should not happen inside
  SLURM jobs.
