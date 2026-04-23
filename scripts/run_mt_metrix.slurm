#!/bin/bash
# mt-metrix SLURM job — AISURREY cluster.
#
# Usage (preferred, via pre-flight wrapper):
#   scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
#
# Or directly (skips pre-flight — not recommended):
#   sbatch --exclude=aisurrey26 scripts/run_mt_metrix.slurm \
#          configs/runs/surrey_legal_cometkiwi.yaml
#
# Override partition / gres / mem on the CLI for smaller or bigger jobs:
#   scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
#   scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --mem=256G
#
# The config path is passed as $1; the job name and log file are derived
# from the config filename (set by submit.sh) or default to the script name.
#
# AISURREY partition cheat-sheet (verify live with `sinfo`):
#   nice-project    2× L40s 48 GB, 3-day limit. NICE-GROUP RESERVED — this is
#                   our default. Zero queue wait, 16 CPU + 128 GB RAM per
#                   node. Covers every full-matrix scorer EXCEPT Tower-72B
#                   (which needs tp=4 on a100) at the right batch sizes.
#   a100            A100 80 GB, 3-day limit. Headline / Tower-72B only —
#                   ~2.2× fairshare weight vs 3090, reserve for what
#                   genuinely needs >48 GB or publication runs.
#   rtx8000         Quadro 48 GB, 3-day limit. Fallback when nice-project full.
#   rtx_a6000_risk  RTX A6000 48 GB, preemptible. 48 GB fallback.
#   l40s_risk       L40S 48 GB, preemptible. Same silicon as nice-project but
#                   preemptible — prefer nice-project first.
#   3090 / 3090_risk RTX 3090 24 GB. Smoke + small ablations.
#   2080ti          RTX 2080 Ti 11 GB. Catch-all; avoid for training.
#   debug           4-hour cap. Quick sanity checks.
# There is NO partition called `gpu` — every sbatch line must name one
# explicitly.
#
# Right-size the GPU by scorer family:
#   sacrebleu (CPU)                       any partition; pin 1 GPU anyway
#   COMET-base / Kiwi-DA (~1.5 GB)        nice-project at batch=64
#   COMET-XL / XCOMET-XL (3.5B, ~7 GB)    nice-project at batch=16
#   COMET-XXL / XCOMET-XXL (10.7B, ~22 GB) nice-project at batch=4 (auto-
#                                         downshift from catalogue's 8) OR
#                                         a100 at batch=8 for speed.
#   Tower-2B / 7B / Mistral-7B            nice-project tp=1 (or transformers)
#   Tower-9B (18 GB fp16)                 nice-project tp=1
#   Tower-13B (26 GB fp16)                nice-project tp=1 on a single L40s
#   Tower-72B (144 GB fp16)               a100 gpu:4 tp=4 — SEPARATE SUBMIT,
#                                         will be skipped on any other config
# Check `summary.json::peak_gpu_memory_gb` from a prior run to right-size.
#
# Right-size CPU RAM (--mem) — separate from GPU VRAM and the root cause
# of the 2026-04-19 matrix OOMs. unbabel-comet + pytorch-lightning
# materialise the full model in CPU RAM first, then move it to GPU.
# Scorers load SEQUENTIALLY (each Scorer.unload() frees CPU + GPU before
# the next loads), so the cap is per-scorer not cumulative:
#   sacrebleu / CometKiwi-DA (~1.5 GB) / Tower-2B (vllm):   ~8 GB peak
#   CometKiwi-XL / XCOMET-XL-QE (~12 GB) / Tower-7B:        ~30 GB peak
#   CometKiwi-XXL / XCOMET-XXL-QE (~40 GB fp32 state dict)  ~60 GB peak
#   Tower-13B (26 GB fp16 + tokeniser + KV):                ~50 GB peak
#   Tower-72B:                                              ~180 GB peak
#
# PARTITION-SPECIFIC CEILINGS (SLURM rejects oversubscribing these):
#   nice-project (aisurrey35):   128 GB host RAM total.  --mem=128G is MAX.
#   rtx8000 / rtx_a6000_risk /   typically 128-256 GB depending on node.
#     l40s_risk / 3090:          check with `scontrol show node <name>`.
#   a100:                        >=256 GB host RAM on all a100 nodes.
#   debug:                       check the specific node.
# Requesting more than the node has triggers ``allocation failure:
# Requested node configuration is not available`` at `sbatch --test-only`
# time. The submit.sh wrapper surfaces that as FAIL before submission.
#
# Plan per scorer family given the ceilings above:
#   Any single scorer on nice-project:      128G in-file default is correct
#                                           (and is the node ceiling, so the
#                                           wrapper won't let you ask more).
#   Full matrix on nice-project:            same 128G — works because scorers
#                                           are sequential (peak ~60 GB).
#   Tower-72B on a100:                      256G override is safe (a100
#                                           nodes have it) and recommended
#                                           because 72B load briefly spikes.
#   Headline / paper-quality on a100:       use the a100 ceiling (256G+) for
#                                           headroom; no functional need.
#
# Watch for `slurmstepd: Detected 1 oom_kill event` with NO CUDA traceback —
# that's the kernel OOM killer enforcing the cgroup limit. If this fires
# on nice-project at 128G, pivot the affected scorer (usually XXL COMET
# or Tower-13B) to a100 via a dedicated run config.
#
# Common submit patterns:
#   # Default full-matrix on nice-project — no flags needed, the in-file
#   # 128G is both the right number AND the node ceiling.
#   scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml
#
#   # Tower-72B follow-up (ONLY partition where it fits, tp=4):
#   scripts/submit.sh configs/runs/surrey_legal_tower72b.yaml \
#       -p a100 --gres=gpu:4 --mem=256G
#
#   # Headline / paper-quality on A100 80 GB (keeps XXL COMET at batch=8
#   # instead of the nice-project downshift to 4):
#   scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml \
#       -p a100 --gres=gpu:1 --mem=256G

#SBATCH --partition=nice-project     # NICE-group dedicated 2× L40s 48 GB; override with -p on CLI
#SBATCH --gres=gpu:1                 # override with --gres=gpu:N (N=4 only for Tower-72B on a100)
#SBATCH --cpus-per-task=8            # dataloader workers
#SBATCH --mem=128G                   # = nice-project node ceiling; a100 can go to 256G via --mem override
#SBATCH --time=24:00:00              # 24h; most runs finish in <4h
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch run_mt_metrix.slurm <config_path>}"
JOBNAME=$(basename "$CONFIG" .yaml)

echo "===== mt-metrix SLURM job ====="
echo "Config:    $CONFIG"
echo "Job name:  $JOBNAME"
echo "Node:      ${SLURMD_NODENAME:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Mem (cg):  ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Date:      $(date)"
echo "==============================="

# --- Scratch layout (compute-visible; $HOME and /vol/research are NOT) ---
export SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}"
mkdir -p "$SCRATCH/models" "$SCRATCH/hf_cache" "$SCRATCH/outputs" "$SCRATCH/tmp" "$SCRATCH/pip_cache" logs

# --- HuggingFace cache (must NOT live under $HOME — too small, not always compute-visible) ---
export HF_HOME="$SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export COMET_CACHE="$SCRATCH/models/comet"

# --- Redirect /tmp and pip cache to scratch so on-the-fly weight downloads
# --- + any surprise pip fetch don't fill the tiny login-node /tmp.
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"

# --- Ray (used by vLLM for tensor-parallel on Tower-13B+) needs its session
# --- dir on a SHORT path because Linux caps AF_UNIX socket paths at 107 bytes
# --- and Ray builds .../ray/session_<timestamp>/sockets/plasma_store inside
# --- whatever TMPDIR points at. $SCRATCH/tmp/... is ~113 bytes by the time
# --- Ray is done, so we point Ray specifically at a short /tmp path. This
# --- only holds small control sockets, not object store data.
export RAY_TMPDIR="/tmp/ray_${USER:-dk0023}"
mkdir -p "$RAY_TMPDIR"

# --- HF auth for gated COMET / Tower models ---
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token:  loaded from ~/.hf_token"
else
    echo "HF token:  NOT FOUND (~/.hf_token missing)."
    echo "           Gated models (wmt22-cometkiwi-da, XCOMET-*, all Tower)"
    echo "           will fail to download. If this run uses only public models"
    echo "           (sacrebleu, wmt20/22-comet-da) you can ignore this warning."
fi

# --- Resource sanity dump (these are the first things to check after an OOM) ---
echo
echo "----- scratch / RAM / disk sanity -----"
echo "SCRATCH=$SCRATCH"
df -h "$SCRATCH" 2>/dev/null || echo "(df -h failed on $SCRATCH)"
df -h /tmp 2>/dev/null | tail -1
free -h 2>/dev/null || echo "(free -h unavailable)"
echo "--------------------------------------"
echo

# --- Conda env (prefix path on scratch — user volume can't hold torch+vllm+comet) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_env"

# --- Torch sanity (cluster driver is CUDA 12.06; torch 2.4.0+cu121 is the pin) ---
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())"

# --- wandb offline on compute nodes (compute nodes are internet-limited) ---
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- Run ---
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mt-metrix score --config "$CONFIG" --output-root "$SCRATCH/outputs"

echo "Done at $(date)"
