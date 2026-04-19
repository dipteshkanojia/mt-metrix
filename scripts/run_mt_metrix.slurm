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
# Override partition / gres on the CLI for smaller or bigger GPUs:
#   scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
#   scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --gres=gpu:2
#
# The config path is passed as $1; the job name and log file are derived
# from the config filename (set by submit.sh) or default to the script name.
#
# AISURREY partition cheat-sheet (verify live with `sinfo`):
#   a100            A100-80GB,   3-day limit. Headline partition.
#   rtx8000         Quadro 48GB, 3-day limit. A100 backup.
#   rtx_a6000_risk  RTX A6000 48GB, preemptible. Good A100 alternative for <48GB.
#   l40s_risk       L40S 48GB, preemptible. Modern Ada.
#   3090 / 3090_risk RTX 3090 24GB. Smoke + small ablations.
#   2080ti          RTX 2080Ti 11GB. Catch-all; avoid for training.
#   debug           4-hour cap. Quick sanity checks.
# There is NO partition called `gpu` — every sbatch line must name one
# explicitly.
#
# Right-size the GPU: COMET-base / CometKiwi-DA / BLEU all fit on 24 GB.
# CometKiwi-XL needs ~40 GB (48 GB card). Tower-7B inference fits on a 48 GB
# card with vLLM; Tower-13B fits on 2× 48 GB or 1× A100. Only Tower-72B and
# COMET-XXL actually need A100. Check summary.json::peak_gpu_memory_gb from
# a prior run to pick the smallest partition that covers your workload.

#SBATCH --partition=a100             # override with -p on the CLI
#SBATCH --gres=gpu:1                 # override with --gres=gpu:N
#SBATCH --cpus-per-task=8            # dataloader workers
#SBATCH --mem=64G                    # CPU RAM; bump for vLLM on big Tower
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
echo "Date:      $(date)"
echo "==============================="

# --- Scratch layout (compute-visible; $HOME and /vol/research are NOT) ---
export SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}"
mkdir -p "$SCRATCH/models" "$SCRATCH/hf_cache" "$SCRATCH/outputs" logs

# --- HuggingFace cache (must NOT live under $HOME — too small, not always compute-visible) ---
export HF_HOME="$SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export COMET_CACHE="$SCRATCH/models/comet"

# --- HF auth for gated COMET / Tower models ---
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

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
