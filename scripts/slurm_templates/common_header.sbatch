#!/usr/bin/env bash
# AISURREY common header — source this before the job body.
#
# Provides:
#   $SCRATCH           → /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix
#   HF_HOME / HF cache → $SCRATCH/hf-cache
#   COMET_CACHE        → $SCRATCH/models/comet
#   HF_TOKEN           → from ~/.hf_token if present (required for gated COMET)
#
# Usage (inside an sbatch script):
#   source /path/to/repo/scripts/slurm_templates/common_header.sbatch

set -euo pipefail

export SCRATCH=${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}
mkdir -p "$SCRATCH"/{models,hf-cache,outputs}

export HF_HOME="$SCRATCH/hf-cache"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf-cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf-cache"
export COMET_CACHE="$SCRATCH/models/comet"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN=$(cat "$HOME/.hf_token")
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Conda env
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate mt-metrix

echo "host:    $(hostname)"
echo "scratch: $SCRATCH"
echo "HF_HOME: $HF_HOME"
if command -v nvidia-smi >/dev/null; then
  echo "gpus:    $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
fi
