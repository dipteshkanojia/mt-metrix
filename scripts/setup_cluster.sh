#!/usr/bin/env bash
# One-time AISURREY setup for mt-metrix.
#
# Run this interactively on the login node (aisurrey-submit01), NOT as a
# SLURM job. Idempotent — safe to re-run after `git pull`.
#
# What it does:
#   1. Clone / update the repo under /mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix.
#   2. Create / update the `mt-metrix` conda env (python 3.10, torch 2.4.1+cu121).
#   3. Install mt-metrix with [comet,tower] extras.
#   4. Create scratch layout (models/, hf_cache/, outputs/).
#   5. Check for ~/.hf_token and warn if missing.
#   6. Run the fast test suite as a smoke test.
#
# After this, submit real jobs with:
#   scripts/submit.sh configs/runs/<something>.yaml
#
# Prereqs on the cluster:
#   - conda on PATH (miniconda or anaconda)
#   - network access to github.com + huggingface.co + pypi.org from login node

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/dipteshkanojia/mt-metrix.git}"  # override with REPO_URL=git@... if SSH key configured
SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}"
WORKDIR="$SCRATCH/repo"
# Conda env lives on scratch as a prefix path, NOT in /mnt/fast/nobackup/users/$USER
# (which has a small quota). mt-metrix + torch 2.4.1 + vllm + comet deps = ~10 GB;
# the user volume fills up fast. Scratch is purged every 4 weeks, but re-running
# this script recreates the env idempotently, so that's fine.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$SCRATCH/conda_env}"
PYTHON_VER="3.10"
TORCH_PIN="torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
yel()   { printf '\033[33m%s\033[0m\n' "$*" >&2; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }

# ------------------------------------------------- 1. scratch layout

green "=== 1. Scratch layout ==="
mkdir -p "$SCRATCH/models" "$SCRATCH/hf_cache" "$SCRATCH/outputs" "$SCRATCH/tmp" "$SCRATCH/pip_cache"
echo "  SCRATCH = $SCRATCH"

# Redirect pip's build isolation and cache to scratch.
# /tmp on AISURREY login nodes is small (few GB); pip install of vllm or
# flash-attn downloads multi-GB CUDA wheels into its isolated build env and
# fills /tmp. PIP_CACHE_DIR on scratch also keeps $HOME quota safe.
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
echo "  TMPDIR           = $TMPDIR"
echo "  PIP_CACHE_DIR    = $PIP_CACHE_DIR"

# ------------------------------------------------- 2. clone / pull

green "=== 2. Repo clone / update ==="
if [ -d "$WORKDIR/.git" ]; then
    echo "  repo exists at $WORKDIR, pulling..."
    cd "$WORKDIR"
    git fetch origin
    git pull --ff-only origin main
else
    echo "  cloning $REPO_URL → $WORKDIR"
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

# ------------------------------------------------- 3. conda env (prefix path on scratch)

green "=== 3. Conda env at $CONDA_ENV_PREFIX ==="
source "$(conda info --base)/etc/profile.d/conda.sh"

# Warn if an old 'mt-metrix' env exists on the user volume — likely leftover
# from a previous failed install that hit the quota. User can remove it to
# free space; we continue either way using the scratch prefix env.
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "mt-metrix"; then
    yel "  NOTE: an old 'mt-metrix' env exists in the default conda envs dir"
    yel "        (likely partial from a previous user-volume ENOSPC failure)."
    yel "        To reclaim space, run:    conda env remove -n mt-metrix -y"
    yel "        Continuing with scratch-prefix env below."
fi

if [ -f "$CONDA_ENV_PREFIX/bin/python" ]; then
    echo "  env exists, activating..."
    conda activate "$CONDA_ENV_PREFIX"
else
    echo "  creating env at $CONDA_ENV_PREFIX (python $PYTHON_VER)..."
    conda create -p "$CONDA_ENV_PREFIX" "python=$PYTHON_VER" -y
    conda activate "$CONDA_ENV_PREFIX"
fi

# torch pin first — cluster driver is CUDA 12.06; 2.4.1+cu121 is the known-good build
echo "  installing torch (2.4.1+cu121)..."
pip install --quiet $TORCH_PIN

echo "  installing mt-metrix with [comet,tower] extras..."
pip install --quiet -e ".[comet,tower]"

# ------------------------------------------------- 4. HF token

green "=== 4. HuggingFace token ==="
if [ -f "$HOME/.hf_token" ]; then
    echo "  ~/.hf_token present ($(wc -c < "$HOME/.hf_token") bytes)"
    chmod 600 "$HOME/.hf_token" 2>/dev/null || true
else
    yel "  ~/.hf_token MISSING. Create it to access gated COMET / Tower models:"
    yel "    echo 'hf_xxx...' > ~/.hf_token && chmod 600 ~/.hf_token"
    yel "  Gated models you likely need to accept licences for at huggingface.co:"
    yel "    Unbabel/wmt22-cometkiwi-da, wmt23-cometkiwi-da-xl, wmt23-cometkiwi-da-xxl,"
    yel "    Unbabel/XCOMET-XL, Unbabel/XCOMET-XXL"
fi

# ------------------------------------------------- 5. smoke test

green "=== 5. Smoke test (fast pytest suite) ==="
if ! pytest tests/ -q --no-header 2>&1 | tail -3; then
    red "  pytest failed — investigate before submitting jobs"
    exit 1
fi

# ------------------------------------------------- 6. next steps

green "=== Setup complete ==="
cat <<EOF

Scratch:   $SCRATCH
Repo:      $WORKDIR
Env:       $CONDA_ENV_PREFIX
           activate with: conda activate $CONDA_ENV_PREFIX
Outputs:   $SCRATCH/outputs/
HF cache:  $SCRATCH/hf_cache/

Next steps:
  cd $WORKDIR
  # dry-run one job to validate the slurm plumbing end-to-end:
  scripts/submit.sh configs/runs/example_quick.yaml --dry-run
  # submit a real smoke job on a small GPU:
  scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
  # submit the flagship Legal matrix on a100:
  scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml

Monitor with:
  squeue -u \$USER
  sacct -j <jobid> --format=JobID,State,ExitCode,NodeList,Elapsed

EOF
