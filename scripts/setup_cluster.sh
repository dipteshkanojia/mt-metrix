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
ENV_NAME="mt-metrix"
PYTHON_VER="3.10"
TORCH_PIN="torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
yel()   { printf '\033[33m%s\033[0m\n' "$*" >&2; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }

# ------------------------------------------------- 1. scratch layout

green "=== 1. Scratch layout ==="
mkdir -p "$SCRATCH/models" "$SCRATCH/hf_cache" "$SCRATCH/outputs"
echo "  SCRATCH = $SCRATCH"

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

# ------------------------------------------------- 3. conda env

green "=== 3. Conda env '$ENV_NAME' ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  env '$ENV_NAME' exists, activating..."
    conda activate "$ENV_NAME"
else
    echo "  creating env '$ENV_NAME' (python $PYTHON_VER)..."
    conda create -n "$ENV_NAME" "python=$PYTHON_VER" -y
    conda activate "$ENV_NAME"
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
Env:       $ENV_NAME (activate with: conda activate $ENV_NAME)
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
