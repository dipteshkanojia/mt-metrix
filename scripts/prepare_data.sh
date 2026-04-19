#!/usr/bin/env bash
# Pre-cache the Surrey NLP domain QE datasets into $HF_HOME so sbatch jobs
# don't download on first run. Idempotent.
#
# Requires: huggingface_hub installed, HF_TOKEN set (Surrey datasets are
# public but the datasets library still benefits from an authed session).

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

for repo in \
    surrey-nlp/Legal-QE \
    surrey-nlp/General-QE \
    surrey-nlp/Tourism-QE \
    surrey-nlp/health-QE
do
    echo "--- caching $repo ---"
    python -c "from datasets import load_dataset; load_dataset('$repo', split='test')"
done

echo "Dataset cache populated."
