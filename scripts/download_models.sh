#!/usr/bin/env bash
# Pre-download all COMET + Tower checkpoints into $SCRATCH/models so sbatch
# jobs don't race on the HF hub. Run this ONCE on the login node after
# accepting licences and putting your HF token in ~/.hf_token.
#
# Usage:
#   bash scripts/download_models.sh                       # everything
#   bash scripts/download_models.sh comet                 # just COMET
#   bash scripts/download_models.sh tower                 # just Tower
#   bash scripts/download_models.sh --ref comet/wmt22-cometkiwi-da   # one ref

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}"
DEST="$SCRATCH/models"
mkdir -p "$DEST"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
    echo "WARN: ~/.hf_token not found — gated models will fail to download."
fi

echo "Downloading to: $DEST"

case "${1:-all}" in
    all)
        mt-metrix download --family comet --to "$DEST"
        mt-metrix download --family tower --to "$DEST"
        ;;
    comet|tower)
        mt-metrix download --family "$1" --to "$DEST"
        ;;
    --ref)
        shift
        mt-metrix download --ref "$1" --to "$DEST"
        ;;
    *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
esac

echo "Done. Model weights under $DEST"
