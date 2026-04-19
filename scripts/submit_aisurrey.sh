#!/usr/bin/env bash
# Convenience wrapper: submit all four Surrey domain full-matrix runs as
# separate sbatch jobs on AISURREY.
#
# Run from the repo root on the login node, with the conda env activated.

set -euo pipefail

if ! command -v mt-metrix >/dev/null; then
    echo "ERROR: mt-metrix not on PATH. Did you 'conda activate mt-metrix'?" >&2
    exit 1
fi

DOMAINS=("${@:-legal general tourism health}")

for domain in $DOMAINS; do
    config="configs/runs/surrey_${domain}_full_matrix.yaml"
    if [ ! -f "$config" ]; then
        echo "WARN: skipping $domain — $config not found" >&2
        continue
    fi
    echo "----------------------------------------------------------------"
    echo "Submitting $config"
    mt-metrix submit --config "$config"
done

echo
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Cancel a job with:               scancel <jobid>"
