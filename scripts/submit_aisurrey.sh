#!/usr/bin/env bash
# submit_aisurrey.sh — submit the four Surrey full-matrix runs in one go.
#
# Thin loop over scripts/submit.sh (the pre-flight-checked submit path).
# Call from the repo root on aisurrey-submit01 with the conda env active.
#
# Usage:
#   scripts/submit_aisurrey.sh                         # all four domains
#   scripts/submit_aisurrey.sh legal general           # just these two
#   scripts/submit_aisurrey.sh legal -p rtx_a6000_risk # override partition

set -euo pipefail

SUBMIT="scripts/submit.sh"
[[ -x "$SUBMIT" ]] || { echo "ERROR: $SUBMIT not executable (run from repo root)" >&2; exit 1; }

# Split args into domains (positional until the first '-' arg) and sbatch overrides
DOMAINS=()
OVERRIDES=()
for arg in "$@"; do
    if [[ "$arg" == -* ]] || [[ ${#OVERRIDES[@]} -gt 0 ]]; then
        OVERRIDES+=("$arg")
    else
        DOMAINS+=("$arg")
    fi
done
[[ ${#DOMAINS[@]} -eq 0 ]] && DOMAINS=(legal general tourism health)

for domain in "${DOMAINS[@]}"; do
    config="configs/runs/surrey_${domain}_full_matrix.yaml"
    if [[ ! -f "$config" ]]; then
        echo "WARN: skipping $domain — $config not found" >&2
        continue
    fi
    echo "=============================================================="
    echo "Submitting $config"
    echo "=============================================================="
    "$SUBMIT" "$config" "${OVERRIDES[@]:-}"
    echo
done

cat <<'EOF'

All jobs submitted (pre-flight-checked). Monitor with:
  squeue -u $USER
Cancel with:
  scancel <jobid>
EOF
