#!/usr/bin/env bash
# submit.sh — AISURREY-safe wrapper around sbatch for mt-metrix runs.
#
# Runs five pre-flight checks before submitting any job, and short-circuits
# on the cheap failures that usually cost 30+ minutes when skipped:
#
#   1. Config file + slurm script exist.
#   2. The partition named in the slurm script (or via -p) actually exists
#      on this cluster right now (sinfo). There is NO 'gpu' partition on
#      AISURREY — this check catches that typo instantly.
#   3. Conda env 'mt-metrix' exists (case-sensitive; check `conda env list`).
#   4. No duplicate of this config already queued / running under your user.
#   5. sbatch --test-only dry-run: resolves partition/gres/mem/time against
#      current cluster state without actually queueing the job.
#
# It also soft-warns if the requested GPU count exceeds 4. Nodes expose up
# to 8 GPUs, but getting all 8 on one node is effectively impossible due to
# cluster contention. Default to 1 GPU; 4 is the soft cap.
#
# Only after all five pass does it submit with --exclude=aisurrey26 (flaky
# node, silent 1:0 exits observed 2026-04). See
# ~/Documents/Claude/agent-context/aisurrey-cluster.md and
# ~/Documents/Claude/agent-context/aisurrey-deploy.md for the deeper cluster
# context.
#
# Usage:
#     scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
#     scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
#     scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run
#
# The first arg is always the config path. Any remaining args pass straight
# to sbatch before the slurm script, so `-p`, `--gres`, `--time` etc. all
# work as overrides.
#
# --dry-run runs only the pre-flight + sbatch --test-only; does not submit.

set -euo pipefail

# ---------------------------------------------------------------- inputs

if [[ $# -lt 1 ]]; then
    cat >&2 <<'EOF'
Usage: scripts/submit.sh <config.yaml> [--dry-run] [sbatch overrides...]

Examples:
    scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
    scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
    scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run

Every AISURREY sbatch goes through this wrapper. If you're tempted to run
sbatch directly: don't. That's how partition / env / node typos get past
the pre-flight and eat half a morning.
EOF
    exit 2
fi

CONFIG="$1"; shift
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

SLURM_SCRIPT="scripts/run_mt_metrix.slurm"
# Conda env is a prefix path on scratch (user volume can't hold torch+vllm+comet).
# Override via CONDA_ENV_PREFIX=... if you moved the env elsewhere.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix/conda_env}"
FLAKY_NODE="aisurrey26"     # silent 1:0 exits 2026-04

red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yel()   { printf '\033[33m%s\033[0m\n' "$*" >&2; }

fail() { red "FAIL: $*"; exit 1; }
ok()   { green "  ok: $*"; }

# ---------------------------------------------------------------- check 1: config + slurm script exist

echo "[1/5] config file check..."
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
[[ -f "$SLURM_SCRIPT" ]] || fail "slurm script not found: $SLURM_SCRIPT (run from repo root)"
ok "config: $CONFIG"

# ---------------------------------------------------------------- check 2: partition exists

echo "[2/5] partition sanity check (no 'gpu' partition on AISURREY)..."
PARTITION=""
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        -p)            PARTITION="${EXTRA_ARGS[$((i+1))]:-}" ;;
        --partition=*) PARTITION="${EXTRA_ARGS[$i]#--partition=}" ;;
    esac
done
if [[ -z "$PARTITION" ]]; then
    PARTITION=$(awk -F= '/^#SBATCH[[:space:]]+--partition=/{print $2; exit}' \
                    "$SLURM_SCRIPT" | awk '{print $1}')
fi
[[ -n "$PARTITION" ]] || fail "could not determine target partition"

if [[ "$PARTITION" == "gpu" ]]; then
    fail "partition=gpu but AISURREY has NO 'gpu' partition. Use a100/rtx8000/rtx_a6000_risk/l40s_risk/3090/3090_risk/2080ti/debug."
fi

if command -v sinfo >/dev/null 2>&1; then
    if ! sinfo -h -p "$PARTITION" -o '%P' 2>/dev/null | grep -q .; then
        fail "partition '$PARTITION' does not exist on this cluster. Run 'sinfo' for the live list."
    fi
    ok "partition '$PARTITION' exists"
else
    yel "  warn: sinfo not on PATH; skipping live partition check (are you on aisurrey-submit01?)"
fi

# ---------------------------------------------------------------- check 3: conda env (prefix path on scratch)

echo "[3/5] conda env check (prefix $CONDA_ENV_PREFIX)..."
if [ ! -f "$CONDA_ENV_PREFIX/bin/python" ]; then
    fail "conda env not found at $CONDA_ENV_PREFIX. Run: bash scripts/setup_cluster.sh"
fi
ok "conda env present at $CONDA_ENV_PREFIX"

# ---------------------------------------------------------------- check 4: no duplicate job

echo "[4/5] duplicate job check..."
JOBNAME=$(basename "$CONFIG" .yaml)
if command -v squeue >/dev/null 2>&1; then
    DUP=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -xc "$JOBNAME" || true)
    if [[ "${DUP:-0}" -gt 0 ]]; then
        yel "  warn: $DUP existing job(s) named '$JOBNAME' in your queue."
        yel "        Ctrl-C to abort, or press Enter within 5s to submit another..."
        read -r -t 5 || true
    else
        ok "no duplicates of '$JOBNAME' in queue"
    fi
else
    yel "  warn: squeue not on PATH; skipping duplicate check"
fi

# ---------------------------------------------------------------- soft cap: GPU count

GPU_COUNT=""
for arg in "${EXTRA_ARGS[@]:-}"; do
    case "$arg" in
        --gres=gpu:*)
            _g="${arg#--gres=gpu:}"
            GPU_COUNT="${_g##*:}"
            ;;
    esac
done
if [[ -z "$GPU_COUNT" ]]; then
    _g=$(awk -F= '/^#SBATCH[[:space:]]+--gres=gpu:/{print $2; exit}' \
              "$SLURM_SCRIPT" | awk '{print $1}')
    GPU_COUNT="${_g##*:}"
fi
if [[ -n "$GPU_COUNT" ]] && [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && (( GPU_COUNT > 4 )); then
    yel "  warn: requesting $GPU_COUNT GPUs — AISURREY soft cap is 4."
    yel "        Getting all 8 on one node is effectively impossible due to contention."
    yel "        Ctrl-C to abort, or press Enter within 5s to submit anyway..."
    read -r -t 5 || true
elif [[ -n "$GPU_COUNT" ]] && [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
    ok "gpu count: $GPU_COUNT (within 4-GPU soft cap)"
fi

# ---------------------------------------------------------------- check 5: sbatch --test-only

echo "[5/5] sbatch --test-only (validates partition/gres/mem/time vs. live cluster)..."
if command -v sbatch >/dev/null 2>&1; then
    if sbatch --test-only --job-name="$JOBNAME" --exclude="$FLAKY_NODE" \
              "${EXTRA_ARGS[@]}" "$SLURM_SCRIPT" "$CONFIG" >/dev/null 2>&1; then
        ok "dry-run accepted"
    else
        red "  sbatch --test-only rejected the submission. Running once more with stderr visible:"
        sbatch --test-only --job-name="$JOBNAME" --exclude="$FLAKY_NODE" \
               "${EXTRA_ARGS[@]}" "$SLURM_SCRIPT" "$CONFIG" || true
        fail "fix the submission before retrying"
    fi
else
    yel "  warn: sbatch not on PATH; cannot dry-run (are you on aisurrey-submit01?)"
fi

# ---------------------------------------------------------------- submit

if [[ "$DRY_RUN" == "1" ]]; then
    green "dry-run mode: pre-flight OK, not submitting."
    exit 0
fi

echo
green "pre-flight OK. Submitting..."
echo "  config:    $CONFIG"
echo "  job-name:  $JOBNAME"
echo "  partition: $PARTITION"
echo "  exclude:   $FLAKY_NODE"
echo "  extra:     ${EXTRA_ARGS[*]:-<none>}"
echo

sbatch --job-name="$JOBNAME" --exclude="$FLAKY_NODE" \
    "${EXTRA_ARGS[@]}" "$SLURM_SCRIPT" "$CONFIG"
