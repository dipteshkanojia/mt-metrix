#!/usr/bin/env bash
# submit.sh — AISURREY-safe wrapper around sbatch for mt-metrix runs.
#
# Runs six pre-flight checks before submitting any job, and short-circuits
# on the cheap failures that usually cost 30+ minutes when skipped:
#
#   1. Config file + slurm script exist.
#   2. The partition named in the slurm script (or via -p) actually exists
#      on this cluster right now (sinfo). There is NO 'gpu' partition on
#      AISURREY — this check catches that typo instantly.
#   3. Conda env 'mt-metrix' exists (case-sensitive; check `conda env list`).
#   4. No duplicate of this config already queued / running under your user.
#   5. Cluster probe: scontrol-driven DETECT + COMPREHEND + ADVISE. Shows
#      free GPUs per partition, infers VRAM need from the config's scorers,
#      and recommends a wide-open alternative when the target is contested.
#      Runs before --test-only so the user sees live capacity before SLURM
#      makes its own decision. See scripts/cluster_probe.py.
#   6. sbatch --test-only dry-run: resolves partition/gres/mem/time against
#      current cluster state without actually queueing the job.
#
# It also soft-warns if the requested GPU count exceeds 4. Nodes expose up
# to 8 GPUs, but getting all 8 on one node is effectively impossible due to
# cluster contention. Default to 1 GPU; 4 is the soft cap.
#
# Only after all six pass does it submit with --exclude=aisurrey26 (flaky
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

# ---------------------------------------------------------------------------
# Queue-aware alternative-picking prompt. See
# docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md.
#
# _prompt_alternative(target_partition, alts_tsv_path, [timeout_default])
#   stdin:  user input ("1"/"2"/"3" or "c"); no keypress within
#           ${SUBMIT_PROMPT_TIMEOUT:-15}s = cancel, UNLESS
#           timeout_default is non-empty (set by the caller when the user
#           passed an explicit -p flag), in which case we return that
#           partition on timeout instead of cancelling.
#   stdout: chosen partition name on success
#   exit:   0 = chosen partition printed on stdout
#           7 = user cancelled (or timed out with no timeout_default)
#
# Test hook: SUBMIT_TEST_SKIP_PREFLIGHT=1 makes submit.sh return after
# defining helpers, so `bash tests/test_submit_prompt.bash` can source
# it and call the helper directly without running scontrol/sbatch.

_prompt_alternative() {
    local target="$1"
    local alts_tsv="$2"
    local timeout_default="${3:-}"
    local timeout="${SUBMIT_PROMPT_TIMEOUT:-15}"

    # Read TSV. Skip header; build an array of partitions with per-line
    # display strings for the prompt.
    local parts=()
    local labels=()
    while IFS=$'\t' read -r rank partition gpus_req wait_s tier reason; do
        [[ "$rank" == "rank" ]] && continue
        parts+=("$partition")
        local wait_human
        if [[ -z "$wait_s" ]]; then
            wait_human="wait: ?"
        elif [[ "$wait_s" == "0" ]]; then
            wait_human="wait: now"
        elif [[ "$wait_s" =~ ^[0-9]+$ ]]; then
            local hh=$((wait_s / 3600))
            local mm=$(((wait_s % 3600) / 60))
            if [[ "$hh" -gt 0 ]]; then
                wait_human="wait: ${hh}h${mm}m"
            else
                wait_human="wait: ${mm}m"
            fi
        else
            wait_human="wait: ?"
        fi
        labels+=("  $rank) $partition ($wait_human; tier=$tier; $reason)")
    done < "$alts_tsv"

    if [[ "${#parts[@]}" -eq 0 ]]; then
        echo "no alternatives available." >&2
        return 7
    fi

    # Auto-route: env var overrides interactive prompt, picks alt 1.
    if [[ "${SUBMIT_AUTO_ROUTE:-0}" == "1" ]]; then
        echo "${parts[0]}"
        return 0
    fi

    local prompt_tail
    if [[ -n "$timeout_default" ]]; then
        prompt_tail="${timeout}s, timeout keeps your explicit -p=${timeout_default}"
    else
        prompt_tail="${timeout}s, no default"
    fi
    {
        echo "Your target: $target"
        echo "Recommender's ranking:"
        for label in "${labels[@]}"; do
            echo "$label"
        done
        echo -n "Pick 1-${#parts[@]} or c to cancel (${prompt_tail}): "
    } >&2

    local choice=""
    if ! read -r -t "$timeout" choice; then
        echo >&2
        if [[ -n "$timeout_default" ]]; then
            echo "timed out; honouring explicit -p=${timeout_default}." >&2
            echo "$timeout_default"
            return 0
        fi
        echo "timed out; cancelling submission." >&2
        return 7
    fi
    case "$choice" in
        c|C) echo "cancelled." >&2; return 7 ;;
        [0-9]|[0-9][0-9])
            local idx=$((choice - 1))
            if [[ "$idx" -lt 0 || "$idx" -ge "${#parts[@]}" ]]; then
                echo "invalid choice '$choice'; cancelling." >&2
                return 7
            fi
            echo "${parts[$idx]}"
            return 0
            ;;
        *)
            echo "invalid choice '$choice'; cancelling." >&2
            return 7
            ;;
    esac
}

# ---------------------------------------------------------------------------
# FreeMem check. CfgTRES tells you what the scheduler is allowed to hand
# out; FreeMem tells you what the OS actually has free right now. SLURM's
# select/cons_tres plugin rejects allocations whose --mem exceeds FreeMem
# regardless of AllocMem. On 2026-04-24 a non-SLURM process was holding
# ~110GB of RAM on aisurrey35; CfgTRES still advertised mem=125G (so the
# CfgTRES-only probe said "fits"), FreeMem was 16456MB, and --mem=120G
# hard-rejected. The CfgTRES check alone classified that as transient
# contention. This helper closes the gap.
#
# _check_freemem(partition, requested_mem_mb)
#   Reads `scontrol show node -o` (or $SUBMIT_TEST_SCONTROL_OUTPUT if
#   set — test hook), filters to schedulable nodes in the target
#   partition, and checks whether ANY usable node has FreeMem >=
#   requested_mem_mb.
#   stdout: one-line diagnostic, e.g.
#     "ok FreeMem=65536MB on aisurrey35 (request 32000MB)"
#     "hard-fail all usable nodes in nice-project have FreeMem below
#      120000MB (max=16456MB on aisurrey35)"
#     "unknown scontrol unavailable or no usable nodes"
#   exit:  0 = fits, 1 = hard-fail, 2 = unknown (fall through to
#          existing transient-contention handling)

_check_freemem() {
    local partition="$1"
    local req_mb="$2"
    local raw=""
    if [[ -n "${SUBMIT_TEST_SCONTROL_OUTPUT:-}" ]]; then
        if [[ -f "$SUBMIT_TEST_SCONTROL_OUTPUT" ]]; then
            raw=$(<"$SUBMIT_TEST_SCONTROL_OUTPUT")
        else
            raw="$SUBMIT_TEST_SCONTROL_OUTPUT"
        fi
    elif command -v scontrol >/dev/null 2>&1; then
        raw=$(scontrol show node -o 2>/dev/null || true)
    fi
    # Trim leading/trailing whitespace — a pure-whitespace payload should
    # behave like an empty one (unknown), not like a node line.
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    if [[ -z "$raw" ]]; then
        echo "unknown scontrol unavailable"
        return 2
    fi

    local max_free=-1
    local max_free_node=""
    local seen_any_usable=0
    while IFS= read -r line; do
        [[ "$line" != *NodeName=* ]] && continue

        # Partitions field may be comma-separated (e.g. 3090,3090_risk).
        # Filter to nodes that actually belong to the target partition.
        local parts_field=""
        if [[ "$line" =~ Partitions=([^[:space:]]+) ]]; then
            parts_field="${BASH_REMATCH[1]}"
        fi
        case ",$parts_field," in
            *",$partition,"*) ;;
            *) continue ;;
        esac

        # State can be "IDLE", "MIXED", "ALLOCATED+DRAIN", "IDLE*"; drop
        # the trailing '*' (nodes mismatched with scheduler) and split on
        # '+' so any unusable sub-state knocks the node out.
        local state_field=""
        if [[ "$line" =~ State=([^[:space:]]+) ]]; then
            state_field="${BASH_REMATCH[1]}"
        fi
        state_field="${state_field%\*}"
        local primary_state="${state_field%%+*}"
        case "$primary_state" in
            IDLE|MIXED|ALLOCATED|COMPLETING) ;;
            *) continue ;;
        esac
        case "+$state_field+" in
            *+DRAIN+*|*+DRAINED+*|*+DRAINING+*|*+DOWN+*|*+FAIL+*|*+FAILING+*|*+MAINT+*|*+NO_RESPOND+*|*+POWER_DOWN+*|*+POWERED_DOWN+*|*+POWERING_DOWN+*|*+POWERING_UP+*|*+REBOOT+*|*+RESERVED+*|*+PLANNED+*)
                continue
                ;;
        esac

        local free_mb=""
        if [[ "$line" =~ FreeMem=([0-9]+) ]]; then
            free_mb="${BASH_REMATCH[1]}"
        fi
        [[ -z "$free_mb" ]] && continue

        local node_name=""
        if [[ "$line" =~ NodeName=([^[:space:]]+) ]]; then
            node_name="${BASH_REMATCH[1]}"
        fi

        seen_any_usable=1
        if (( free_mb > max_free )); then
            max_free="$free_mb"
            max_free_node="$node_name"
        fi
    done <<< "$raw"

    if [[ "$seen_any_usable" -eq 0 ]]; then
        echo "unknown no usable nodes with FreeMem in partition $partition"
        return 2
    fi

    if (( max_free >= req_mb )); then
        echo "ok FreeMem=${max_free}MB on ${max_free_node} (request ${req_mb}MB)"
        return 0
    fi
    echo "hard-fail all usable nodes in $partition have FreeMem below ${req_mb}MB (max=${max_free}MB on ${max_free_node})"
    return 1
}

# Early exit for the test harness. Defines the helper above and returns
# before running any of the pre-flight steps.
if [[ "${SUBMIT_TEST_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

# ---------------------------------------------------------------- inputs

if [[ $# -lt 1 ]]; then
    cat >&2 <<'EOF'
Usage: scripts/submit.sh <config.yaml> [--dry-run] [--stay-on-target] [sbatch overrides...]

Examples:
    scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml
    scripts/submit.sh configs/runs/example_quick.yaml -p 3090 --gres=gpu:1
    scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml --dry-run

--stay-on-target:
    Skip the interactive prompt if the cluster probe recommends a
    different partition. Respects the partition you explicitly chose.

SUBMIT_AUTO_ROUTE=1 scripts/submit.sh ...:
    Skip the interactive prompt and accept the probe's first
    recommendation. Intended for unattended scripts.

SUBMIT_PROMPT_TIMEOUT=<seconds> scripts/submit.sh ...:
    Override the interactive prompt's timeout (default 15s). No
    keypress within this window cancels the submission.

Every AISURREY sbatch goes through this wrapper. If you're tempted to run
sbatch directly: don't. That's how partition / env / node typos get past
the pre-flight and eat half a morning.
EOF
    exit 2
fi

CONFIG="$1"; shift
DRY_RUN=0
STAY_ON_TARGET=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --stay-on-target) STAY_ON_TARGET=1; shift ;;
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

# Convert a SLURM --mem value (e.g. 120G, 256G, 64000, 1T) into megabytes.
# Empty input returns empty string. Unrecognised suffix returns empty.
# SLURM's default unit is MB when no suffix is given.
_parse_mem_mb() {
    local raw="${1:-}"
    [[ -z "$raw" ]] && return 0
    if   [[ "$raw" =~ ^([0-9]+)T$ ]]; then echo "$(( ${BASH_REMATCH[1]} * 1024 * 1024 ))"
    elif [[ "$raw" =~ ^([0-9]+)G$ ]]; then echo "$(( ${BASH_REMATCH[1]} * 1024 ))"
    elif [[ "$raw" =~ ^([0-9]+)M$ ]]; then echo "${BASH_REMATCH[1]}"
    elif [[ "$raw" =~ ^([0-9]+)K$ ]]; then echo "$(( ${BASH_REMATCH[1]} / 1024 ))"
    elif [[ "$raw" =~ ^([0-9]+)$  ]]; then echo "${BASH_REMATCH[1]}"
    fi
}

# ---------------------------------------------------------------- check 1: config + slurm script exist

echo "[1/6] config file check..."
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
[[ -f "$SLURM_SCRIPT" ]] || fail "slurm script not found: $SLURM_SCRIPT (run from repo root)"
ok "config: $CONFIG"

# ---------------------------------------------------------------- check 2: partition exists

echo "[2/6] partition sanity check (no 'gpu' partition on AISURREY)..."
PARTITION=""
# USER_EXPLICIT_P: 1 when the caller passed -p / --partition= on the CLI.
# The recommender can pick a different partition than the user requested;
# if the user deliberately picked one, the timeout default in
# _prompt_alternative keeps their choice rather than cancelling.
USER_EXPLICIT_P=0
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        -p)            PARTITION="${EXTRA_ARGS[$((i+1))]:-}"; USER_EXPLICIT_P=1 ;;
        --partition=*) PARTITION="${EXTRA_ARGS[$i]#--partition=}"; USER_EXPLICIT_P=1 ;;
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

echo "[3/6] conda env check (prefix $CONDA_ENV_PREFIX)..."
if [ ! -f "$CONDA_ENV_PREFIX/bin/python" ]; then
    fail "conda env not found at $CONDA_ENV_PREFIX. Run: bash scripts/setup_cluster.sh"
fi
ok "conda env present at $CONDA_ENV_PREFIX"

# ---------------------------------------------------------------- check 4: no duplicate job

echo "[4/6] duplicate job check..."
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

# ---------------------------------------------------------------- check 5: cluster probe (DETECT + COMPREHEND + ADVISE)
#
# Runs scripts/cluster_probe.py to show live per-partition GPU availability,
# infer VRAM need from the config's scorers, and recommend a ready
# alternative if the target partition is contested or can't ever fit the
# job. The probe is stdlib-only Python so it runs here, BEFORE conda
# activate inside the job.
#
# Exit codes from cluster_probe.py:
#   0 → ready (proceed)
#   1 → no-fit: partition can never run this (fail hard)
#   2 → contested: partition fits but has no free GPUs right now (warn + grace)
#   3 → probe itself failed (scontrol missing etc. — proceed, let --test-only decide)

echo "[5/6] cluster probe (live capacity + VRAM fit)..."
CLUSTER_PROBE="scripts/cluster_probe.py"
if [[ -f "$CLUSTER_PROBE" ]] && command -v python3 >/dev/null 2>&1; then
    # Forward the same overrides the real sbatch will see, so the probe
    # evaluates the actual request (not just the in-file defaults).
    PROBE_ARGS=("--config" "$CONFIG" "--partition" "$PARTITION"
                "--slurm-script" "$SLURM_SCRIPT")
    if [[ -n "${GPU_COUNT:-}" ]] && [[ "${GPU_COUNT:-}" =~ ^[0-9]+$ ]]; then
        PROBE_ARGS+=("--gpus" "$GPU_COUNT")
    fi
    for arg in "${EXTRA_ARGS[@]:-}"; do
        case "$arg" in
            --mem=*)           PROBE_ARGS+=("--mem" "${arg#--mem=}") ;;
            --cpus-per-task=*) PROBE_ARGS+=("--cpus" "${arg#--cpus-per-task=}") ;;
        esac
    done
    # TSV sidecar: per-invocation temp file so stale data from a prior
    # failed run can't be misread. cluster_probe writes alternatives here.
    ALTS_TSV="$(mktemp)"
    trap 'rm -f "${ALTS_TSV:-}"' EXIT
    PROBE_ARGS+=("--tee-alternatives" "$ALTS_TSV")
    # We want the probe's human-readable output streamed to the user.
    set +e
    python3 "$CLUSTER_PROBE" "${PROBE_ARGS[@]}"
    PROBE_RC=$?
    set -e
    case "$PROBE_RC" in
        0)
            ok "cluster probe: target partition has capacity"
            # Even when target is READY, the recommender may prefer a
            # different partition (tier / a100-penalty). Surface that
            # choice unless the user said --stay-on-target or set
            # SUBMIT_AUTO_ROUTE to accept it unattended.
            if [[ "${STAY_ON_TARGET:-0}" == "1" ]]; then
                :
            elif [[ -f "$ALTS_TSV" ]]; then
                TOP_ALT=$(awk -F'\t' 'NR>1 && $1==1 {print $2; exit}' "$ALTS_TSV")
                if [[ -n "$TOP_ALT" ]] && [[ "$TOP_ALT" != "$PARTITION" ]]; then
                    yel "  recommender prefers '$TOP_ALT' over '$PARTITION'."
                    set +e
                    _PROMPT_DEFAULT=""
                    [[ "$USER_EXPLICIT_P" == "1" ]] && _PROMPT_DEFAULT="$PARTITION"
                    ROUTE_TO=$(_prompt_alternative "$PARTITION" "$ALTS_TSV" "$_PROMPT_DEFAULT")
                    PROMPT_RC=$?
                    set -e
                    if [[ "$PROMPT_RC" -eq 7 ]]; then
                        fail "user cancelled; not submitting"
                    elif [[ -n "$ROUTE_TO" ]] && [[ "$ROUTE_TO" != "$PARTITION" ]]; then
                        ok "re-routing submission to $ROUTE_TO"
                        _REROUTED=0
                        for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
                            case "${EXTRA_ARGS[$i]}" in
                                -p) EXTRA_ARGS[$((i+1))]="$ROUTE_TO"; _REROUTED=1 ;;
                                --partition=*) EXTRA_ARGS[$i]="--partition=$ROUTE_TO"; _REROUTED=1 ;;
                            esac
                        done
                        if [[ "$_REROUTED" -eq 0 ]]; then
                            EXTRA_ARGS+=("-p" "$ROUTE_TO")
                        fi
                        PARTITION="$ROUTE_TO"
                    fi
                fi
            fi
            ;;
        1)
            red "  cluster probe rejected the target partition (shape violation)."
            fail "target partition cannot run this shape; pick a different -p"
            ;;
        2)
            yel "  cluster probe: target partition is fully allocated right now."
            if [[ "${STAY_ON_TARGET:-0}" == "1" ]]; then
                yel "  --stay-on-target: skipping alternative prompt; staying on $PARTITION."
            elif [[ -f "$ALTS_TSV" ]]; then
                set +e
                _PROMPT_DEFAULT=""
                [[ "$USER_EXPLICIT_P" == "1" ]] && _PROMPT_DEFAULT="$PARTITION"
                ROUTE_TO=$(_prompt_alternative "$PARTITION" "$ALTS_TSV" "$_PROMPT_DEFAULT")
                PROMPT_RC=$?
                set -e
                if [[ "$PROMPT_RC" -eq 7 ]]; then
                    fail "user cancelled; not submitting"
                elif [[ -n "$ROUTE_TO" ]] && [[ "$ROUTE_TO" != "$PARTITION" ]]; then
                    ok "re-routing submission to $ROUTE_TO"
                    # Rewrite any -p / --partition= arg already in EXTRA_ARGS,
                    # else append -p <ROUTE_TO>.
                    _REROUTED=0
                    for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
                        case "${EXTRA_ARGS[$i]}" in
                            -p) EXTRA_ARGS[$((i+1))]="$ROUTE_TO"; _REROUTED=1 ;;
                            --partition=*) EXTRA_ARGS[$i]="--partition=$ROUTE_TO"; _REROUTED=1 ;;
                        esac
                    done
                    if [[ "$_REROUTED" -eq 0 ]]; then
                        EXTRA_ARGS+=("-p" "$ROUTE_TO")
                    fi
                    PARTITION="$ROUTE_TO"
                fi
            else
                yel "  (no alternatives sidecar; continuing with transient warning)"
            fi
            ;;
        3)
            yel "  warn: cluster probe couldn't query scontrol; proceeding to --test-only."
            ;;
        4)
            # Exit 4 = no partition structurally accepts this shape
            # (e.g. --gres=gpu:8 on a cluster whose biggest node has 4).
            # After the 2026-04-24 fix, check_fit also trusts live queue
            # evidence, so a fully-allocated partition with pending GPU
            # jobs no longer lands here — it lands in exit 2 (CONTESTED)
            # where submit.sh warns and queues. If you're still seeing
            # exit 4, the job's per-node shape exceeds the cluster's
            # physical ceiling; there is nothing to wait for.
            red "  cluster probe: no partition can fit this job shape at --gres=gpu:$GPU_COUNT."
            red "  (if the probe table shows every partition as NO-FIT with"
            red "   per-node GPU ceiling (0), scontrol may be missing Gres"
            red "   info; rerun the probe on a submit node or check"
            red "   scripts/cluster_probe.py's GPU_VRAM_GB map for a new card.)"
            fail "nothing fits; re-shape the job (scorers, gpus, mem)"
            ;;
        5)
            red "  cluster probe: target partition '$PARTITION' is in the blocklist (not ours)."
            fail "pick a partition that belongs to our group (start with nice-project)"
            ;;
        *)
            yel "  warn: cluster probe returned unexpected exit $PROBE_RC; proceeding."
            ;;
    esac
else
    yel "  warn: cluster probe unavailable (python3 or $CLUSTER_PROBE missing); skipping."
fi

# ---------------------------------------------------------------- check 6: sbatch --test-only
#
# sbatch --test-only green-lights only when the job could start IMMEDIATELY
# on some node in the target partition. On a single-node partition (e.g.
# nice-project = aisurrey35) that's frequently "no" even when the shape is
# fine, because all GPUs on the one node are currently allocated. We need
# to tell three outcomes apart:
#
#   (a) accepted                           → ok, submit
#   (b) transient: no free slot right now  → warn loudly, still submit
#                                            (sbatch will queue the job PD
#                                            and start when a slot frees)
#   (c) genuine shape violation            → fail hard, user must fix
#
# SLURM's error text is ambiguous between (b) and (c): both can print
# "Requested node configuration is not available". We disambiguate by
# reading CfgTRES for the partition's largest node and comparing against
# what the user is asking for. If any requested resource (--mem,
# --cpus-per-task, --gres=gpu:N) genuinely exceeds the node ceiling → (c).
# Otherwise it's (b), warn and continue.

echo "[6/6] sbatch --test-only (validates partition/gres/mem/time vs. live cluster)..."
if command -v sbatch >/dev/null 2>&1; then
    TESTONLY_OUT=$(sbatch --test-only --job-name="$JOBNAME" --exclude="$FLAKY_NODE" \
                          "${EXTRA_ARGS[@]}" "$SLURM_SCRIPT" "$CONFIG" 2>&1) && TESTONLY_RC=0 || TESTONLY_RC=$?
    if [[ "$TESTONLY_RC" -eq 0 ]]; then
        ok "dry-run accepted"
    elif echo "$TESTONLY_OUT" | grep -qiE 'Requested node configuration is not available|Required node not available|Nodes required for job are DOWN, DRAINED'; then
        # Could be (b) transient contention or (c) genuine shape violation.
        # Parse the effective request and compare against the partition's
        # largest CfgTRES to distinguish.
        REQ_MEM_MB=""
        REQ_CPUS=""
        REQ_GPUS=""
        # From EXTRA_ARGS (CLI overrides win over in-file defaults).
        for arg in "${EXTRA_ARGS[@]:-}"; do
            case "$arg" in
                --mem=*)           REQ_MEM_MB=$(_parse_mem_mb "${arg#--mem=}") ;;
                --cpus-per-task=*) REQ_CPUS="${arg#--cpus-per-task=}" ;;
                --gres=gpu:*)      _g="${arg#--gres=gpu:}"; REQ_GPUS="${_g##*:}" ;;
            esac
        done
        # Fall back to the in-file sbatch header for anything the CLI
        # didn't override.
        if [[ -z "$REQ_MEM_MB" ]]; then
            _m=$(awk -F= '/^#SBATCH[[:space:]]+--mem=/{print $2; exit}' "$SLURM_SCRIPT" \
                   | awk '{print $1}')
            REQ_MEM_MB=$(_parse_mem_mb "$_m")
        fi
        if [[ -z "$REQ_CPUS" ]]; then
            REQ_CPUS=$(awk -F= '/^#SBATCH[[:space:]]+--cpus-per-task=/{print $2; exit}' \
                         "$SLURM_SCRIPT" | awk '{print $1}')
        fi
        if [[ -z "$REQ_GPUS" ]]; then
            _g=$(awk -F= '/^#SBATCH[[:space:]]+--gres=gpu:/{print $2; exit}' "$SLURM_SCRIPT" \
                   | awk '{print $1}')
            REQ_GPUS="${_g##*:}"
        fi
        # Query the partition's max CfgTRES (largest per-node ceiling).
        # sinfo shape: '%m' = RealMemory MB, '%c' = CPUs, '%G' = gres list.
        # We take the max RealMemory across nodes in the partition as the
        # "biggest node" ceiling. It's a safe upper bound: if the request
        # exceeds this, no node can ever satisfy it.
        _BIG_MEM_MB=$(sinfo -h -p "$PARTITION" -N -o '%m' 2>/dev/null | sort -n | tail -1)
        _BIG_CPU=$(sinfo   -h -p "$PARTITION" -N -o '%c' 2>/dev/null | sort -n | tail -1)
        _BIG_GPU=$(sinfo   -h -p "$PARTITION" -N -o '%G' 2>/dev/null \
                     | grep -oE 'gpu:[a-zA-Z0-9_]+:[0-9]+|gpu:[0-9]+' \
                     | grep -oE '[0-9]+$' | sort -n | tail -1)
        SHAPE_BAD=0
        SHAPE_REASONS=()
        if [[ -n "$REQ_MEM_MB" ]] && [[ -n "$_BIG_MEM_MB" ]] \
             && (( REQ_MEM_MB > _BIG_MEM_MB )); then
            SHAPE_BAD=1
            SHAPE_REASONS+=("--mem=${REQ_MEM_MB}M > largest node RealMemory=${_BIG_MEM_MB}M")
        fi
        if [[ -n "$REQ_CPUS" ]] && [[ -n "$_BIG_CPU" ]] \
             && (( REQ_CPUS > _BIG_CPU )); then
            SHAPE_BAD=1
            SHAPE_REASONS+=("--cpus-per-task=$REQ_CPUS > largest node CPUs=$_BIG_CPU")
        fi
        if [[ -n "$REQ_GPUS" ]] && [[ -n "$_BIG_GPU" ]] \
             && (( REQ_GPUS > _BIG_GPU )); then
            SHAPE_BAD=1
            SHAPE_REASONS+=("--gres=gpu:$REQ_GPUS > largest node GPUs=$_BIG_GPU")
        fi
        if [[ "$SHAPE_BAD" -eq 1 ]]; then
            red "  sbatch --test-only rejected the submission — genuine shape violation:"
            for reason in "${SHAPE_REASONS[@]}"; do
                red "    $reason"
            done
            red "  raw error: $TESTONLY_OUT"
            fail "request exceeds the partition's per-node ceilings; fix and retry"
        fi

        # CfgTRES says fits — but sbatch may still be rejecting because
        # FreeMem (OS-visible free RAM) on every usable node is below
        # --mem. This happens when a non-SLURM process holds RAM: CfgTRES
        # still advertises the full capacity, AllocMem says 0, but the
        # select/cons_tres plugin refuses the allocation. Seen live on
        # 2026-04-24 (aisurrey35 FreeMem=16456MB, --mem=120G, hard
        # rejected). Without this check, submit.sh would invite an Enter
        # keypress and the real sbatch would reject a second time.
        FREEMEM_OUT=""
        FREEMEM_RC=2
        if [[ -n "$REQ_MEM_MB" ]]; then
            set +e
            FREEMEM_OUT=$(_check_freemem "$PARTITION" "$REQ_MEM_MB")
            FREEMEM_RC=$?
            set -e
        fi
        if [[ "$FREEMEM_RC" -eq 1 ]]; then
            red "  sbatch --test-only rejected the submission — non-SLURM process is holding RAM:"
            red "    $FREEMEM_OUT"
            red "  SLURM's select/cons_tres plugin compares --mem to FreeMem (OS-visible free"
            red "  RAM), not AllocMem, so this is a HARD rejection, not transient contention."
            red "  raw error: $TESTONLY_OUT"
            if [[ -f "${ALTS_TSV:-}" ]]; then
                TOP_ALT=$(awk -F'\t' -v p="$PARTITION" \
                          'NR>1 && $1==1 && $2!=p {print $2; exit}' "$ALTS_TSV")
                if [[ -z "$TOP_ALT" ]]; then
                    TOP_ALT=$(awk -F'\t' -v p="$PARTITION" \
                              'NR>1 && $2!=p {print $2; exit}' "$ALTS_TSV")
                fi
                if [[ -n "$TOP_ALT" ]]; then
                    red "  Recommended: re-run with -p $TOP_ALT"
                fi
            fi
            fail "partition '$PARTITION' is effectively unavailable; retry on a different -p"
        fi

        yel "  warn: sbatch --test-only couldn't place this job RIGHT NOW:"
        yel "        $TESTONLY_OUT"
        yel "        Shape fits the partition's per-node ceilings (mem=${REQ_MEM_MB}M ≤ ${_BIG_MEM_MB}M,"
        yel "        cpu=${REQ_CPUS} ≤ ${_BIG_CPU}, gpu=${REQ_GPUS} ≤ ${_BIG_GPU}), so this is"
        yel "        transient — all GPUs in '$PARTITION' are probably allocated right now."
        if [[ "$FREEMEM_RC" -eq 0 ]]; then
            yel "        FreeMem check: ${FREEMEM_OUT}"
        fi
        yel "        Check: sinfo -p $PARTITION -o '%n %C %m %G %t'"
        yel "        The real sbatch will queue the job in PD and start when a slot frees."
        yel "        Ctrl-C within 5s to abort, or Enter to submit anyway..."
        read -r -t 5 || true
        ok "proceeding past transient resource-contention warning"
    else
        red "  sbatch --test-only rejected the submission with a non-transient error:"
        red "    $TESTONLY_OUT"
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
