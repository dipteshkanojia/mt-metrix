#!/usr/bin/env bash
# Smoke tests for submit.sh's _check_freemem helper.
#
# Context — 2026-04-24 incident: sbatch rejected a submission with
# "Requested node configuration is not available" even though CfgTRES on
# the only idle node in the partition was 125G and we asked --mem=120G.
# The cause was ``FreeMem=16456MB`` — a non-SLURM process was holding
# ~110GB of RAM, and SLURM's select/cons_tres plugin rejects allocations
# whose --mem exceeds FreeMem regardless of what AllocMem shows. The
# existing CfgTRES-only comparison in submit.sh classified the rejection
# as transient contention and invited the user to press Enter to submit;
# the real sbatch then hard-rejected. _check_freemem closes that gap by
# reading FreeMem (OS-visible free memory) from scontrol and treating
# FreeMem < --mem on every usable node as a HARD rejection.
#
# The test sources submit.sh with SUBMIT_TEST_SKIP_PREFLIGHT=1 (same
# pattern as test_submit_prompt.bash) so the file only DEFINES helpers
# and returns. Each test then invokes _check_freemem directly with a
# canned scontrol payload piped in via the SUBMIT_TEST_SCONTROL_OUTPUT
# env-var hook.
#
# Run: bash tests/test_submit_freemem.bash
# Exits 0 on all-pass, 1 on first failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUBMIT_TEST_SKIP_PREFLIGHT=1
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/submit.sh" </dev/null >/dev/null 2>&1 || true

pass=0
fail=0

# The 2026-04-24 scenario, reconstructed: aisurrey35 is MIXED, CfgTRES
# advertises mem=125G (128000MB RealMemory - 3000MB MemSpec), but FreeMem
# reports only 16456MB because something off-SLURM is eating RAM.
_scontrol_stub_starved() {
    cat <<'EOF'
NodeName=aisurrey35 State=MIXED RealMemory=128000 FreeMem=16456 Partitions=nice-project AllocTRES=cpu=14,mem=0M,gres/gpu=0,gres/gpu:nvidia_l40s=0 CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2 Gres=gpu:nvidia_l40s:2(IDX:0-1)
EOF
}

# A healthy node with 64GB free on a 128GB box.
_scontrol_stub_healthy() {
    cat <<'EOF'
NodeName=aisurrey35 State=IDLE RealMemory=128000 FreeMem=65536 Partitions=nice-project AllocTRES= CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2 Gres=gpu:nvidia_l40s:2(IDX:0-1)
EOF
}

# A drained node — its FreeMem must not rescue the partition from a
# HARD-FAIL, because SLURM can't schedule onto it.
_scontrol_stub_drained_only() {
    cat <<'EOF'
NodeName=aisurrey26 State=DRAIN+DOWN RealMemory=128000 FreeMem=99999 Partitions=nice-project AllocTRES= CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2 Gres=gpu:nvidia_l40s:2(IDX:0-1)
EOF
}

# One starved node in the target partition, one healthy node in a
# different partition — the helper must honour the partition filter and
# NOT borrow the other partition's headroom.
_scontrol_stub_wrong_partition_healthy() {
    cat <<'EOF'
NodeName=aisurrey35 State=MIXED RealMemory=128000 FreeMem=16456 Partitions=nice-project AllocTRES=cpu=14,mem=0M CfgTRES=cpu=14,mem=125G Gres=gpu:nvidia_l40s:2
NodeName=aisurrey10 State=IDLE RealMemory=64000 FreeMem=60000 Partitions=3090,3090_risk AllocTRES= CfgTRES=cpu=16,mem=62G Gres=gpu:nvidia_rtx_3090:1
EOF
}

t_hard_fail_when_freemem_below_request() {
    # The headline case from the 2026-04-24 incident.
    local out rc
    set +e
    out=$(SUBMIT_TEST_SCONTROL_OUTPUT="$(_scontrol_stub_starved)" \
        _check_freemem "nice-project" 120000 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 1 ]] && [[ "$out" == *"hard-fail"* ]] && [[ "$out" == *"16456"* ]]
}

t_fits_when_freemem_ample() {
    local out rc
    set +e
    out=$(SUBMIT_TEST_SCONTROL_OUTPUT="$(_scontrol_stub_healthy)" \
        _check_freemem "nice-project" 32000 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && [[ "$out" == *"ok"* ]]
}

t_ignores_drained_nodes() {
    # Drained/down nodes contribute no schedulable capacity — even if
    # their FreeMem is huge, treating them as fit would regress the
    # exact bug this helper exists to catch.
    local out rc
    set +e
    out=$(SUBMIT_TEST_SCONTROL_OUTPUT="$(_scontrol_stub_drained_only)" \
        _check_freemem "nice-project" 50000 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
}

t_honours_partition_filter() {
    # Healthy 3090 node must NOT rescue nice-project from its starved state.
    local out rc
    set +e
    out=$(SUBMIT_TEST_SCONTROL_OUTPUT="$(_scontrol_stub_wrong_partition_healthy)" \
        _check_freemem "nice-project" 120000 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 1 ]] && [[ "$out" == *"hard-fail"* ]]
}

t_unknown_when_scontrol_empty() {
    # Empty scontrol payload → unknown (rc 2); caller should fall through
    # to existing transient-contention handling rather than inventing a
    # verdict.
    local out rc
    set +e
    out=$(SUBMIT_TEST_SCONTROL_OUTPUT=" " \
        _check_freemem "nice-project" 50000 2>&1)
    rc=$?
    set -e
    [[ "$rc" -eq 2 ]]
}

for t in t_hard_fail_when_freemem_below_request t_fits_when_freemem_ample \
         t_ignores_drained_nodes t_honours_partition_filter \
         t_unknown_when_scontrol_empty; do
    if $t; then
        echo "  PASS  $t"
        ((pass++))
    else
        echo "  FAIL  $t"
        ((fail++))
    fi
done

echo
echo "passed: $pass  failed: $fail"
[[ "$fail" -eq 0 ]]
