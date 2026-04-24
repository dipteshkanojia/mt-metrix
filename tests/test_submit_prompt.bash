#!/usr/bin/env bash
# Smoke tests for submit.sh's interactive alternative-picking prompt.
#
# Sources submit.sh with SUBMIT_TEST_SKIP_PREFLIGHT=1 so the file only
# DEFINES the helper functions (including _prompt_alternative) and then
# returns. Each test then invokes the helper directly with a canned
# alternatives TSV and piped stdin.
#
# Run: bash tests/test_submit_prompt.bash
# Exits 0 on all-pass, 1 on first failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUBMIT_TEST_SKIP_PREFLIGHT=1
# Short timeout so CI doesn't block on the 15s prompt.
export SUBMIT_PROMPT_TIMEOUT=2
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/submit.sh" </dev/null >/dev/null 2>&1 || true
# Note: `source` under set -euo pipefail still runs the file. The block
# below only executes the helper definitions because submit.sh returns
# early when SUBMIT_TEST_SKIP_PREFLIGHT=1 — see the top of submit.sh.

pass=0
fail=0

_mkalts() {
    local tmp
    tmp=$(mktemp)
    cat >"$tmp" <<'EOF'
rank	partition	gpus_requested	wait_s	tier	reason
1	nice-project	1	0	1	free now, group partition
2	rtx_a6000_risk	1	0	2	free now
3	a100	1	8040	5	only >48 GB partition
EOF
    echo "$tmp"
}

_mkalts_empty() {
    local tmp
    tmp=$(mktemp)
    cat >"$tmp" <<'EOF'
rank	partition	gpus_requested	wait_s	tier	reason
EOF
    echo "$tmp"
}

_mkalts_malformed_wait() {
    local tmp
    tmp=$(mktemp)
    cat >"$tmp" <<'EOF'
rank	partition	gpus_requested	wait_s	tier	reason
1	nice-project	1	abc	1	malformed wait_s
EOF
    echo "$tmp"
}

_mkalts_one_row() {
    local tmp
    tmp=$(mktemp)
    cat >"$tmp" <<'EOF'
rank	partition	gpus_requested	wait_s	tier	reason
1	nice-project	1	0	1	free now, group partition
EOF
    echo "$tmp"
}

t_pick_alt_1() {
    local alts; alts=$(_mkalts)
    local chosen
    chosen=$(echo "1" | _prompt_alternative "a100" "$alts" 2>/dev/null || true)
    [[ "$chosen" == "nice-project" ]]
}

t_pick_alt_2() {
    local alts; alts=$(_mkalts)
    local chosen
    chosen=$(echo "2" | _prompt_alternative "a100" "$alts" 2>/dev/null || true)
    [[ "$chosen" == "rtx_a6000_risk" ]]
}

t_cancel_c() {
    local alts; alts=$(_mkalts)
    set +e
    echo "c" | _prompt_alternative "a100" "$alts" >/dev/null 2>&1
    rc=$?
    set -e
    # Cancel returns exit 7 (chosen sentinel); submit.sh's main flow
    # uses that to exit without calling sbatch.
    [[ "$rc" -eq 7 ]]
}

t_timeout_cancel() {
    # No stdin — prompt times out and cancels.
    local alts; alts=$(_mkalts)
    set +e
    _prompt_alternative "a100" "$alts" </dev/null >/dev/null 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 7 ]]
}

t_auto_route() {
    local alts; alts=$(_mkalts)
    local chosen
    chosen=$(SUBMIT_AUTO_ROUTE=1 _prompt_alternative "a100" "$alts" 2>/dev/null || true)
    [[ "$chosen" == "nice-project" ]]
}

t_invalid_digit_out_of_range() {
    # "9" is a valid single digit but out-of-range for a 3-row TSV,
    # so the numeric branch rejects it and the helper returns 7.
    local alts; alts=$(_mkalts)
    set +e
    echo "9" | _prompt_alternative "a100" "$alts" >/dev/null 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 7 ]]
}

t_invalid_input() {
    # Non-digit, non-"c" input hits the catch-all branch and returns 7.
    local alts; alts=$(_mkalts)
    set +e
    echo "xyz" | _prompt_alternative "a100" "$alts" >/dev/null 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 7 ]]
}

t_empty_tsv() {
    # Header-only TSV -> parts[] stays empty -> ${#parts[@]} -eq 0 -> rc 7.
    local alts; alts=$(_mkalts_empty)
    set +e
    echo "1" | _prompt_alternative "a100" "$alts" >/dev/null 2>&1
    rc=$?
    set -e
    [[ "$rc" -eq 7 ]]
}

t_malformed_wait_s() {
    # wait_s="abc" must not crash the arithmetic under set -euo pipefail;
    # the guard falls through to "wait: ?" and picking index 0 succeeds.
    local alts; alts=$(_mkalts_malformed_wait)
    local chosen
    chosen=$(echo "1" | _prompt_alternative "a100" "$alts" 2>/dev/null || true)
    [[ "$chosen" == "nice-project" ]]
}

t_alt_count_three_rows() {
    # Standard 3-row TSV. String-equality (not -eq) so an unimplemented
    # helper that returns empty stdout doesn't pass by accident.
    local alts; alts=$(_mkalts)
    local n
    n=$(_alt_count "$alts" 2>/dev/null)
    [[ "$n" == "3" ]]
}

t_alt_count_one_row() {
    local alts; alts=$(_mkalts_one_row)
    local n
    n=$(_alt_count "$alts" 2>/dev/null)
    [[ "$n" == "1" ]]
}

t_alt_count_empty_tsv() {
    # Header-only TSV — zero data rows. The bug: without _alt_count,
    # submit.sh's case 2) hit _prompt_alternative with parts=(), got
    # rc=7, and treated it as "user cancelled" even though the user
    # never saw a prompt. _alt_count lets the caller distinguish
    # "nothing to prompt about" from "user declined".
    local alts; alts=$(_mkalts_empty)
    local n
    n=$(_alt_count "$alts" 2>/dev/null)
    [[ "$n" == "0" ]]
}

t_alt_count_missing_file() {
    # Missing file must return "0" cleanly, not crash under set -euo pipefail.
    local n
    n=$(_alt_count "/nonexistent/path/that/does/not/exist.tsv" 2>/dev/null)
    [[ "$n" == "0" ]]
}

t_alt_count_no_arg() {
    # Called with no argument — treat as missing file.
    local n
    n=$(_alt_count 2>/dev/null)
    [[ "$n" == "0" ]]
}

t_timeout_honours_explicit_p() {
    # When submit.sh detects the user passed an explicit -p <target>, it
    # now hands the target as a 3rd arg to _prompt_alternative. If the
    # user walks away (timeout), the helper honours that explicit choice
    # instead of cancelling — otherwise a drive-by recommender proposal
    # eats the 15s window and trashes a run the user had already decided.
    local alts; alts=$(_mkalts)
    local chosen rc
    set +e
    chosen=$(_prompt_alternative "l40s_risk" "$alts" "l40s_risk" </dev/null 2>/dev/null)
    rc=$?
    set -e
    [[ "$chosen" == "l40s_risk" ]] && [[ "$rc" -eq 0 ]]
}

for t in t_pick_alt_1 t_pick_alt_2 t_cancel_c t_timeout_cancel t_auto_route \
         t_invalid_digit_out_of_range t_invalid_input t_empty_tsv t_malformed_wait_s \
         t_alt_count_three_rows t_alt_count_one_row t_alt_count_empty_tsv \
         t_alt_count_missing_file t_alt_count_no_arg \
         t_timeout_honours_explicit_p; do
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
