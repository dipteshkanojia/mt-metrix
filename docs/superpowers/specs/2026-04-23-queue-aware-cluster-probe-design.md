# Queue-Aware Cluster Probe & Dynamic Partition Routing — Design

**Status:** Draft  |  **Date:** 2026-04-23  |  **Target:** mt-metrix main

## Summary

Teach `scripts/cluster_probe.py` to look at the live queue (`squeue`) in
addition to live node state (`scontrol show node`), and to pick a
partition that balances three things: VRAM fit, group ownership
(blocklist), and expected wait time. When the user's target partition is
sub-optimal or NO-FIT, `scripts/submit.sh` offers the top three
alternatives as an interactive 15-second prompt before calling `sbatch`.

## Motivation

The probe shipped in `772d67e` covers the static picture — does the
partition exist, does my job's VRAM fit on any of its GPU types — but
it's oblivious to two things that decide whether a submission *actually
runs* in useful time:

1. **Ownership.** `cogvis-project` has 80 GB A100s and looks attractive,
   but it belongs to another faculty and we shouldn't queue on it.
   `nice-project` is our group's node and should be strongly preferred.
2. **Contention.** An idle 48 GB card on `rtx_a6000_risk` beats a
   10-deep queue on `a100` for anything that fits on 48 GB. We already
   know which GPUs are free today; we should also know what's queued
   ahead of us and how long those jobs have left.

The current probe happily reports READY on a partition that has 32
pending jobs — the user only finds out after `sbatch` returns a JobID
that sits in PD for six hours. That wastes iteration cycles, especially
for ablations where the *next* job's wait is more costly than the job's
runtime.

A100 policy is distinct: A100s have ~2.2× the fairshare billing weight
of 3090s and are the only ≥80 GB hardware on the cluster. Using them
when the job fits on 48 GB is wasteful; reserving them for 72B workloads
and for the case where every 48 GB partition is contested is the right
default.

## Scope

**In scope:**

- `squeue` parsing with stdlib-only Python (probe runs before conda
  activate).
- Partition blocklist — starts at `{"cogvis-project"}`, extensible via
  a module-level constant.
- Partition tier — nice-project (1), 48 GB partitions (2), 24 GB (3),
  debug/2080ti (4), a100 (5). Tier is the tie-breaker after VRAM fit.
- Per-partition wait-time estimate (`wait_s`):
  - `0` if `gpus_free >= gpus_requested` (we can start now).
  - Else estimate how long until *enough* GPUs free up. With
    `deficit = gpus_requested - gpus_free + pending_gpu_demand_ahead`,
    take the `deficit`-th-smallest `time_left_s` across running jobs.
    `pending_gpu_demand_ahead` is the sum of `num_gpus` across PD jobs
    whose priority-order precedes ours — we proxy this with "all PD
    jobs on that partition" since `squeue` output is already sorted by
    priority and `--reason` won't tell us who's ahead.
  - If fewer running jobs than `deficit` have a known `time_left_s`,
    `wait_s = None` (rendered as `?`). Recommender treats `None` as
    "worse than any finite wait" for ranking.
- Top-3 alternatives surfaced in probe output (table + JSON).
- Probe writes a sidecar TSV (`--tee-alternatives=<path>`) so the
  submit wrapper can read structured alternatives back.
- `submit.sh` interactive prompt when either:
  - probe exits 2 (target NO-FIT), or
  - probe exits 0 but `recommended != target` (a better partition exists).
  15-second timeout, user must pick a number or cancel; no auto-default.
- `SUBMIT_AUTO_ROUTE=1` env var → skip prompt, pick alternative #1.
- `--stay-on-target` flag → skip prompt, keep user's partition even if
  sub-optimal (escape hatch for deliberate experiments).
- A100 de-prioritisation: when `vram_need <= 48`, a100 gets `+4` added
  to its tier score **only if** at least one non-a100 READY partition
  exists with `wait_s == 0`. If every non-a100 ≥`vram_need` partition
  either NO-FITs or has `wait_s > 0`, the penalty is dropped and a100
  competes on equal tier terms. When `vram_need > 48`, no penalty
  applies (a100 is the only option).

**Out of scope:**

- Changing the probe's VRAM-inference logic (landed in `772d67e`).
- Automatic partition rotation mid-run (if a job sits in PD too long,
  no automatic scancel+resubmit; user decides).
- Fairshare weight in the scoring (AISURREY exposes it via
  `sshare`, but we don't have reliable per-user quotas to model).
- GPU-hour budget tracking across sessions.

## Architecture

```
                  +-------------------+
                  |   scontrol show   |
                  |   node -o         |
                  +---------+---------+
                            |
                            v
+--------------+   +--------+---------+   +-------------------+
| squeue       |-->| aggregate_       |-->| pick_recommended  |
| --noheader   |   | partitions()     |   | (tier + wait +    |
+--------------+   |  + attach queue  |   |  blocklist + a100)|
                   |  stats per part. |   +---------+---------+
                   +------------------+             |
                                                    v
                                        +-----------+---------+
                                        |  top-3 alts + JSON  |
                                        |  + optional TSV     |
                                        +-----------+---------+
                                                    |
                                                    v
                                        +-----------+---------+
                                        |  submit.sh prompts  |
                                        |  15s or user picks  |
                                        +---------------------+
```

Three logical units, each independently testable:

1. **Queue reader** — runs `squeue`, parses lines to `Job` dataclasses,
   gracefully degrades to "no queue info" if the command fails or
   returns malformed output.
2. **Recommender** — pure function of (partitions, vram_need, gpus,
   blocklist, tier_map, a100_policy) → ranked list of partitions with
   scoring rationale.
3. **Interactive wrapper** — `submit.sh` prompt harness. Probe is
   non-interactive; all user interaction lives in the shell wrapper.

## Components

### `Job` dataclass

```python
@dataclass
class Job:
    job_id: str
    partition: str
    state: str           # R, PD, CG, CD, F, ...
    time_left_s: int | None   # None = unknown / unlimited
    time_used_s: int | None
    num_gpus: int        # parsed from tres-per-node gres=gpu:N
    reason: str          # e.g. "Resources", "Priority", "(null)"
    user: str
```

### Partition extensions

```python
@dataclass
class Partition:
    # ... existing fields ...
    running_jobs: list[Job]
    pending_jobs: list[Job]
    earliest_free_s: int | None     # min time_left across running jobs
    pending_gpu_demand: int         # sum of num_gpus across PD jobs
```

### Scoring tuple

`pick_recommended` returns partitions sorted by this tuple (ascending):

```
(
  ready_now_rank,       # 0 if gpus_free >= gpus_requested, else 1
  tier + a100_penalty,  # int; a100_penalty per A100 de-prioritisation rule
                        # (see Scope); 0 for every other partition
  wait_s_sortable,      # 0 if ready_now; else wait_s as int; None → INT_MAX
  vram_waste_gb,        # max_vram_on_partition - vram_need (non-negative)
  -gpus_free,           # more free = better (breaks remaining ties)
)
```

Unknown-VRAM partitions and blocklisted partitions are excluded
entirely, not just ranked low — they're not options.

### Output changes

Table columns (widened, see render_table):

```
partition     GPUs  free  VRAM  type          next free  pending  fit
nice-project  2/2   2     48G   NVIDIA L40S   now        0        READY
rtx_a6000_risk 4/8  4     48G   RTX A6000     now        2        READY
a100          0/4   0     80G   A100          2h14m      3        READY (a100-penalised)
cogvis-project 2/2  2     48G+  A6000         now        0        [not ours]
```

JSON output (`--json`) adds:

```json
{
  "recommended": "nice-project",
  "alternatives": [
    {"partition": "nice-project", "wait_s": 0, "tier": 1, "reason": "free now, group partition"},
    {"partition": "rtx_a6000_risk", "wait_s": 0, "tier": 2, "reason": "free now"},
    {"partition": "a100", "wait_s": 8040, "tier": 5, "reason": "only >48 GB partition available"}
  ]
}
```

Sidecar TSV (`--tee-alternatives=<path>`):

```
rank	partition	gpus_requested	wait_s	tier	reason
1	nice-project	1	0	1	free now
2	rtx_a6000_risk	1	0	2	free now
3	a100	1	8040	5	only fits, but contested
```

### `submit.sh` interactive prompt

Triggered when:

- Probe exit 2 (target NO-FIT), OR
- Probe exit 0 but `alternatives[0].partition != target_partition`
  (recommender has a different first choice).

Never triggered when:

- Probe exit 0 and target is already the recommender's first choice.
- `--stay-on-target` flag supplied.
- `SUBMIT_AUTO_ROUTE=1` in the environment (auto-picks alt #1).

Behaviour:

```
Your target: a100 (1× a100, wait 2h14m)
Better fit available:
  1) nice-project   (wait: now,   48 GB, our group)
  2) rtx_a6000_risk (wait: now,   48 GB, open queue)
  3) a100           (wait: 2h14m, 80 GB, you chose this)
Pick 1-3 or c to cancel (15s, no default):
```

No keypress within 15s → cancel. User presses `c` → cancel. `1`/`2`/`3`
→ re-exec sbatch with that partition's flags.

Escape hatches:

- `SUBMIT_AUTO_ROUTE=1 scripts/submit.sh ...` → no prompt, pick
  alternative #1 automatically (for unattended scripts).
- `scripts/submit.sh ... --stay-on-target` → no prompt, keep the
  user's original partition.

## Data flow

```
probe input: (config.yaml, --target-partition, --gpus, --tee-alternatives)
  1. scontrol show node -o  → Node list
  2. squeue --noheader ...   → Job list
  3. aggregate Node + Job per partition → Partition list
  4. infer_vram_need(config, gpus) → ConfigVRAMNeed
  5. check_fit(partition, need) for each → FitStatus list
  6. pick_recommended(fits, blocklist, tier, a100_policy)
       → (recommended, alternatives[])
  7. render table + JSON + optional TSV
  8. exit:
       0  target VRAM-fits (may still be sub-optimal vs. recommender)
       2  target NO-FIT (but at least one other partition fits)
       3  probe itself failed (scontrol missing or unparseable) —
          preserved from the original probe for submit.sh back-compat
       4  no partition can fit the job at the requested gpus count
       5  target partition is blocklisted

submit.sh preflight step [5/6]:
  - probe --tee-alternatives=$TMP/alts.tsv
  - read probe exit code and alts.tsv
  - if exit == 2, OR target not in top-3, prompt (unless SUBMIT_AUTO_ROUTE=1
    or --stay-on-target)
  - re-render sbatch command with chosen partition
  - continue to step [6/6] sbatch --test-only
```

## Error handling

- **`squeue` fails / returns garbage** → partition `running_jobs` and
  `pending_jobs` stay empty, `earliest_free_s=None`, wait-time estimate
  is "?", recommender falls back to tier-only scoring. Probe prints a
  warning but does not exit non-zero.
- **Malformed squeue line** → log once, skip line, continue. One bad
  line must not block the whole parse.
- **`D-HH:MM:SS` / `HH:MM:SS` / `MM:SS` / `UNLIMITED` / `N/A` /
  `INVALID`** all handled by `parse_time_duration` returning
  `int | None`.
- **Blocklisted partition is user's target** → exit 3, `submit.sh`
  refuses to submit and prints the reason (no prompt — user should pick
  a real partition, not a silent override).
- **All partitions blocklisted or NO-FIT** → probe exits 4,
  `submit.sh` prints the report and bails.

## Testing

All new logic is unit-tested with fixtures, no live cluster calls:

- `test_parse_time_duration_table` — every supported format + invalid
  inputs.
- `test_parse_squeue_line` — valid lines, gres variants, malformed
  tokens.
- `test_aggregate_partitions_attaches_queue_stats` — fixture with
  running + pending jobs in two partitions.
- `test_wait_time_estimate_*` — zero-contention, single-running, many-
  pending-long-wait cases.
- `test_blocklist_excludes_partition` — cogvis excluded entirely.
- `test_tier_ordering` — nice > rtx_a6000_risk > a100 when all fit.
- `test_a100_penalty_when_alternatives_exist` — a100 outranked by
  48 GB partition when vram_need <= 48 and both are READY.
- `test_a100_selected_when_vram_need_exceeds_48` — 72B with gpus=4 goes
  to a100, no penalty.
- `test_a100_selected_when_everything_else_contested` — all 48 GB
  partitions have wait > threshold, a100 wins.
- `test_tee_alternatives_writes_tsv` — TSV parsing round-trip.
- `test_pick_recommended_degrades_without_queue_info` — squeue
  unavailable → tier-only fallback, no crash.
- `test_submit_interactive_pick_alt_1` — piped stdin `1\n`, submit
  shells out to the chosen partition.
- `test_submit_interactive_cancel` — piped stdin `c\n`, submit exits
  without calling sbatch.
- `test_submit_interactive_timeout` — no stdin, 15s elapsed, submit
  exits without calling sbatch (smoke test uses a reduced timeout
  setting so CI stays fast).
- `test_submit_auto_route_env_var` — `SUBMIT_AUTO_ROUTE=1`, no prompt,
  picks alt #1.
- `test_submit_stay_on_target_flag` — `--stay-on-target`, no prompt,
  keeps user's partition even if sub-optimal.

Existing probe tests (93 of them) must all continue to pass — this
change strictly adds code paths.

## Rollout & compatibility

- **No config changes required.** Existing `submit.sh <cfg>` calls
  keep working; behaviour is identical when the user's chosen
  partition is already optimal.
- **New env var** `SUBMIT_AUTO_ROUTE=1` is opt-in.
- **New flag** `--stay-on-target` is opt-in.
- **`scripts/setup_cluster.sh` unchanged** — cogvis-project stays
  listed in partition docs for visibility; the blocklist lives in the
  probe, not in the cluster-topology doc.
- **Rollback path:** revert the single commit; pre-this-commit probe
  ignores `squeue` and produces the same output as before.

## Follow-up tickets (out of scope for this PR)

- `--prefer-partition=<name>` flag as a soft preference (distinct from
  `--target-partition` which is mandatory today).
- Per-user history of wait times → smarter `wait_s` estimate than the
  naive `pending_demand × earliest_free_s`.
- `mt-metrix submit` Python wrapper mirrors the interactive prompt.
- Make the blocklist configurable via a YAML file under `configs/` so
  it's not a Python constant edit.
