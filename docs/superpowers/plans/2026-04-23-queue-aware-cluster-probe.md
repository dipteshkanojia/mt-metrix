# Queue-Aware Cluster Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach `scripts/cluster_probe.py` to read `squeue`, score partitions by (VRAM fit + ownership + wait time), and return ranked alternatives; teach `scripts/submit.sh` to prompt the user (15 s, no default) when the recommender disagrees with their target partition.

**Architecture:** Three new logical units in the probe — queue reader (`squeue` parse + `Job` dataclass), recommender (pure function: partitions + policy → ranked `Alternative` list), and a TSV sidecar (`--tee-alternatives`) that submit.sh reads back. `submit.sh` gains a 15 s interactive prompt gated by `SUBMIT_AUTO_ROUTE=1` and `--stay-on-target` escape hatches. The existing scontrol pipeline, VRAM inference, and fit-check logic stay untouched.

**Tech Stack:** Python 3.10+ stdlib only (probe must run before `conda activate`); bash 4+ for `submit.sh`; pytest for tests. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md`

## File structure

| File | Role | Change |
|------|------|--------|
| `scripts/cluster_probe.py` | Stdlib Python CLI, runs before `conda activate`. One module today; staying one module — new pieces attach cleanly to the existing dataclasses. | Modify |
| `scripts/submit.sh` | Bash pre-flight wrapper. Section `[5/6] cluster probe` is the only block that changes; the surrounding six-check structure stays. | Modify |
| `tests/test_cluster_probe.py` | One big `pytest` module loading `cluster_probe.py` via importlib. New fixtures + tests appended; existing 93 tests must keep passing. | Modify |
| `docs/superpowers/plans/2026-04-23-queue-aware-cluster-probe.md` | This plan. | Create |
| `examples/04_aisurrey_submission.md` | User-facing submission walkthrough. One new subsection documenting the prompt + env var + flag. | Modify |
| `docs/AISURREY.md` | Cluster runbook. One new subsection mirroring the example. | Modify |

The probe's shape stays flat (constants → dataclasses → parsers → aggregator → checker → recommender → renderer → main) so readers can still follow top-to-bottom.

---

## Task 1: Constants + `parse_time_duration`

**Files:**
- Modify: `scripts/cluster_probe.py` (top-of-file constants section; new free function before the scontrol parsing block)
- Modify: `tests/test_cluster_probe.py` (append to "GPU type → VRAM mapping" section)

Pure utility layer: no cluster calls, no dataclasses yet. Lands blocklist + tier policy as module-level data so everything downstream can import them, and the SLURM duration parser used by the queue reader.

- [ ] **Step 1: Write failing tests for the new constants + helper**

Append to `tests/test_cluster_probe.py` (after `test_vram_for_gpu_type_ada_does_not_shadow_quadro`):

```python
# ---------------------------------------------------------------------------
# Blocklist, tier, duration parsing (queue-aware additions)
# ---------------------------------------------------------------------------

def test_partitions_blocklist_contains_cogvis_project(probe):
    """cogvis-project belongs to another faculty; we mustn't queue on it
    even though it has 48 GB A6000s that would otherwise be attractive."""
    assert "cogvis-project" in probe.PARTITIONS_BLOCKLIST


def test_is_blocklisted(probe):
    assert probe.is_blocklisted("cogvis-project") is True
    assert probe.is_blocklisted("nice-project") is False
    assert probe.is_blocklisted("a100") is False


@pytest.mark.parametrize(
    "partition,expected_tier",
    [
        ("nice-project", 1),       # our group's partition
        ("rtx_a6000_risk", 2),     # 48 GB general
        ("l40s_risk", 2),
        ("rtx8000", 2),
        ("3090", 3),               # 24 GB
        ("3090_risk", 3),
        ("debug", 4),              # short wall-time / 2080ti class
        ("2080ti", 4),
        ("a100", 5),               # reserved for >48 GB or when nothing else fits
        ("unlisted-partition", 3), # unknown → default 3 so we don't crown it
    ],
)
def test_partition_tier(probe, partition, expected_tier):
    assert probe.partition_tier(partition) == expected_tier


@pytest.mark.parametrize(
    "raw,expected_s",
    [
        ("0:30", 30),               # MM:SS (30 s)
        ("5:00", 300),              # MM:SS (5 m)
        ("1:30:00", 5400),          # HH:MM:SS (1 h 30 m)
        ("12:00:00", 43200),        # HH:MM:SS (12 h)
        ("2-06:00:00", 194400),     # D-HH:MM:SS (2 d 6 h)
        ("0-00:15:00", 900),        # explicit zero days
        ("N/A", None),
        ("UNLIMITED", None),
        ("INVALID", None),
        ("", None),
        ("not-a-time", None),
    ],
)
def test_parse_time_duration(probe, raw, expected_s):
    assert probe.parse_time_duration(raw) == expected_s
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_cluster_probe.py -k "blocklist or partition_tier or parse_time_duration" -v`
Expected: `AttributeError: module 'cluster_probe' has no attribute 'PARTITIONS_BLOCKLIST'` (or similar) on every case.

- [ ] **Step 3: Add the constants + helper to `cluster_probe.py`**

In `scripts/cluster_probe.py`, immediately after the `GPU_VRAM_GB` list and its `vram_for_gpu_type` helper (so all module-level policy lives together), add:

```python
# ---------------------------------------------------------------------------
# Queue-aware policy. The probe reads ``squeue`` (below) to enrich
# per-partition fit checks with wait-time estimates, and chooses a
# recommended partition by combining VRAM fit with two policies:
#
#   - PARTITIONS_BLOCKLIST: partitions that belong to other faculties /
#     groups, and that we mustn't queue on even when they would otherwise
#     fit. Starts as {"cogvis-project"}; extend as we learn more.
#   - PARTITION_TIER: our preference ordering among partitions that DO
#     accept our jobs. Lower = preferred. Ties are broken by wait time,
#     then VRAM waste, then free-GPU count (see pick_recommended).
#
# Unknown partitions fall through to tier 3 so a stranger doesn't
# accidentally outrank nice-project — 3 matches the 24 GB general tier,
# a conservative middle ground.
# ---------------------------------------------------------------------------

PARTITIONS_BLOCKLIST: set[str] = {
    "cogvis-project",
}

PARTITION_TIER: dict[str, int] = {
    # Tier 1 — our group's partition
    "nice-project": 1,
    # Tier 2 — 48 GB open partitions
    "rtx_a6000_risk": 2,
    "l40s_risk":      2,
    "rtx8000":        2,
    # Tier 3 — 24 GB
    "3090":           3,
    "3090_risk":      3,
    # Tier 4 — short / 11 GB
    "debug":          4,
    "2080ti":         4,
    # Tier 5 — A100. De-prioritised for jobs that fit on 48 GB; see the
    # a100-penalty logic in pick_recommended.
    "a100":           5,
}

DEFAULT_TIER = 3


def is_blocklisted(partition: str) -> bool:
    return partition in PARTITIONS_BLOCKLIST


def partition_tier(partition: str) -> int:
    return PARTITION_TIER.get(partition, DEFAULT_TIER)
```

Then, just before the `# scontrol / sinfo parsing.` banner, add the duration parser (it'll be used by the squeue reader in the next task):

```python
# ---------------------------------------------------------------------------
# SLURM duration parsing. ``squeue`` emits job times in ``D-HH:MM:SS``,
# ``HH:MM:SS``, or ``MM:SS`` depending on how much time has elapsed; it
# also emits ``N/A``, ``UNLIMITED``, and (rarely) ``INVALID`` for jobs
# that haven't started, have no wall-time cap, or have state-machine
# glitches respectively. None of these map to an int, so we return
# ``None`` and let the recommender treat ``None`` as "worse than any
# finite wait" when ranking.
# ---------------------------------------------------------------------------

_DURATION_FULL_RE = re.compile(
    r"^(?:(?P<days>\d+)-)?(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})$"
)
_DURATION_MS_RE = re.compile(r"^(?P<m>\d+):(?P<s>\d{2})$")


def parse_time_duration(raw: str) -> Optional[int]:
    """Parse a SLURM duration string into seconds; ``None`` if unknowable.

    Accepts ``MM:SS``, ``HH:MM:SS``, ``D-HH:MM:SS``. Returns ``None`` for
    ``N/A``, ``UNLIMITED``, empty string, and any other unrecognised
    input (``INVALID``, garbage, etc.) — callers treat ``None`` as
    "worse than any finite value" when ranking.
    """
    if not raw:
        return None
    trimmed = raw.strip()
    if trimmed in {"N/A", "UNLIMITED", "INVALID"}:
        return None
    full = _DURATION_FULL_RE.match(trimmed)
    if full:
        days = int(full.group("days") or 0)
        h = int(full.group("h"))
        m = int(full.group("m"))
        s = int(full.group("s"))
        return days * 86400 + h * 3600 + m * 60 + s
    ms = _DURATION_MS_RE.match(trimmed)
    if ms:
        m = int(ms.group("m"))
        s = int(ms.group("s"))
        return m * 60 + s
    return None
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "blocklist or partition_tier or parse_time_duration" -v`
Expected: all new tests PASS. The pre-existing 93 tests are not touched.

- [ ] **Step 5: Run the full probe test file to check nothing regressed**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: add partition blocklist, tier, duration parser

Three stdlib utilities that downstream queue-aware routing depends on:

- PARTITIONS_BLOCKLIST starts at {"cogvis-project"} so the recommender
  never suggests a partition that belongs to another faculty.
- PARTITION_TIER pins our group's preference order
  (nice-project > 48 GB > 24 GB > debug/2080ti > a100); unknown
  partitions default to tier 3.
- parse_time_duration handles SLURM's MM:SS / HH:MM:SS / D-HH:MM:SS
  formats plus N/A / UNLIMITED / INVALID. Returns None when unknowable;
  callers treat None as "worse than any finite value" when ranking.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `Job` dataclass + `parse_squeue_line` + `run_squeue`

**Files:**
- Modify: `scripts/cluster_probe.py` (append to the scontrol section; `Job` lives next to `Node`)
- Modify: `tests/test_cluster_probe.py` (new fixture + parsing tests)

Reads the live queue with a single `squeue` invocation and converts each line to a typed `Job`. Defensive: malformed lines are skipped with a warning, not crashed on.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cluster_probe.py` after the duration parser tests:

```python
# ---------------------------------------------------------------------------
# squeue reader
# ---------------------------------------------------------------------------

# squeue --noheader --format="%i|%P|%T|%L|%M|%r|%b|%D|%u"
# Fields: JobID | Partition | State | TimeLeft | TimeUsed | Reason |
#         tres-per-node | NumNodes | User
SQUEUE_FIXTURE = "\n".join([
    # Running, 1× A100, 2h14m left, our user
    "1234001|a100|RUNNING|2:14:00|3:46:00|None|gres:gpu:1|1|dk0023",
    # Pending, 4× A100, waiting for resources
    "1234002|a100|PENDING|N/A|N/A|Resources|gres:gpu:4|1|someone",
    # Running, 1× L40s, 10h left
    "1234010|nice-project|RUNNING|10:00:00|1:00:00|None|gres:gpu:1|1|dk0023",
    # Running, 1× A6000, 2-06:00:00 left (multi-day)
    "1234020|rtx_a6000_risk|RUNNING|2-06:00:00|6:00:00|None|gres:gpu:1|1|someone",
    # Pending with UNLIMITED time — legit on some admin jobs
    "1234030|rtx_a6000_risk|PENDING|UNLIMITED|N/A|Priority|gres:gpu:1|1|someone",
    # Malformed: missing fields → must skip without crash
    "GARBAGE LINE",
    # Valid but job has no gres (CPU job) — num_gpus=0
    "1234099|debug|RUNNING|0:10:00|0:05:00|None|cpu=4|1|dk0023",
])


def test_parse_squeue_line_running_a100(probe):
    line = SQUEUE_FIXTURE.splitlines()[0]
    j = probe.parse_squeue_line(line)
    assert j is not None
    assert j.job_id == "1234001"
    assert j.partition == "a100"
    assert j.state == "RUNNING"
    assert j.time_left_s == 2 * 3600 + 14 * 60
    assert j.time_used_s == 3 * 3600 + 46 * 60
    assert j.num_gpus == 1
    assert j.reason == "None"
    assert j.user == "dk0023"


def test_parse_squeue_line_pending_4gpu(probe):
    line = SQUEUE_FIXTURE.splitlines()[1]
    j = probe.parse_squeue_line(line)
    assert j is not None
    assert j.state == "PENDING"
    assert j.time_left_s is None
    assert j.time_used_s is None
    assert j.num_gpus == 4
    assert j.reason == "Resources"


def test_parse_squeue_line_multi_day(probe):
    line = SQUEUE_FIXTURE.splitlines()[3]
    j = probe.parse_squeue_line(line)
    assert j.time_left_s == 2 * 86400 + 6 * 3600


def test_parse_squeue_line_unlimited(probe):
    line = SQUEUE_FIXTURE.splitlines()[4]
    j = probe.parse_squeue_line(line)
    assert j.time_left_s is None      # UNLIMITED → None


def test_parse_squeue_line_garbage_returns_none(probe):
    assert probe.parse_squeue_line("GARBAGE LINE") is None
    assert probe.parse_squeue_line("") is None
    # Too few fields
    assert probe.parse_squeue_line("123|a100") is None


def test_parse_squeue_line_cpu_job_zero_gpus(probe):
    line = SQUEUE_FIXTURE.splitlines()[6]
    j = probe.parse_squeue_line(line)
    assert j is not None
    assert j.num_gpus == 0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_cluster_probe.py -k "squeue_line" -v`
Expected: `AttributeError: module 'cluster_probe' has no attribute 'parse_squeue_line'`.

- [ ] **Step 3: Add the `Job` dataclass, parser, and `run_squeue` to `cluster_probe.py`**

Immediately after the `@dataclass class Node` block (so `Job` is visually adjacent to `Node`), add:

```python
# ---------------------------------------------------------------------------
# squeue reader. Emits ``Job`` dataclasses for every running or pending
# job. The recommender uses these to estimate wait time per partition
# rather than blindly trusting "free GPUs now" — a partition with 0 free
# and 30 short-running jobs is different from 0 free + 1 long-running.
#
# Format string is kept in sync with parse_squeue_line's field order.
# squeue emits empty strings when a field is N/A; the parser turns those
# into None so callers don't have to special-case each one.
# ---------------------------------------------------------------------------

SQUEUE_FORMAT = "%i|%P|%T|%L|%M|%r|%b|%D|%u"

_SQUEUE_GRES_GPU_RE = re.compile(r"gpu(?::[a-zA-Z0-9_]+)?:(\d+)")


@dataclass
class Job:
    """One row from ``squeue --noheader --format=...``.

    ``time_left_s`` and ``time_used_s`` are ``None`` for pending jobs
    (squeue prints ``N/A``) and for jobs with ``UNLIMITED`` wall clock;
    the recommender treats ``None`` as "worse than any finite value" so
    unknown-duration running jobs don't shrink wait estimates.
    """

    job_id: str
    partition: str
    state: str                # RUNNING, PENDING, COMPLETING, CONFIGURING, ...
    time_left_s: Optional[int]
    time_used_s: Optional[int]
    num_gpus: int
    reason: str
    user: str


def parse_squeue_line(line: str) -> Optional[Job]:
    """Parse a single ``squeue`` line into a :class:`Job`.

    Returns ``None`` if the line has fewer than the 9 expected fields or
    cannot be unambiguously parsed. Callers can safely iterate over all
    lines in the output and drop ``None`` entries.
    """
    if not line or "|" not in line:
        return None
    parts = line.split("|")
    if len(parts) < 9:
        return None
    job_id, partition, state, time_left, time_used, reason, tres, _nodes, user = parts[:9]
    if not job_id or not partition:
        return None
    num_gpus = 0
    if tres:
        m = _SQUEUE_GRES_GPU_RE.search(tres)
        if m:
            num_gpus = int(m.group(1))
    return Job(
        job_id=job_id.strip(),
        partition=partition.strip(),
        state=state.strip().upper(),
        time_left_s=parse_time_duration(time_left),
        time_used_s=parse_time_duration(time_used),
        num_gpus=num_gpus,
        reason=reason.strip(),
        user=user.strip(),
    )


def run_squeue() -> Optional[str]:
    """Run ``squeue --noheader --format=...`` and return its stdout.

    ``None`` if squeue isn't on PATH or returns non-zero — the
    recommender falls back to tier-only scoring without wait estimates.
    Mirrors ``run_scontrol_show_node`` for consistency.
    """
    if shutil.which("squeue") is None:
        return None
    try:
        res = subprocess.run(
            ["squeue", "--noheader", f"--format={SQUEUE_FORMAT}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "squeue_line" -v`
Expected: all tests PASS.

- [ ] **Step 5: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all 99+ tests PASS (93 pre-existing + 4 constants + 6 squeue).

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: Job dataclass + squeue reader

Adds stdlib subprocess wrapper for `squeue --noheader --format=...` and
a robust line parser producing Job(job_id, partition, state,
time_left_s, time_used_s, num_gpus, reason, user). Malformed or
too-short lines return None rather than crashing the parse.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Partition queue-stats fields + `attach_queue_stats`

**Files:**
- Modify: `scripts/cluster_probe.py` (`Partition` dataclass + new pure function)
- Modify: `tests/test_cluster_probe.py` (new tests)

Adds `running_jobs`, `pending_jobs`, `earliest_free_s`, `pending_gpu_demand` to `Partition`. Keeps them default-empty so the existing probe code and existing tests keep working. A new `attach_queue_stats(partitions, jobs)` function mutates in place.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cluster_probe.py`:

```python
# ---------------------------------------------------------------------------
# attach_queue_stats
# ---------------------------------------------------------------------------

def test_attach_queue_stats_mixed_partition_traffic(probe):
    _, parts_map = _parse_all(probe)
    jobs = [
        j for line in SQUEUE_FIXTURE.splitlines()
        if (j := probe.parse_squeue_line(line)) is not None
    ]
    probe.attach_queue_stats(parts_map, jobs)

    # a100: 1 running (2h14m), 1 pending (4× a100)
    a = parts_map["a100"]
    assert len(a.running_jobs) == 1
    assert len(a.pending_jobs) == 1
    assert a.earliest_free_s == 2 * 3600 + 14 * 60
    assert a.pending_gpu_demand == 4

    # nice-project: 1 running (10h), 0 pending
    n = parts_map["nice-project"]
    assert len(n.running_jobs) == 1
    assert n.pending_jobs == []
    assert n.earliest_free_s == 10 * 3600
    assert n.pending_gpu_demand == 0

    # rtx_a6000_risk: 1 running (2d6h), 1 pending (UNLIMITED)
    r = parts_map["rtx_a6000_risk"]
    assert len(r.running_jobs) == 1
    assert len(r.pending_jobs) == 1
    assert r.earliest_free_s == 2 * 86400 + 6 * 3600
    assert r.pending_gpu_demand == 1

    # debug: 1 running CPU-only job (num_gpus=0)
    d = parts_map["debug"]
    # debug is not in our fixture's scontrol output (no Partitions=debug
    # node), so attach_queue_stats must silently skip jobs on unknown
    # partitions rather than crash.
    assert "debug" not in parts_map


def test_attach_queue_stats_handles_unknown_partition(probe):
    """Jobs whose .partition isn't in the scontrol-derived partitions map
    must be dropped silently — they don't affect any aggregates."""
    _, parts_map = _parse_all(probe)
    ghost = probe.Job(
        job_id="999", partition="nonexistent",
        state="PENDING", time_left_s=None, time_used_s=None,
        num_gpus=1, reason="Resources", user="someone",
    )
    probe.attach_queue_stats(parts_map, [ghost])
    # parts_map untouched: nonexistent partition isn't injected.
    assert "nonexistent" not in parts_map


def test_attach_queue_stats_ignores_jobs_with_unknown_time_left(probe):
    """Running jobs whose TimeLeft was UNLIMITED / N/A shouldn't decide
    earliest_free_s — we only know the min across KNOWN finite values.
    """
    _, parts_map = _parse_all(probe)
    nice_jobs = [
        probe.Job(
            job_id="J1", partition="nice-project", state="RUNNING",
            time_left_s=3600, time_used_s=0, num_gpus=1,
            reason="None", user="u",
        ),
        probe.Job(
            job_id="J2", partition="nice-project", state="RUNNING",
            time_left_s=None, time_used_s=0, num_gpus=1,
            reason="None", user="u",
        ),
    ]
    probe.attach_queue_stats(parts_map, nice_jobs)
    assert parts_map["nice-project"].earliest_free_s == 3600


def test_aggregate_partitions_defaults_queue_fields(probe):
    """Existing aggregator call (no attach_queue_stats yet) leaves the
    new fields empty/defaulted — no crash, no stale data."""
    _, parts_map = _parse_all(probe)
    p = parts_map["nice-project"]
    assert p.running_jobs == []
    assert p.pending_jobs == []
    assert p.earliest_free_s is None
    assert p.pending_gpu_demand == 0
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_cluster_probe.py -k "attach_queue_stats or queue_fields" -v`
Expected: `AttributeError` on each (e.g. `'Partition' object has no attribute 'running_jobs'`).

- [ ] **Step 3: Extend `Partition` and add `attach_queue_stats`**

In `scripts/cluster_probe.py`, edit the `Partition` dataclass: add four fields *after* `node_names`:

```python
@dataclass
class Partition:
    name: str
    nodes_total: int = 0
    nodes_usable: int = 0
    gpus_total: int = 0
    gpus_alloc: int = 0
    gpu_types: list[str] = field(default_factory=list)
    max_vram_gb: int = 0
    max_vram_gpu_type: str = ""
    unknown_gpu_types: list[str] = field(default_factory=list)
    max_mem_mb: int = 0
    max_cpus: int = 0
    max_gpu_per_node: int = 0
    node_names: list[str] = field(default_factory=list)
    # ---- queue-aware additions (Task 3) ----
    running_jobs: list["Job"] = field(default_factory=list)
    pending_jobs: list["Job"] = field(default_factory=list)
    earliest_free_s: Optional[int] = None
    pending_gpu_demand: int = 0
```

(The forward reference `"Job"` keeps the field typed even though `Job` is defined further down the module.)

Immediately after `aggregate_partitions`, add:

```python
def attach_queue_stats(
    partitions: dict[str, "Partition"], jobs: list["Job"]
) -> None:
    """Populate per-partition running/pending/earliest_free/pending_demand.

    Jobs on partitions we didn't learn about from scontrol are skipped —
    they can't influence any partition we might recommend. Running jobs
    whose ``time_left_s`` is None don't push ``earliest_free_s`` either
    (unknown finish time is worse than any known one for ranking).
    """
    for job in jobs:
        part = partitions.get(job.partition)
        if part is None:
            continue
        if job.state == "RUNNING":
            part.running_jobs.append(job)
            if job.time_left_s is not None and (
                part.earliest_free_s is None
                or job.time_left_s < part.earliest_free_s
            ):
                part.earliest_free_s = job.time_left_s
        elif job.state == "PENDING":
            part.pending_jobs.append(job)
            part.pending_gpu_demand += job.num_gpus
        # Ignore COMPLETING/CONFIGURING/CANCELLED etc. — not load-bearing.
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "attach_queue_stats or queue_fields" -v`
Expected: all tests PASS.

- [ ] **Step 5: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all tests PASS (~105 now).

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: attach squeue stats to partitions

Extends Partition with running_jobs / pending_jobs / earliest_free_s /
pending_gpu_demand (all default-empty) and a pure
attach_queue_stats(partitions, jobs) that populates them in place.
Jobs on unknown partitions are dropped silently; running jobs with
unknown TimeLeft don't push earliest_free_s.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `estimate_wait_s`

**Files:**
- Modify: `scripts/cluster_probe.py` (new free function after `attach_queue_stats`)
- Modify: `tests/test_cluster_probe.py` (table-driven test)

Concrete wait-time formula per the spec:

```
deficit = gpus_requested - gpus_free + pending_gpu_demand
if deficit <= 0:     return 0
take the deficit-th smallest time_left_s among running jobs with known durations
if fewer than deficit known durations exist:  return None
```

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# estimate_wait_s
# ---------------------------------------------------------------------------

def _make_partition(probe, **kw):
    """Construct a Partition for wait-time unit tests without scontrol."""
    p = probe.Partition(name=kw.pop("name", "test"))
    p.gpus_total = kw.pop("gpus_total", 0)
    p.gpus_alloc = kw.pop("gpus_alloc", 0)
    p.pending_gpu_demand = kw.pop("pending_gpu_demand", 0)
    p.running_jobs = kw.pop("running_jobs", [])
    for leftover in kw:  # pragma: no cover — typo guard
        raise TypeError(f"unexpected kwarg: {leftover}")
    return p


def _job_with_time_left(probe, t_left: Optional[int], gpus: int = 1):
    return probe.Job(
        job_id="x", partition="test", state="RUNNING",
        time_left_s=t_left, time_used_s=0,
        num_gpus=gpus, reason="None", user="u",
    )


def test_estimate_wait_s_ready_now_returns_zero(probe):
    p = _make_partition(probe, gpus_total=4, gpus_alloc=1)   # 3 free
    assert probe.estimate_wait_s(p, gpus_requested=2) == 0


def test_estimate_wait_s_single_running_blocks(probe):
    """All GPUs allocated, 1 running job with 1h left, request 1 GPU."""
    p = _make_partition(
        probe,
        gpus_total=1, gpus_alloc=1,
        running_jobs=[_job_with_time_left(probe, 3600)],
    )
    assert probe.estimate_wait_s(p, gpus_requested=1) == 3600


def test_estimate_wait_s_takes_nth_smallest(probe):
    """Need 2 GPUs, 3 running jobs (30m, 1h, 2h left) holding all 3 GPUs.
    Deficit=2 means we need 2 of them to finish; the 2nd-shortest is 1h.
    """
    p = _make_partition(
        probe,
        gpus_total=3, gpus_alloc=3,
        running_jobs=[
            _job_with_time_left(probe, 1800),
            _job_with_time_left(probe, 3600),
            _job_with_time_left(probe, 7200),
        ],
    )
    assert probe.estimate_wait_s(p, gpus_requested=2) == 3600


def test_estimate_wait_s_pending_demand_extends_wait(probe):
    """Pending GPU demand counts as 'ahead of us' even if priority would
    actually sort differently — conservative proxy."""
    p = _make_partition(
        probe,
        gpus_total=1, gpus_alloc=1,
        pending_gpu_demand=1,
        running_jobs=[
            _job_with_time_left(probe, 1800),
            _job_with_time_left(probe, 3600),
        ],
    )
    # Deficit = 1 - 0 + 1 = 2 → 2nd-shortest running (3600).
    assert probe.estimate_wait_s(p, gpus_requested=1) == 3600


def test_estimate_wait_s_unknown_when_too_few_known(probe):
    """3 running jobs but only 1 has a known TimeLeft → deficit=2 → None."""
    p = _make_partition(
        probe,
        gpus_total=3, gpus_alloc=3,
        running_jobs=[
            _job_with_time_left(probe, 1800),
            _job_with_time_left(probe, None),
            _job_with_time_left(probe, None),
        ],
    )
    assert probe.estimate_wait_s(p, gpus_requested=2) is None


def test_estimate_wait_s_unknown_when_no_running_jobs(probe):
    """All GPUs allocated but attach_queue_stats couldn't read squeue →
    running_jobs empty. We can't estimate; return None rather than lie
    with 0."""
    p = _make_partition(probe, gpus_total=1, gpus_alloc=1)
    assert probe.estimate_wait_s(p, gpus_requested=1) is None
```

You'll need `Optional` imported in the test file; if `from typing import Optional` isn't already there, add it at the top.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_cluster_probe.py -k "estimate_wait_s" -v`
Expected: `AttributeError: module 'cluster_probe' has no attribute 'estimate_wait_s'`.

- [ ] **Step 3: Implement `estimate_wait_s`**

Add directly after `attach_queue_stats` in `cluster_probe.py`:

```python
def estimate_wait_s(
    partition: "Partition", gpus_requested: int
) -> Optional[int]:
    """Estimate when ``gpus_requested`` GPUs will free up.

    Returns:
      - ``0`` if the partition already has that many GPUs free.
      - An integer (seconds) equal to the ``deficit``-th-smallest
        ``time_left_s`` across running jobs with known durations, where
        ``deficit = gpus_requested - gpus_free + pending_gpu_demand``.
        ``pending_gpu_demand`` is a conservative proxy for "jobs ahead
        of us in the queue" — squeue doesn't expose priority ordering
        reliably.
      - ``None`` if we can't tell because fewer than ``deficit`` running
        jobs have a known finite ``time_left_s`` (or none at all).
        Callers (pick_recommended) treat ``None`` as worse than any
        finite wait.
    """
    gpus_free = partition.gpus_free
    deficit = gpus_requested - gpus_free + partition.pending_gpu_demand
    if deficit <= 0:
        return 0
    known_finite = sorted(
        j.time_left_s for j in partition.running_jobs
        if j.time_left_s is not None
    )
    if len(known_finite) < deficit:
        return None
    # 0-indexed: we need the deficit-th to finish, which is index deficit-1.
    return known_finite[deficit - 1]
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "estimate_wait_s" -v`
Expected: all PASS.

- [ ] **Step 5: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: estimate_wait_s from squeue + scontrol state

Pure function: for a partition that isn't ready_now, returns the
deficit-th-smallest time_left_s across running jobs, where
deficit = gpus_requested - gpus_free + pending_gpu_demand. Returns None
when too few running jobs have known durations — callers treat None as
worse than any finite wait. No heuristics, no clock calls — fully
unit-testable.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `Alternative` dataclass + new `pick_recommended` with tier/a100/wait scoring

**Files:**
- Modify: `scripts/cluster_probe.py` (replace existing `pick_recommended`, add `Alternative` dataclass)
- Modify: `tests/test_cluster_probe.py` (new tests; adjust existing recommender tests that encode the old smallest-VRAM-first behaviour)

This is the policy core. The old `pick_recommended` returned `Optional[str]` (just the name). The new one returns `tuple[Optional[str], list[Alternative]]` — the recommended partition name (or `None` if the target is already ready and recommended) plus up to three ranked alternatives with full scoring rationale for the UI.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_cluster_probe.py`:

```python
# ---------------------------------------------------------------------------
# pick_recommended (queue-aware)
# ---------------------------------------------------------------------------

def _full_probe(probe, *, target: str, gpus: int, vram_need_gb: int, squeue_fixture: str = ""):
    """Convenience: run the full detect → aggregate → attach → fit pipeline
    against SCONTROL_FIXTURE (+ an optional squeue fixture) and return the
    pieces pick_recommended needs."""
    nodes = [
        n for line in SCONTROL_FIXTURE.splitlines()
        if (n := probe.parse_scontrol_node_line(line)) is not None
    ]
    parts_map = probe.aggregate_partitions(nodes)
    if squeue_fixture:
        jobs = [
            j for line in squeue_fixture.splitlines()
            if (j := probe.parse_squeue_line(line)) is not None
        ]
        probe.attach_queue_stats(parts_map, jobs)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition=target, gpus=gpus, mem_mb=0, cpus=0,
        vram_need_gb=vram_need_gb,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    return partitions, fits, req


def test_pick_recommended_returns_tuple(probe):
    """New signature: (recommended_name_or_None, list[Alternative])."""
    partitions, fits, req = _full_probe(
        probe, target="a100", gpus=1, vram_need_gb=80,
    )
    result = probe.pick_recommended(partitions, fits, req)
    assert isinstance(result, tuple) and len(result) == 2
    recommended, alternatives = result
    assert isinstance(alternatives, list)
    for alt in alternatives:
        # Alternative has at least these fields
        assert hasattr(alt, "partition")
        assert hasattr(alt, "wait_s")
        assert hasattr(alt, "tier")
        assert hasattr(alt, "reason")


def test_pick_recommended_prefers_nice_project_by_tier(probe):
    """Kiwi-DA (8 GB) on rtx_a6000_risk (contested). Both nice-project
    and l40s_risk are 48 GB READY with 0 wait — nice-project wins on
    tier (1 < 2)."""
    partitions, fits, req = _full_probe(
        probe, target="rtx_a6000_risk", gpus=1, vram_need_gb=8,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended == "nice-project"
    # Alternatives list: up to 3, first is the recommendation.
    assert len(alternatives) >= 1
    assert alternatives[0].partition == "nice-project"
    assert alternatives[0].wait_s == 0
    assert alternatives[0].tier == 1


def test_pick_recommended_excludes_blocklist(probe):
    """cogvis-project has 2 idle A6000s — exactly what this job wants —
    but it's blocklisted. Recommender MUST NOT suggest it."""
    partitions, fits, req = _full_probe(
        probe, target="rtx_a6000_risk", gpus=1, vram_need_gb=48,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended != "cogvis-project"
    for alt in alternatives:
        assert alt.partition != "cogvis-project"


def test_pick_recommended_a100_penalty_when_48g_available(probe):
    """Kiwi-DA (8 GB) with target=a100. nice-project (tier 1, 48 GB, READY)
    exists and is wait=0 — a100 must NOT win despite being free. A100
    penalty applies because ≥1 non-a100 READY partition is free."""
    partitions, fits, req = _full_probe(
        probe, target="a100", gpus=1, vram_need_gb=8,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended == "nice-project"
    # a100 is in alternatives but ranked lower than nice-project.
    a100_alt = next((a for a in alternatives if a.partition == "a100"), None)
    nice_alt = next((a for a in alternatives if a.partition == "nice-project"), None)
    assert nice_alt is not None
    if a100_alt is not None:
        # nice-project's tier (1) + 0 penalty < a100's tier (5) + 4 penalty.
        assert alternatives.index(nice_alt) < alternatives.index(a100_alt)


def test_pick_recommended_a100_no_penalty_when_vram_need_over_48(probe):
    """72B on gpus=4 needs 80 GB VRAM — only a100 fits. No penalty."""
    partitions, fits, req = _full_probe(
        probe, target="a100", gpus=4, vram_need_gb=80,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended == "a100"


def test_pick_recommended_a100_no_penalty_when_all_48g_contested(probe):
    """If every 48 GB partition has wait_s > 0, a100 competes on equal
    tier terms even for a 48 GB job. We simulate by manually setting
    wait_s via a squeue fixture that blocks every 48 GB partition."""
    # Build squeue where nice-project and l40s_risk are both fully
    # allocated (running jobs consuming every GPU, plus pending demand).
    squeue_fixture = "\n".join([
        # nice-project: all 2 L40s allocated, 1 very-long-running job
        "2001|nice-project|RUNNING|23:00:00|1:00:00|None|gres:gpu:2|1|u",
        # l40s_risk: its 1 IDLE node becomes fully busy
        "2002|l40s_risk|RUNNING|22:00:00|1:00:00|None|gres:gpu:2|1|u",
        # a100 free as in the base fixture (no jobs)
    ])
    # We need aggregated partitions whose gpus_free is 0 to make the
    # situation realistic — in SCONTROL_FIXTURE nice-project already has
    # 1 GPU alloc'd so the running-job time for its 2nd GPU dominates.
    nodes = [
        n for line in SCONTROL_FIXTURE.splitlines()
        if (n := probe.parse_scontrol_node_line(line)) is not None
    ]
    parts_map = probe.aggregate_partitions(nodes)
    # Pretend both 48 GB partitions are fully allocated:
    parts_map["nice-project"].gpus_alloc = parts_map["nice-project"].gpus_total
    parts_map["l40s_risk"].gpus_alloc = parts_map["l40s_risk"].gpus_total
    jobs = [
        j for line in squeue_fixture.splitlines()
        if (j := probe.parse_squeue_line(line)) is not None
    ]
    probe.attach_queue_stats(parts_map, jobs)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=48,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    # a100 should now be admissible (penalty dropped because no
    # non-a100 READY partition is wait=0 any more).
    a100_alt = next((a for a in alternatives if a.partition == "a100"), None)
    assert a100_alt is not None
    # a100 is ranked at or near the top — not buried beneath contested
    # 48 GB options with non-zero wait.
    assert alternatives[0].partition in {"a100", "3090", "l40s_risk", "rtx_a6000_risk"}


def test_pick_recommended_no_alternatives_when_target_is_best(probe):
    """Kiwi-DA on nice-project — already tier 1, wait=0. Recommended
    equals target → caller treats this as 'no prompt needed'."""
    partitions, fits, req = _full_probe(
        probe, target="nice-project", gpus=1, vram_need_gb=8,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended == "nice-project"    # already best
    # Alternatives list still populated (submit.sh may show a table),
    # and the first entry is nice-project.
    assert alternatives[0].partition == "nice-project"


def test_pick_recommended_no_recommendation_when_nothing_fits(probe):
    """72B on gpus=1 after tp-skip → vram_need collapses to 8 GB, every
    partition fits. But force an impossible mem requirement to verify
    the 'nothing fits' path."""
    partitions, fits, req = _full_probe(
        probe, target="nice-project", gpus=1, vram_need_gb=8,
    )
    # Pretend nothing fits by replacing every fit with shape_ok=False.
    fits_blocked = {
        name: probe.FitStatus(
            partition=name, shape_ok=False, has_capacity_now=False,
            vram_known=True, reasons=["forced"],
        )
        for name in fits
    }
    recommended, alternatives = probe.pick_recommended(
        partitions, fits_blocked, req,
    )
    assert recommended is None
    assert alternatives == []


def test_pick_recommended_returns_at_most_three_alternatives(probe):
    """Many eligible partitions → truncate to top 3."""
    partitions, fits, req = _full_probe(
        probe, target="nice-project", gpus=1, vram_need_gb=8,
    )
    _, alternatives = probe.pick_recommended(partitions, fits, req)
    assert len(alternatives) <= 3


# ---------------------------------------------------------------------------
# Old pick_recommended behaviour that the new scoring intentionally changes
# is updated here: the legacy tests (test_pick_recommended_when_target_contested,
# test_pick_recommended_prefers_smallest_fitting, test_pick_recommended_skips_unknown_vram,
# test_pick_recommended_none_when_target_ready) now expect the tuple return
# shape and tier-driven outcome.
# ---------------------------------------------------------------------------
```

Then edit the existing recommender tests so they unpack the tuple and encode the *new* policy:

Old `test_pick_recommended_when_target_contested`: replace its body with:

```python
def test_pick_recommended_when_target_contested(probe):
    """Target rtx_a6000_risk contested; nice-project wins on tier."""
    partitions, fits, req = _full_probe(
        probe, target="rtx_a6000_risk", gpus=1, vram_need_gb=48,
    )
    recommended, _ = probe.pick_recommended(partitions, fits, req)
    assert recommended == "nice-project"
```

Old `test_pick_recommended_none_when_target_ready`: replace with:

```python
def test_pick_recommended_none_when_target_ready(probe):
    """a100 target, 80 GB VRAM need, free now — recommendation equals
    target (caller treats matching recommendation as 'no prompt')."""
    partitions, fits, req = _full_probe(
        probe, target="a100", gpus=1, vram_need_gb=80,
    )
    recommended, _ = probe.pick_recommended(partitions, fits, req)
    assert recommended == "a100"
```

Old `test_pick_recommended_skips_unknown_vram`: replace with:

```python
def test_pick_recommended_skips_unknown_vram(probe):
    partitions, fits, req = _full_probe(
        probe, target="rtx_a6000_risk", gpus=1, vram_need_gb=48,
    )
    recommended, alternatives = probe.pick_recommended(partitions, fits, req)
    assert recommended != "experimental"
    for alt in alternatives:
        assert alt.partition != "experimental"
```

Old `test_pick_recommended_prefers_smallest_fitting` — delete it. The new policy prefers by tier, not by VRAM waste as primary. VRAM waste is the 4th tiebreaker, and the test scenario (3090 wins for an 8 GB job) is no longer true when nice-project is available.

- [ ] **Step 2: Run the new tests; they should fail against the old `pick_recommended`**

Run: `pytest tests/test_cluster_probe.py -k "pick_recommended" -v`
Expected: many failures — old `pick_recommended` returns `Optional[str]`, not a tuple, and has no tier logic.

- [ ] **Step 3: Replace `pick_recommended` in `cluster_probe.py`**

Find the existing `def pick_recommended(...)` and replace the *entire* function plus add an `Alternative` dataclass immediately above it:

```python
@dataclass
class Alternative:
    """A ranked partition suggestion returned by :func:`pick_recommended`.

    ``wait_s`` is ``0`` when the partition has capacity right now; a
    positive int for estimated wait in seconds; ``None`` when squeue
    info is incomplete (callers render ``?`` and treat ``None`` as
    worse-than-any-finite-wait when ranking).
    """

    partition: str
    wait_s: Optional[int]
    tier: int
    vram_waste_gb: int
    gpus_free: int
    reason: str


def _a100_penalty_applies(
    vram_need_gb: int, fits: dict[str, "FitStatus"],
    partitions: list["Partition"],
) -> bool:
    """True iff a100 should be de-prioritised for this request.

    a100 gets a +4 tier penalty when ``vram_need <= 48`` AND at least
    one non-a100 partition is both READY (``has_capacity_now``) and
    VRAM-known. That's the 'don't waste the 80 GB card when 48 GB is
    idle' case. When vram_need > 48 OR every non-a100 option is
    contested/unknown, the penalty drops and a100 competes on equal
    tier terms.
    """
    if vram_need_gb > 48:
        return False
    for p in partitions:
        if p.name == "a100" or is_blocklisted(p.name):
            continue
        fit = fits.get(p.name)
        if fit is None:
            continue
        if fit.has_capacity_now and fit.vram_known:
            return True
    return False


_WAIT_UNKNOWN = 10**12  # sort key proxy for "worse than any finite wait"


def pick_recommended(
    partitions: list["Partition"],
    fits: dict[str, "FitStatus"],
    req: "Request",
) -> tuple[Optional[str], list[Alternative]]:
    """Rank eligible partitions and return (recommended_name, alternatives).

    Eligibility rules (partition must pass ALL to appear in alternatives):
      - not blocklisted (see PARTITIONS_BLOCKLIST)
      - shape_ok AND vram_known
      - max_vram_gb >= req.vram_need_gb (or vram_need_gb is 0 / default)

    Ranking tuple (ascending; lower is better):
      (ready_now_rank,
       tier + a100_penalty,
       wait_s_sortable,        # 0 if ready_now else wait_s; None → _WAIT_UNKNOWN
       vram_waste_gb,
       -gpus_free)

    Returns:
      - recommended: the highest-ranked partition's name, or ``None`` if
        no partition is eligible.
      - alternatives: the top-3 ranked partitions with full scoring
        rationale. May be empty when nothing is eligible.
    """
    a100_penalty_on = _a100_penalty_applies(req.vram_need_gb, fits, partitions)

    eligible: list["Partition"] = []
    for p in partitions:
        if is_blocklisted(p.name):
            continue
        fit = fits.get(p.name)
        if fit is None or not fit.shape_ok or not fit.vram_known:
            continue
        if req.vram_need_gb and p.max_vram_gb < req.vram_need_gb:
            continue
        eligible.append(p)

    if not eligible:
        return None, []

    def score(p: "Partition") -> tuple[int, int, int, int, int]:
        fit = fits[p.name]
        ready_now = fit.has_capacity_now
        ready_now_rank = 0 if ready_now else 1
        tier = partition_tier(p.name)
        a100_bonus = 4 if (p.name == "a100" and a100_penalty_on) else 0
        if ready_now:
            wait_key = 0
        else:
            w = estimate_wait_s(p, req.gpus)
            wait_key = _WAIT_UNKNOWN if w is None else w
        vram_waste = max(p.max_vram_gb - req.vram_need_gb, 0)
        return (ready_now_rank, tier + a100_bonus, wait_key, vram_waste, -p.gpus_free)

    ranked = sorted(eligible, key=score)

    def _reason(p: "Partition") -> str:
        fit = fits[p.name]
        if fit.has_capacity_now:
            if partition_tier(p.name) == 1:
                return "free now, group partition"
            return "free now"
        w = estimate_wait_s(p, req.gpus)
        if w is None:
            return "no immediate capacity; wait unknown"
        return f"no immediate capacity; est. wait {w // 60} min"

    alternatives: list[Alternative] = []
    for p in ranked[:3]:
        fit = fits[p.name]
        if fit.has_capacity_now:
            wait_s: Optional[int] = 0
        else:
            wait_s = estimate_wait_s(p, req.gpus)
        alternatives.append(
            Alternative(
                partition=p.name,
                wait_s=wait_s,
                tier=partition_tier(p.name),
                vram_waste_gb=max(p.max_vram_gb - req.vram_need_gb, 0),
                gpus_free=p.gpus_free,
                reason=_reason(p),
            )
        )

    return alternatives[0].partition, alternatives
```

- [ ] **Step 4: Run the new tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "pick_recommended" -v`
Expected: all pick_recommended tests PASS (including updated legacy tests).

- [ ] **Step 5: Full regression — `main()` still uses the old signature**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: The `main()` end-to-end tests may fail because `main()` still does `recommended = pick_recommended(...)` and treats it as a string. That's Task 6's job. For now, temporarily fix `main()` to unpack the tuple so regression stays green:

Edit `cluster_probe.py` `main()`:

Find:
```python
    recommended = pick_recommended(partitions, fits, req)
```

Replace with:
```python
    recommended_name, alternatives = pick_recommended(partitions, fits, req)
    # Legacy alias kept for the current rendering block; Task 6 replaces
    # the downstream behaviour with full alternative-list rendering.
    recommended = (
        recommended_name
        if recommended_name and recommended_name != req.partition
        else None
    )
```

Re-run the full probe tests:

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: tier + blocklist + a100-aware recommender

Rewrites pick_recommended to return (recommended, alternatives[≤3]).
Ranking: ready_now, then tier+a100_penalty, then wait_s (estimate, None
treated as worst), then vram_waste, then -gpus_free.

A100 gets a +4 tier penalty only when vram_need<=48 AND at least one
non-a100 READY VRAM-known partition exists — else it competes on equal
tier terms. cogvis-project is hard-excluded via PARTITIONS_BLOCKLIST.

main() temporarily adapts to the new tuple return (rendering stays the
same shape for now; Task 6 will surface the alternatives list).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Exit codes 3 & 4 + blocklisted-target rejection

**Files:**
- Modify: `scripts/cluster_probe.py` (`main()` exit-code policy + new block)
- Modify: `tests/test_cluster_probe.py` (new end-to-end tests)

Exit code 3 reserved by the *original* probe for "scontrol unavailable" — we keep that meaning (probe failure). For blocklisted target we use exit code 5 (new) and for "nothing fits anywhere" exit code 4 (new). This preserves submit.sh's existing handling.

Wait — re-read the spec. The spec says:

```
0  target VRAM-fits
2  target NO-FIT (but at least one other partition fits)
3  target is blocklisted
4  no partition can fit the job at the requested gpus count
```

And the *existing* probe uses `3` for "probe itself failed". That collision breaks backward compat. Resolution: shift blocklisted → 5, keep probe-failure → 3, add nothing-fits → 4. Update the design spec inline before implementing.

- [ ] **Step 1: Fix the spec to resolve the exit-code collision**

Edit `docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md`. Find:

```
  8. exit:
       0  target VRAM-fits (may still be sub-optimal vs. recommender)
       2  target NO-FIT (but at least one other partition fits)
       3  target is blocklisted
       4  no partition can fit the job at the requested gpus count
```

Replace with:

```
  8. exit:
       0  target VRAM-fits (may still be sub-optimal vs. recommender)
       2  target NO-FIT (but at least one other partition fits)
       3  probe itself failed (scontrol missing or unparseable) —
          preserved from the original probe for submit.sh back-compat
       4  no partition can fit the job at the requested gpus count
       5  target partition is blocklisted
```

Commit separately so the change is traceable:

```bash
git add docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md
git commit -m "$(cat <<'EOF'
spec: pin exit codes to avoid collision with legacy '3 = probe failed'

The existing probe already uses exit 3 for 'scontrol unavailable /
unparseable', and submit.sh's [5/6] block branches on it. Keep that
meaning and allocate the new exits as:

  4 = no partition can fit the job
  5 = target partition is blocklisted

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Write failing tests for the new exit codes + rejection**

Append to `tests/test_cluster_probe.py`:

```python
# ---------------------------------------------------------------------------
# Exit codes: blocklisted target + nothing-fits
# ---------------------------------------------------------------------------

def test_main_blocklisted_target_exit_five(probe, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: "")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "cogvis-project", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 5
    out = capsys.readouterr().out + capsys.readouterr().err
    # The rejection message names the partition and mentions blocklist.
    assert "cogvis-project" in out
    assert "blocklist" in out.lower()


def test_main_nothing_fits_exit_four(probe, monkeypatch, capsys, tmp_path):
    """Force every partition to fail shape (oversize mem) → exit 4."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: "")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--mem", "999999G",   # enormous → no node qualifies
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 4


def test_main_probe_exit_three_preserved(probe, monkeypatch):
    """scontrol unavailable still returns 3, same as before."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: None)
    rc = probe.main([
        "--partition", "a100",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 3
```

- [ ] **Step 3: Run the tests and confirm failure**

Run: `pytest tests/test_cluster_probe.py -k "blocklisted_target or nothing_fits or probe_exit_three" -v`
Expected: `test_main_blocklisted_target_exit_five` and `test_main_nothing_fits_exit_four` FAIL (wrong exit codes); `test_main_probe_exit_three_preserved` PASSES.

- [ ] **Step 4: Add blocklist rejection + nothing-fits exit to `main()`**

In `cluster_probe.py` `main()`, after the existing `target_fit = fits.get(req.partition)` check and before the `recommended = pick_recommended(...)` line, insert:

```python
    # Blocklist rejection. A blocklisted partition is a hard NO — no
    # prompt, no sbatch; user must pick something else. submit.sh
    # surfaces the exit code; here we just print + exit 5.
    if is_blocklisted(req.partition):
        msg = (
            f"partition '{req.partition}' is in the cluster-probe "
            f"blocklist (belongs to another faculty/group). Pick a "
            f"different partition."
        )
        if args.json:
            print(json.dumps({"error": msg, "blocklist": sorted(PARTITIONS_BLOCKLIST)}, indent=2))
        else:
            print(f"cluster_probe: {msg}", file=sys.stderr)
            # Still print the survey so the user sees what they could use.
            print()
            print(render_table(partitions, fits, req, colour=(not args.no_colour) and sys.stdout.isatty()))
        return 5
```

Then update the existing exit-code policy at the bottom. Find:

```python
    # Exit code policy.
    if not target_fit.shape_ok:
        return 1
    if not target_fit.has_capacity_now:
        return 2
    return 0
```

Replace with:

```python
    # Exit code policy.
    # 4 (nothing fits anywhere) takes precedence over 1/2: if we can't
    # recommend any partition at all, the user's config is broken regardless
    # of which target they named.
    recommended_name, alternatives = pick_recommended_result
    if not alternatives and not target_fit.shape_ok:
        return 4
    if not target_fit.shape_ok:
        return 1
    if not target_fit.has_capacity_now:
        return 2
    return 0
```

And find the earlier temporary adapter line:

```python
    recommended_name, alternatives = pick_recommended(partitions, fits, req)
    # Legacy alias kept for the current rendering block; Task 6 replaces
    # the downstream behaviour with full alternative-list rendering.
    recommended = (
        recommended_name
        if recommended_name and recommended_name != req.partition
        else None
    )
```

Replace with:

```python
    pick_recommended_result = pick_recommended(partitions, fits, req)
    recommended_name, alternatives = pick_recommended_result
    # Legacy alias kept for the current rendering block; Task 8 replaces
    # the downstream behaviour with full alternative-list rendering.
    recommended = (
        recommended_name
        if recommended_name and recommended_name != req.partition
        else None
    )
```

(The change is just giving the tuple a name so the exit-code block can reach `alternatives`.)

- [ ] **Step 5: Run tests and confirm they pass**

Run: `pytest tests/test_cluster_probe.py -k "blocklisted_target or nothing_fits or probe_exit_three" -v`
Expected: all PASS.

- [ ] **Step 6: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: blocklist rejection (exit 5) + nothing-fits (exit 4)

Exit code policy is now:
  0  target fits and has capacity
  1  target NO-FIT (shape violation) but alternatives exist
  2  target shape-fits but no free GPUs right now
  3  probe itself failed (unchanged — submit.sh back-compat)
  4  no partition can fit the job at the requested --gres=gpu:N
  5  target partition is blocklisted (hard reject, no prompt)

Blocklisted target prints the survey + the rejection reason and exits
5 — no prompt, no sbatch. submit.sh will catch both 4 and 5 in the
next task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `--tee-alternatives=<path>` TSV writer + wire squeue into `main()`

**Files:**
- Modify: `scripts/cluster_probe.py` (new CLI flag, wire `run_squeue` → `attach_queue_stats` → TSV)
- Modify: `tests/test_cluster_probe.py` (TSV round-trip test)

Exposes the alternatives list to `submit.sh` as structured data. Also finally wires `run_squeue()` + `attach_queue_stats()` into `main()` — until now they were callable but not called.

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# --tee-alternatives TSV sidecar
# ---------------------------------------------------------------------------

def test_main_tee_alternatives_writes_tsv(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: "")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    tee = tmp_path / "alts.tsv"
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--tee-alternatives", str(tee),
        "--no-colour",
    ])
    assert rc == 0
    assert tee.is_file()
    lines = tee.read_text().splitlines()
    # Header + at least 1 alternative
    assert lines[0] == "rank\tpartition\tgpus_requested\twait_s\ttier\treason"
    assert len(lines) >= 2
    # Each data row has 6 tab-separated fields
    for row in lines[1:]:
        assert len(row.split("\t")) == 6


def test_main_tee_alternatives_kiwi_da_recommends_nice_project(probe, monkeypatch, tmp_path):
    """TSV content must reflect the tier-1 preference for a small job."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: "")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    tee = tmp_path / "alts.tsv"
    probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--tee-alternatives", str(tee),
        "--no-colour",
    ])
    lines = tee.read_text().splitlines()
    # Second line (rank 1) partition is nice-project.
    cols = lines[1].split("\t")
    assert cols[0] == "1"
    assert cols[1] == "nice-project"


def test_main_calls_run_squeue(probe, monkeypatch, capsys, tmp_path):
    """Smoke: main() must actually invoke run_squeue — otherwise the
    queue-aware scoring is decoration, not behaviour."""
    called = {"n": 0}
    def fake_squeue():
        called["n"] += 1
        return ""  # no jobs, but it was called
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", fake_squeue)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert called["n"] == 1
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_cluster_probe.py -k "tee_alternatives or calls_run_squeue" -v`
Expected: `test_main_calls_run_squeue` fails (run_squeue not invoked); the two TSV tests fail (no `--tee-alternatives` flag).

- [ ] **Step 3: Add the flag + TSV writer + wire squeue into `main()`**

In `main()`, add the flag next to `--no-colour`:

```python
    p.add_argument("--tee-alternatives", type=Path, default=None,
                   help="write ranked alternatives to this TSV path "
                        "(one row per alternative; used by submit.sh).")
```

Right after `nodes = [...]` and before `partitions_map = aggregate_partitions(nodes)` (keep the scontrol failure branch above untouched), add the squeue hookup:

```python
    raw_squeue = run_squeue()
    jobs: list[Job] = []
    if raw_squeue:
        for line in raw_squeue.splitlines():
            j = parse_squeue_line(line)
            if j is not None:
                jobs.append(j)
    # attach_queue_stats tolerates jobs on unknown partitions; it's safe
    # to call with an empty jobs list when squeue was unavailable.
```

Then, right after `partitions_map = aggregate_partitions(nodes)`:

```python
    attach_queue_stats(partitions_map, jobs)
```

Finally, add the TSV writer. Find the bottom of `main()` — the block that returns the exit code. Immediately BEFORE the exit-code policy, add:

```python
    if args.tee_alternatives is not None:
        try:
            args.tee_alternatives.parent.mkdir(parents=True, exist_ok=True)
            with args.tee_alternatives.open("w") as fh:
                fh.write("rank\tpartition\tgpus_requested\twait_s\ttier\treason\n")
                for i, alt in enumerate(alternatives, start=1):
                    wait_s_str = "" if alt.wait_s is None else str(alt.wait_s)
                    fh.write(
                        f"{i}\t{alt.partition}\t{req.gpus}\t"
                        f"{wait_s_str}\t{alt.tier}\t{alt.reason}\n"
                    )
        except OSError as e:
            print(
                f"cluster_probe: could not write --tee-alternatives file "
                f"{args.tee_alternatives}: {e}",
                file=sys.stderr,
            )
            # Non-fatal: we still return the probe's normal exit code.
```

- [ ] **Step 4: Run the TSV tests + squeue-called test**

Run: `pytest tests/test_cluster_probe.py -k "tee_alternatives or calls_run_squeue" -v`
Expected: all PASS.

- [ ] **Step 5: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: --tee-alternatives TSV + wire squeue into main

Writes the top-3 alternatives returned by pick_recommended as a TSV
(rank, partition, gpus_requested, wait_s, tier, reason) so submit.sh
can present an interactive prompt without re-parsing the probe's
human-readable output.

Also wires run_squeue + attach_queue_stats into main() so every run
picks up live queue data; attach_queue_stats already tolerates empty
jobs lists when squeue is unavailable, so this is safe on submit nodes
without squeue on PATH.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Render updates — table columns + JSON alternatives + UNKNOWN tags

**Files:**
- Modify: `scripts/cluster_probe.py` (`render_table`, JSON payload, final footer in `main`)
- Modify: `tests/test_cluster_probe.py` (assertions on new rendering)

Widens the survey table: new `next free` and `pending` columns; blocklisted partitions rendered with `[not ours]` tag (and never as the recommended one). JSON payload gains an `alternatives` list mirroring the TSV.

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# Render + JSON updates
# ---------------------------------------------------------------------------

def test_render_table_has_queue_columns(probe, monkeypatch, capsys, tmp_path):
    """The human-readable table must show `next free` and `pending` columns."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: SQUEUE_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    out = capsys.readouterr().out
    assert "next free" in out
    assert "pending" in out


def test_render_table_tags_blocklisted(probe, monkeypatch, capsys, tmp_path):
    """cogvis-project row must be tagged [not ours] and not recommended."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: "")
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    out = capsys.readouterr().out
    assert "cogvis-project" in out
    assert "[not ours]" in out


def test_json_output_includes_alternatives(probe, monkeypatch, capsys, tmp_path):
    import json as jsonmod
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    monkeypatch.setattr(probe, "run_squeue", lambda: SQUEUE_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--json",
    ])
    payload = jsonmod.loads(capsys.readouterr().out)
    assert "alternatives" in payload
    assert isinstance(payload["alternatives"], list)
    assert len(payload["alternatives"]) >= 1
    first = payload["alternatives"][0]
    for key in ("partition", "wait_s", "tier", "vram_waste_gb", "gpus_free", "reason"):
        assert key in first
    # With this config (Kiwi-DA 8 GB), nice-project should top the list.
    assert payload["alternatives"][0]["partition"] == "nice-project"
    assert payload["recommendation"] == "nice-project"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/test_cluster_probe.py -k "queue_columns or tags_blocklisted or includes_alternatives" -v`
Expected: all FAIL.

- [ ] **Step 3: Update `render_table` with the new columns + blocklist tag**

Replace the whole `render_table` function body. The column widths and sort stay largely the same; two new columns appear after `vram`:

Find the existing `render_table` and replace with:

```python
def render_table(
    partitions: list[Partition],
    fits: dict[str, FitStatus],
    req: Request,
    colour: bool = True,
) -> str:
    c = _ansi(colour)
    lines = []
    lines.append(
        f"{c['bold']}AISURREY partition survey (target: "
        f"{c['yel']}{req.partition}{c['reset']}{c['bold']} × "
        f"{req.gpus} GPU, need ≥{req.vram_need_gb} GB VRAM){c['reset']}"
    )
    header = (
        f"  {'partition':<18}  {'status':<13}  {'gpus (free/total)':<18}  "
        f"{'type':<22}  {'vram':<6}  {'next free':<10}  {'pending':<7}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    def sort_key(p: Partition) -> tuple[int, int, int]:
        fit = fits.get(p.name)
        if p.name == req.partition:
            return (0, 0, 0)
        if fit is None:
            return (5, 0, 0)
        if is_blocklisted(p.name):
            return (4, 0, 0)
        rank = {
            "ready": 1,
            "contested": 2,
            "unknown-vram": 3,
            "no-fit": 5,
        }[fit.status_word()]
        return (rank, -p.gpus_free, 0)
    has_mixed_any = False
    for p in sorted(partitions, key=sort_key):
        fit = fits.get(p.name)
        blocked = is_blocklisted(p.name)
        if blocked:
            colour_code = c["dim"]
            tag = "[not ours]"
        elif fit is None:
            colour_code = c["dim"]
            tag = "?"
        else:
            status = fit.status_word()
            if status == "ready":
                colour_code, tag = c["grn"], "READY"
            elif status == "contested":
                colour_code, tag = c["yel"], "CONTESTED"
            elif status == "unknown-vram":
                colour_code, tag = c["yel"], "UNKNOWN-VRAM"
            elif status == "no-fit":
                colour_code, tag = c["red"], "NO-FIT"
            else:
                colour_code, tag = c["dim"], "?"
        mark = " *" if p.name == req.partition else "  "
        gpu_cell = f"{p.gpus_free}/{p.gpus_total}"
        gpu_type = p.dominant_gpu_type or "—"
        if p.has_mixed_gpu_types:
            gpu_type = f"{gpu_type}+"
            has_mixed_any = True
        vram = f"{p.max_vram_gb}G" if p.max_vram_gb else "—"
        # next free: "now" if ready + capacity; HHhMMm if running job has a
        # known time left; "?" otherwise.
        if p.gpus_free >= req.gpus and not blocked:
            nf = "now"
        elif p.earliest_free_s is not None:
            hh = p.earliest_free_s // 3600
            mm = (p.earliest_free_s % 3600) // 60
            nf = f"{hh}h{mm:02d}m" if hh else f"{mm}m"
        else:
            nf = "?"
        pending = str(len(p.pending_jobs))
        lines.append(
            f"{mark}{p.name:<18}  {colour_code}{tag:<13}{c['reset']}  "
            f"{gpu_cell:<18}  {gpu_type:<22}  {vram:<6}  "
            f"{nf:<10}  {pending:<7}"
        )
    if has_mixed_any:
        lines.append(
            f"  {c['dim']}'+' = mixed-hardware partition (vram/type shown "
            f"for the fattest card){c['reset']}"
        )
    return "\n".join(lines)
```

Then update the JSON payload in `main()`. Find:

```python
    if args.json:
        payload = {
            "request": asdict(req),
            "target_fit": asdict(target_fit),
            "partitions": [asdict(p) | {"gpus_free": p.gpus_free} for p in partitions],
            "recommendation": recommended,
            "vram_inference": asdict(vram_detail) if vram_detail else None,
        }
        print(json.dumps(payload, indent=2))
```

Replace with:

```python
    if args.json:
        payload = {
            "request": asdict(req),
            "target_fit": asdict(target_fit),
            "partitions": [asdict(p) | {"gpus_free": p.gpus_free} for p in partitions],
            "recommendation": recommended_name,
            "alternatives": [asdict(a) for a in alternatives],
            "vram_inference": asdict(vram_detail) if vram_detail else None,
        }
        print(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Run the rendering tests**

Run: `pytest tests/test_cluster_probe.py -k "queue_columns or tags_blocklisted or includes_alternatives" -v`
Expected: all PASS.

- [ ] **Step 5: Full regression**

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS. Existing `test_main_json_output_structure` should still pass: the payload additions are additive, and the assertions only check for a subset of keys.

- [ ] **Step 6: Commit**

```bash
git add scripts/cluster_probe.py tests/test_cluster_probe.py
git commit -m "$(cat <<'EOF'
cluster_probe: surface queue stats + alternatives in rendering

Table grows two columns: 'next free' (earliest time a GPU frees on
that partition, from attach_queue_stats) and 'pending' (pending job
count). Blocklisted partitions render with a '[not ours]' tag in dim
grey and sort after real options. JSON output gains an 'alternatives'
list with partition/wait_s/tier/vram_waste_gb/gpus_free/reason per
entry, so machine consumers see the same ranking as the TSV sidecar.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `submit.sh` interactive prompt + `--stay-on-target` + `SUBMIT_AUTO_ROUTE`

**Files:**
- Modify: `scripts/submit.sh` (section `[5/6] cluster probe`)
- Create: `tests/test_submit_sh_prompt.bats` — optional; we mainly test via bash directly below

The `[5/6]` block gets larger. It now:

1. Passes `--tee-alternatives=$TMPDIR/alts.tsv` to the probe.
2. Branches on the new exit codes 4 and 5.
3. When exit is 2 (contested), or exit is 0 AND the recommendation ≠ target, prompts the user (or auto-routes).
4. `--stay-on-target` flag and `SUBMIT_AUTO_ROUTE=1` env var skip the prompt.

Integration tests use piped stdin to simulate user input. Since `submit.sh` requires `sbatch`, `conda`, `scontrol` on PATH (none of which exist in this dev checkout), we test the *prompt harness* as an extracted helper function that's sourceable and unit-testable.

- [ ] **Step 1: Design: factor out `_prompt_alternative()` in a way we can test**

Write a minimal shell unit test harness under `tests/test_submit_prompt.bash` that sources `submit.sh` with `SUBMIT_TEST_SKIP_PREFLIGHT=1` — a new env var that lets the prompt function be tested without actually running scontrol, conda, sinfo, etc.

Create `tests/test_submit_prompt.bash`:

```bash
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

for t in t_pick_alt_1 t_pick_alt_2 t_cancel_c t_timeout_cancel t_auto_route; do
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
```

- [ ] **Step 2: Run the test harness and confirm it fails**

Run: `bash tests/test_submit_prompt.bash`
Expected: `_prompt_alternative: command not found`.

- [ ] **Step 3: Add `_prompt_alternative` (and the test hook) to `submit.sh`**

At the very top of `submit.sh`, right after `set -euo pipefail`, add:

```bash
# ---------------------------------------------------------------------------
# Queue-aware alternative-picking prompt. See
# docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md.
#
# _prompt_alternative(target_partition, alts_tsv_path)
#   stdin:  user input ("1"/"2"/"3" or "c"); no keypress within
#           ${SUBMIT_PROMPT_TIMEOUT:-15}s = cancel
#   stdout: chosen partition name on success
#   exit:   0 = chosen partition printed on stdout
#           7 = user cancelled (or timed out)
#
# Test hook: SUBMIT_TEST_SKIP_PREFLIGHT=1 makes submit.sh return after
# defining helpers, so `bash tests/test_submit_prompt.bash` can source
# it and call the helper directly without running scontrol/sbatch.

_prompt_alternative() {
    local target="$1"
    local alts_tsv="$2"
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
        else
            local hh=$((wait_s / 3600))
            local mm=$(((wait_s % 3600) / 60))
            if [[ "$hh" -gt 0 ]]; then
                wait_human="wait: ${hh}h${mm}m"
            else
                wait_human="wait: ${mm}m"
            fi
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

    {
        echo "Your target: $target"
        echo "Recommender's ranking:"
        for label in "${labels[@]}"; do
            echo "$label"
        done
        echo -n "Pick 1-${#parts[@]} or c to cancel (${timeout}s, no default): "
    } >&2

    local choice=""
    if ! read -r -t "$timeout" choice; then
        echo >&2
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

# Early exit for the test harness. Defines the helper above and returns
# before running any of the pre-flight steps.
if [[ "${SUBMIT_TEST_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
```

- [ ] **Step 4: Run the test harness and confirm it passes**

Run: `bash tests/test_submit_prompt.bash`
Expected: `passed: 5  failed: 0`.

- [ ] **Step 5: Wire the helper into the `[5/6] cluster probe` block**

Replace the existing `case "$PROBE_RC" in ... esac` block in `submit.sh` with:

```bash
    case "$PROBE_RC" in
        0)
            ok "cluster probe: target partition has capacity"
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
                ROUTE_TO=$(_prompt_alternative "$PARTITION" "$ALTS_TSV")
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
            red "  cluster probe: no partition can fit this job shape at --gres=gpu:$GPU_COUNT."
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
```

Before the `case`, right after `PROBE_ARGS=(...)` is built, set the TSV path and pass it through:

```bash
    ALTS_TSV="$(mktemp)"
    PROBE_ARGS+=("--tee-alternatives" "$ALTS_TSV")
```

Also, at the top argument-parsing loop (around `while [[ $# -gt 0 ]]`), add the new flag:

Find:
```bash
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
```

Replace with:
```bash
STAY_ON_TARGET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --stay-on-target) STAY_ON_TARGET=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
```

And inside the usage/help heredoc at the top, add `[--stay-on-target]` and a short sentence:

Find the usage heredoc (the one starting `Usage: scripts/submit.sh`) and adjust it:

```bash
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

Every AISURREY sbatch goes through this wrapper. If you're tempted to run
sbatch directly: don't. That's how partition / env / node typos get past
the pre-flight and eat half a morning.
```

Finally, add a sub-optimal-but-READY branch right after the probe step succeeds (code 0) that triggers the same prompt if the TSV's rank-1 partition isn't the user's target. Replace the `0)` case with:

```bash
        0)
            ok "cluster probe: target partition has capacity"
            # Even when target is READY, the recommender may prefer a
            # different partition (tier / a100-penalty). Surface that
            # choice unless the user said --stay-on-target or set
            # SUBMIT_AUTO_ROUTE to accept it unattended.
            if [[ "${STAY_ON_TARGET:-0}" == "1" ]]; then
                :
            elif [[ -f "$ALTS_TSV" ]]; then
                TOP_ALT=$(awk -F'\t' 'NR==2{print $2}' "$ALTS_TSV")
                if [[ -n "$TOP_ALT" ]] && [[ "$TOP_ALT" != "$PARTITION" ]]; then
                    yel "  recommender prefers '$TOP_ALT' over '$PARTITION'."
                    set +e
                    ROUTE_TO=$(_prompt_alternative "$PARTITION" "$ALTS_TSV")
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
```

- [ ] **Step 6: Run the bash tests again + the full pytest suite**

Run: `bash tests/test_submit_prompt.bash`
Expected: `passed: 5  failed: 0`.

Run: `pytest tests/test_cluster_probe.py -v`
Expected: all PASS (submit.sh changes don't touch Python).

- [ ] **Step 7: Smoke test `submit.sh --stay-on-target` + `SUBMIT_AUTO_ROUTE` locally**

Run (expect failure about missing sbatch/conda env, which is fine — we only care the flag/env var are accepted):

```bash
scripts/submit.sh --stay-on-target configs/runs/example_quick.yaml 2>&1 | head -30
SUBMIT_AUTO_ROUTE=1 scripts/submit.sh configs/runs/example_quick.yaml 2>&1 | head -30
```
Expected: fails at the conda env check with a `FAIL: conda env not found at ...` — i.e. the flags are silently accepted, not complained about.

- [ ] **Step 8: Commit**

```bash
git add scripts/submit.sh tests/test_submit_prompt.bash
git commit -m "$(cat <<'EOF'
submit.sh: interactive prompt for queue-aware partition rerouting

Cluster-probe step [5/6] now branches on the full exit-code set:
  0 → ok; also prompt if recommender's top choice != target
  1 → target NO-FIT, hard fail
  2 → contested; prompt with alternatives from --tee-alternatives TSV
  3 → probe couldn't run, proceed to --test-only (unchanged)
  4 → nothing fits anywhere; hard fail
  5 → target is blocklisted (cogvis-project); hard fail

Prompt: 15 s default timeout, reads 1/2/3 or c (cancel). No default on
timeout — user must pick or cancel. Escape hatches:
  --stay-on-target   flag, skips prompt entirely
  SUBMIT_AUTO_ROUTE=1 env var, accepts alt #1 unattended

Added tests/test_submit_prompt.bash that sources submit.sh with
SUBMIT_TEST_SKIP_PREFLIGHT=1 and exercises _prompt_alternative in
isolation (piped stdin; SUBMIT_PROMPT_TIMEOUT=2 for fast CI).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Documentation updates

**Files:**
- Modify: `examples/04_aisurrey_submission.md`
- Modify: `docs/AISURREY.md` (add subsection; current file already explains submit.sh's pre-flight)

- [ ] **Step 1: Update `examples/04_aisurrey_submission.md`**

Replace the submission-output example in section `## 1. Dry-run the submission (pre-flight only)` — the block currently ending at `dry-run mode: pre-flight OK, not submitting.` — with a version that shows the queue-aware prompt and the new columns.

Find:
```
The wrapper prints its checks as it goes:
```

... and the block that follows it. Replace that block with:

````
The wrapper prints its checks as it goes:
```
[1/6] config file check...     ok: configs/runs/surrey_legal_full_matrix.yaml
[2/6] partition sanity check...ok: partition 'nice-project' exists
[3/6] conda env check...       ok: conda env present at .../conda_env
[4/6] duplicate job check...   ok: no duplicates of 'surrey_legal_full_matrix' in queue
[5/6] cluster probe (live capacity + VRAM fit)...

  AISURREY partition survey (target: nice-project × 1 GPU, need ≥48 GB VRAM)
    partition           status         gpus (free/total)   type                    vram    next free   pending
    ------------------------------------------------------------------------------------------------------
   *nice-project        READY          2/2                 nvidia_l40s             48G     now         0
    rtx_a6000_risk      READY          4/8                 nvidia_rtx_a6000        48G     now         2
    a100                READY          0/4                 nvidia_a100             80G     2h14m       3
    cogvis-project      [not ours]     2/2                 nvidia_rtx_a6000+       48G     now         0

  inferred VRAM need: 48 GB (from 3/4 scorers runnable at --gres=gpu:1; peak = 48 GB because scorers run sequentially)
  skipped at tp>--gres=gpu:1: tower/tower-plus-72b — runner will skip these at load time; re-submit with more GPUs if you need them.

  → nice-project: READY — proceed with pre-flight.
[6/6] sbatch --test-only...    ok: dry-run accepted
dry-run mode: pre-flight OK, not submitting.
```

If the recommender prefers a different partition, you get an interactive prompt:

```
[5/6] cluster probe (live capacity + VRAM fit)...
  ...survey table...
  → a100: CONTESTED — shape fits but 0 free GPUs right now.
  recommender prefers 'nice-project' over 'a100'.
Your target: a100
Recommender's ranking:
  1) nice-project (wait: now; tier=1; free now, group partition)
  2) rtx_a6000_risk (wait: now; tier=2; free now)
  3) a100 (wait: 2h14m; tier=5; no immediate capacity; est. wait 134 min)
Pick 1-3 or c to cancel (15s, no default):
```

Pick `1` / `2` / `3` to re-route, or `c` to cancel. No keypress within 15 s also cancels — there's no silent default.

Escape hatches:

- `--stay-on-target` — keep your original partition even if the probe prefers another. Good for deliberate "I want a100 for headline timing" runs.
- `SUBMIT_AUTO_ROUTE=1 scripts/submit.sh ...` — accept the #1 recommendation without prompting. Use in cron / unattended scripts.
````

- [ ] **Step 2: Add a short section to `docs/AISURREY.md`**

Find the "pre-flight" discussion (the file explains each of the six checks). After the check-5 paragraph, add a subsection:

```markdown
### Queue-aware routing (check 5)

`scripts/cluster_probe.py` pairs `scontrol show node -o` with
`squeue --noheader` to produce a ranked list of partitions. Ranking
combines four signals, in this order:

1. Ready-now (capacity ≥ `--gres=gpu:N` right now).
2. Tier — `nice-project` (1) > 48 GB open partitions (2) > 24 GB (3) >
   `debug`/`2080ti` (4) > `a100` (5). A100 gets a +4 penalty when the
   job fits on 48 GB *and* a non-a100 READY partition is available,
   keeping headline hardware for headline runs.
3. Wait — `deficit`-th-smallest `TimeLeft` across running jobs on that
   partition (where `deficit = gpus_requested - gpus_free + pending_demand`).
4. VRAM waste — smaller is better.
5. Free GPU count — more is better (final tiebreaker).

`cogvis-project` is in `PARTITIONS_BLOCKLIST` (module constant at the
top of `cluster_probe.py`); the recommender never suggests it even
though it has idle A6000s. The partition is still shown in the survey
table, tagged `[not ours]`. Extend the blocklist when another
faculty-specific partition shows up.

If the recommender's top choice is not your `-p`, `submit.sh` prompts
interactively (15 s, no default; pick 1/2/3 or `c` to cancel).
`--stay-on-target` skips the prompt; `SUBMIT_AUTO_ROUTE=1`
auto-accepts #1. See
`docs/superpowers/specs/2026-04-23-queue-aware-cluster-probe-design.md`
for the full design rationale.
```

- [ ] **Step 3: Commit**

```bash
git add examples/04_aisurrey_submission.md docs/AISURREY.md
git commit -m "$(cat <<'EOF'
docs: queue-aware cluster probe + submit.sh prompt

Updates the AISURREY submission example and runbook with the new
survey-table columns (next free / pending), the [not ours] tag,
the interactive prompt, and the --stay-on-target / SUBMIT_AUTO_ROUTE
escape hatches.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Final full regression + smoke**

Run both suites one last time:

```bash
pytest tests/test_cluster_probe.py -v
bash tests/test_submit_prompt.bash
```

Expected: Python suite all pass (~115 tests); bash suite `passed: 5 failed: 0`.

---

## Plan self-review

**Spec coverage:**

- ✅ `squeue` parsing + stdlib-only — Task 2.
- ✅ `PARTITIONS_BLOCKLIST = {"cogvis-project"}` — Task 1.
- ✅ `PARTITION_TIER` with exact tiers from spec — Task 1.
- ✅ Per-partition wait-time estimate with concrete formula — Task 4.
- ✅ Top-3 alternatives — Task 5 (`ranked[:3]`).
- ✅ `--tee-alternatives=<path>` TSV — Task 7.
- ✅ `submit.sh` interactive prompt on exit 2 OR (exit 0 and rec ≠ target) — Task 9.
- ✅ 15 s timeout with no default, user must pick or cancel — Task 9 (`_prompt_alternative` returns 7 on timeout).
- ✅ `SUBMIT_AUTO_ROUTE=1` — Task 9.
- ✅ `--stay-on-target` — Task 9.
- ✅ A100 de-prioritisation with penalty rule — Task 5 (`_a100_penalty_applies`).
- ✅ Error handling: malformed squeue line → None, squeue fail → empty jobs, blocklisted target → exit 5, nothing fits → exit 4 — Tasks 2, 3, 6.
- ✅ Table `next free` + `pending` columns, `[not ours]` tag — Task 8.
- ✅ JSON output adds `alternatives` — Task 8.
- ✅ Graceful degradation when squeue unavailable — Task 7 (empty jobs list is safe) + Task 5 (None wait treated as worst).
- ✅ Testing: every new unit has a unit test; interactive prompt covered by bash harness.
- ✅ Rollout: all new flags/env vars are opt-in; existing `submit.sh <cfg>` unchanged when recommender agrees with target.

**Placeholder scan:** no "TODO", "TBD", "similar to", or "add error handling" found. Every task has concrete code, exact commands, and expected outputs.

**Type consistency:**
- `pick_recommended` returns `tuple[Optional[str], list[Alternative]]` — same shape in Tasks 5, 6, 7, 8.
- `Alternative` fields: `partition`, `wait_s`, `tier`, `vram_waste_gb`, `gpus_free`, `reason` — consistent across Task 5 (definition), Task 7 (TSV writer reads them), Task 8 (JSON serialiser uses `asdict`).
- `Partition` new fields `running_jobs`, `pending_jobs`, `earliest_free_s`, `pending_gpu_demand` — defined Task 3, populated Task 3, read in Tasks 4, 5, 8.
- Exit codes 3 (probe fail), 4 (nothing fits), 5 (blocklisted) — consistent across spec fix, Task 6, Task 9.
- `_prompt_alternative` exit codes: 0 (success) / 7 (cancel) — consistent between definition (Task 9 step 3) and harness test (Task 9 step 1).

No gaps or inconsistencies. The plan is implementation-ready.
