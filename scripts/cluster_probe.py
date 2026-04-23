#!/usr/bin/env python3
"""Cluster probe for mt-metrix SLURM submissions on AISURREY.

Runs BEFORE ``sbatch --test-only`` to answer three questions:

  1. DETECT    — what does the cluster look like right now? Which partitions
                 have how many GPUs free, what type, what VRAM?
  2. COMPREHEND — given the run config, which partitions can actually run
                 this job (shape-wise)? Which fit right now, which only
                 queue?
  3. ADVISE    — print a partition-by-partition table, flag the target
                 partition's status, and if it's contested, recommend a
                 wide-open alternative. Never routes automatically — the
                 user picks.

Why this exists: ``sbatch --test-only`` green-lights only jobs that could
start IMMEDIATELY, and ``sbatch`` (actual submission) on this cluster
ALSO rejects when the partition has zero free GPUs — it does NOT queue
and wait. So before we even try --test-only we need a picture of which
partitions have capacity right now.

Standalone by design: stdlib only, so it can run with the cluster's
system Python BEFORE ``conda activate`` in ``submit.sh``. No mt_metrix
imports, no PyYAML (config is regex-scraped for ``ref:`` entries).

Usage (from submit.sh, automatic):
    python3 scripts/cluster_probe.py --config <yaml> --partition <p> --gpus <N>

Usage (ad-hoc, standalone):
    python3 scripts/cluster_probe.py                 # survey all partitions
    python3 scripts/cluster_probe.py --json          # machine-readable

Exit codes:
    0 = target partition has immediate capacity OR no target given (survey)
    1 = target partition genuinely cannot fit the shape at all (no node in
        the partition has enough GPUs of the right type; user must change -p)
    2 = target partition fits shape-wise but has zero free GPUs right now;
        alternatives exist that fit. submit.sh uses this to warn loudly.
    3 = probe itself failed (scontrol/sinfo unavailable, unparseable output).
        submit.sh should fall through to the existing --test-only path.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# GPU-type → VRAM (GiB) lookup. Substring match is case-insensitive on the
# Gres string. This is the AISURREY fleet as of 2026-04; extend if the
# cluster grows new cards.
#
# ORDER MATTERS — substring match is first-win. More-specific substrings MUST
# come before shorter generic ones (``rtx_5000_ada`` before ``rtx_5000``;
# ``rtx_a5000`` before ``a5000``; etc.), otherwise a 32 GB Ada card would be
# reported as a 16 GB Quadro RTX 5000.
# ---------------------------------------------------------------------------

GPU_VRAM_GB: list[tuple[str, int]] = [
    # 80 GB
    ("a100",            80),
    ("h100",            80),
    # 48 GB
    ("quadro_rtx_8000", 48),
    ("rtx_8000",        48),
    ("rtx8000",         48),
    ("l40s",            48),
    ("rtx_a6000",       48),
    ("a6000",           48),
    # 32 GB — MUST precede ``rtx_5000`` / ``quadro_rtx_5000`` (16 GB)
    ("rtx_5000_ada",    32),
    ("rtx5000_ada",     32),
    # 24 GB
    ("rtx_a5000",       24),
    ("a5000",           24),
    ("rtx_3090",        24),
    ("rtx3090",         24),
    ("geforce_rtx_3090", 24),
    ("3090",            24),
    # 20 GB
    ("rtx_a4500",       20),
    ("a4500",           20),
    # 16 GB — AFTER the 32 GB Ada variants
    ("quadro_rtx_5000", 16),
    ("rtx_5000",        16),
    # 11 GB
    ("rtx_2080",        11),
    ("2080ti",          11),
    ("2080",            11),
]


def vram_for_gpu_type(gpu_type: str) -> int:
    """Return VRAM (GiB) for a Gres GPU type string. 0 if unknown."""
    if not gpu_type:
        return 0
    low = gpu_type.lower()
    for key, gb in GPU_VRAM_GB:
        if key in low:
            return gb
    return 0


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


# ---------------------------------------------------------------------------
# VRAM inference from a run config. The catalogue is rich enough that we
# can't do this with perfect accuracy without importing mt_metrix, but a
# regex pass over ``ref:`` entries gets the right ballpark for every
# scorer in the current catalogue — that's all we need to decide the
# partition.
# ---------------------------------------------------------------------------

# Order matters — first match wins (most specific → least specific).
# Each entry is (pattern, vram_gb, tp, label). ``tp`` is the tensor-parallel
# size the runner actually asks vLLM for (matches scorer.default_tp). When
# a scorer needs tp > requested --gres=gpu:N, the runner skips it at
# load time, so the probe must do the same when inferring peak VRAM —
# otherwise a single 72B entry in a mixed matrix pollutes the whole
# estimate and every partition misfires to NO-FIT.
VRAM_HINTS_GB: list[tuple[re.Pattern[str], int, int, str]] = [
    # 72B Tower on a100 only (tp=4, 144 GB fp16 total).
    (re.compile(r"(?i)72b"),                                  80, 4, "Tower-72B (needs tp=4, a100 only)"),
    # XXL COMET fp32 state dict + 10.7B backbone → 48 GB floor.
    (re.compile(r"(?i)\bxxl\b|cometkiwi-da-xxl|xcomet-xxl"),  48, 1, "COMET-XXL"),
    # 13B Tower = 26 GB fp16 — 48 GB card, tp=1 on a 48 GB.
    (re.compile(r"(?i)13b"),                                  48, 1, "Tower-13B"),
    # 9B Tower = 18 GB fp16 — fits 24 GB but tight; 48 GB comfortable, tp=1.
    (re.compile(r"(?i)\b9b\b|tower-plus-9b"),                 24, 1, "Tower-9B"),
    # 7B Tower, Mistral-7B, COMET-XL (3.5B → 7 GB peak).
    (re.compile(r"(?i)\b7b\b|mistral|\bxl\b"),                24, 1, "Tower-7B / COMET-XL"),
    # 2B Tower, Kiwi-DA base, everything smaller.
    (re.compile(r"(?i)\b2b\b"),                               11, 1, "Tower-2B"),
    # Fallback default (Kiwi-DA, COMET-base, Cometinho, sacrebleu).
    # Anything below matches this fallthrough.
]

DEFAULT_VRAM_GB = 8


@dataclass
class ConfigVRAMNeed:
    """Inferred VRAM envelope for a run config.

    We run scorers SEQUENTIALLY with Scorer.unload() between each (verified
    in the catalogue and unload tests), so the job's real VRAM need is the
    MAX per-scorer peak, not the sum. That's the number we compare against
    each partition's per-node VRAM.

    ``skipped_tp`` reports scorers that WOULD run on more GPUs but will be
    skipped by the runner given the currently requested --gres=gpu:N. Their
    VRAM is NOT counted toward ``max_vram_gb`` — the partition probe should
    target the job that will actually execute, not the one the user wrote
    down. A runnable full matrix minus Tower-72B typically drops peak VRAM
    from 80 → 48 GB, which is the difference between 'a100 only' and
    'nice-project works'.
    """

    max_vram_gb: int
    scorers: list[dict[str, object]] = field(default_factory=list)
    """Per-scorer detail: {ref, rule, vram_gb, tp, skipped}."""
    skipped_tp: list[str] = field(default_factory=list)
    """Scorer refs that won't run because tp exceeds the requested --gres."""


def infer_vram_need(config_text: str, gpus: int = 1) -> ConfigVRAMNeed:
    """Scan a run config's ``ref:`` entries and compute the max VRAM peak.

    ``gpus`` is the ``--gres=gpu:N`` the job would be submitted with.
    Scorers whose inferred ``tp`` exceeds this are recorded in
    ``skipped_tp`` and excluded from ``max_vram_gb`` — the runner skips
    them at load time anyway, so they shouldn't dictate which partition
    the job targets.

    Returns a default of ``DEFAULT_VRAM_GB`` if no runnable scorer matches
    a VRAM hint — those are all small COMET / sacrebleu metrics.
    """
    refs = re.findall(
        r"^\s*-\s*ref:\s*(\S+)", config_text, flags=re.MULTILINE
    )
    max_vram = DEFAULT_VRAM_GB
    scorers: list[dict[str, object]] = []
    skipped_tp: list[str] = []
    for ref in refs:
        vram, tp, rule = DEFAULT_VRAM_GB, 1, "default (≤8 GB)"
        for pat, gb, t, label in VRAM_HINTS_GB:
            if pat.search(ref):
                vram, tp, rule = gb, t, label
                break
        skipped = tp > gpus
        scorers.append({
            "ref": ref,
            "rule": rule,
            "vram_gb": vram,
            "tp": tp,
            "skipped": skipped,
        })
        if skipped:
            skipped_tp.append(ref)
            continue
        if vram > max_vram:
            max_vram = vram
    return ConfigVRAMNeed(
        max_vram_gb=max_vram,
        scorers=scorers,
        skipped_tp=skipped_tp,
    )


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


# ---------------------------------------------------------------------------
# scontrol / sinfo parsing.
# ---------------------------------------------------------------------------

# ``scontrol show node -o`` emits one line per node, keys=space-separated.
# Gres field looks like: Gres=gpu:nvidia_l40s:2(IDX:0-1),gpu:tesla_a100:...
# AllocTRES: cpu=8,mem=64G,gres/gpu=1,gres/gpu:nvidia_l40s=1

_NODE_KV_RE = re.compile(r"(\w[\w/]*?)=(\S+?)(?=\s+\w[\w/]*?=|\s*$)")
_GRES_GPU_RE = re.compile(r"gpu:([a-zA-Z0-9_]+):(\d+)|gpu:(\d+)")
_ALLOC_GRES_GPU_RE = re.compile(r"gres/gpu=(\d+)")
_REALMEM_G_RE = re.compile(r"(\d+)([KMGT]?)")


USABLE_STATES = {
    # States where a node can still schedule new work.
    "IDLE",
    "MIXED",
    "ALLOCATED",
    "COMPLETING",
}
UNUSABLE_STATES = {
    "DOWN",
    "DRAIN",
    "DRAINED",
    "DRAINING",
    "FAIL",
    "FAILING",
    "MAINT",
    "NO_RESPOND",
    "POWER_DOWN",
    "POWERED_DOWN",
    "POWERING_DOWN",
    "POWERING_UP",
    "REBOOT",
    "RESERVED",
    "PLANNED",
}


@dataclass
class Node:
    name: str
    partitions: list[str]
    state: str
    is_usable: bool
    gpu_type: str       # e.g. "nvidia_l40s"; "" if no GPUs configured
    gpu_total: int
    gpu_alloc: int
    realmemory_mb: int
    cpus: int

    @property
    def gpu_free(self) -> int:
        if not self.is_usable:
            return 0
        free = self.gpu_total - self.gpu_alloc
        return max(free, 0)


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


def _normalise_state(raw: str) -> tuple[str, bool]:
    """State field can be ``MIXED+DRAIN`` or ``IDLE``. Return (primary, usable)."""
    if not raw:
        return "", False
    # Drop any trailing '*' (nodes with mismatched scheduler) and split on '+'.
    parts = [p.strip("*+") for p in re.split(r"[+]", raw) if p]
    if not parts:
        return "", False
    primary = parts[0].upper()
    # Any sub-state in UNUSABLE knocks the node out.
    usable = primary in USABLE_STATES and not any(
        p.upper() in UNUSABLE_STATES for p in parts
    )
    return primary, usable


def _parse_realmemory_mb(raw: str) -> int:
    if not raw:
        return 0
    m = _REALMEM_G_RE.match(raw)
    if not m:
        return 0
    n = int(m.group(1))
    unit = (m.group(2) or "").upper()
    if unit == "T":
        return n * 1024 * 1024
    if unit == "G":
        return n * 1024
    if unit == "K":
        return n // 1024
    return n  # bare number = MB


def _parse_gres_gpu(gres: str) -> tuple[str, int]:
    """Return (gpu_type, total_count) from a Gres field.

    If multiple GPU types are configured on the same node (rare here) we
    take the FIRST typed entry. Untyped ``gpu:N`` entries return ("", N).
    """
    if not gres or gres.lower() in {"(null)", "none"}:
        return "", 0
    for m in _GRES_GPU_RE.finditer(gres):
        typ, n_typed, n_bare = m.group(1), m.group(2), m.group(3)
        if typ and n_typed:
            return typ, int(n_typed)
        if n_bare:
            return "", int(n_bare)
    return "", 0


def _parse_alloc_gpu(alloc_tres: str) -> int:
    if not alloc_tres:
        return 0
    m = _ALLOC_GRES_GPU_RE.search(alloc_tres)
    return int(m.group(1)) if m else 0


def parse_scontrol_node_line(line: str) -> Optional[Node]:
    """Parse one ``scontrol show node -o`` line.

    Returns ``None`` if the line doesn't look like a node record (so
    callers can safely iterate over all lines in the output).
    """
    if "NodeName=" not in line:
        return None
    kv = dict(_NODE_KV_RE.findall(line))
    name = kv.get("NodeName", "")
    if not name:
        return None
    state_raw = kv.get("State", "")
    primary_state, usable = _normalise_state(state_raw)
    parts_raw = kv.get("Partitions", "")
    partitions = [p for p in parts_raw.split(",") if p] if parts_raw else []
    gpu_type, gpu_total = _parse_gres_gpu(kv.get("Gres", ""))
    gpu_alloc = _parse_alloc_gpu(kv.get("AllocTRES", ""))
    # CfgTRES takes precedence over RealMemory for schedulable memory,
    # because MemSpecLimit reserves some of the physical RAM for the OS.
    cfg_mem_mb = 0
    cfg_tres = kv.get("CfgTRES", "")
    mem_in_cfg = re.search(r"(?:^|,)mem=(\d+)([KMGT]?)", cfg_tres)
    if mem_in_cfg:
        n = int(mem_in_cfg.group(1))
        unit = (mem_in_cfg.group(2) or "").upper()
        cfg_mem_mb = (
            n * 1024 * 1024 if unit == "T" else
            n * 1024 if unit == "G" else
            n // 1024 if unit == "K" else
            n
        )
    realmemory_mb = cfg_mem_mb or _parse_realmemory_mb(kv.get("RealMemory", ""))
    # Same for CPUs: prefer CfgTRES cpu count (respects CoreSpecCount).
    cfg_cpus = 0
    cpu_in_cfg = re.search(r"(?:^|,)cpu=(\d+)", cfg_tres)
    if cpu_in_cfg:
        cfg_cpus = int(cpu_in_cfg.group(1))
    cpus_raw = kv.get("CPUTot", "") or kv.get("CPUs", "")
    try:
        cpus = cfg_cpus or int(cpus_raw) if cpus_raw else cfg_cpus
    except ValueError:
        cpus = cfg_cpus
    return Node(
        name=name,
        partitions=partitions,
        state=primary_state,
        is_usable=usable,
        gpu_type=gpu_type,
        gpu_total=gpu_total,
        gpu_alloc=gpu_alloc,
        realmemory_mb=realmemory_mb,
        cpus=cpus,
    )


def run_scontrol_show_node() -> Optional[str]:
    """Run ``scontrol show node -o`` and return its stdout.

    Returns ``None`` if scontrol isn't on PATH or returns non-zero —
    caller falls back to sinfo-only output (less detailed).
    """
    if shutil.which("scontrol") is None:
        return None
    try:
        res = subprocess.run(
            ["scontrol", "show", "node", "-o"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout


# ---------------------------------------------------------------------------
# Aggregate per partition.
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    name: str
    nodes_total: int = 0
    nodes_usable: int = 0
    gpus_total: int = 0
    gpus_alloc: int = 0
    gpu_types: list[str] = field(default_factory=list)
    max_vram_gb: int = 0
    # The GPU type that produced ``max_vram_gb``. For mixed-hardware
    # partitions (e.g. cogvis-project has both nvidia_geforce_rtx_3090 and
    # nvidia_rtx_a5000) this is the card we're actually reporting VRAM
    # for — NOT just the first-seen type, which can mislead when the max
    # and the first-seen disagree.
    max_vram_gpu_type: str = ""
    # Types whose VRAM we couldn't identify. Used to flag partitions
    # whose capacity is ambiguous so the user can verify manually rather
    # than inheriting a blanket READY.
    unknown_gpu_types: list[str] = field(default_factory=list)
    max_mem_mb: int = 0
    max_cpus: int = 0
    # Largest GPU count on any single node in the partition — this is the
    # per-node ceiling SLURM will enforce on --gres=gpu:N requests.
    max_gpu_per_node: int = 0
    node_names: list[str] = field(default_factory=list)
    # ---- queue-aware additions (Task 3) ----
    running_jobs: list["Job"] = field(default_factory=list)
    pending_jobs: list["Job"] = field(default_factory=list)
    earliest_free_s: Optional[int] = None
    pending_gpu_demand: int = 0

    @property
    def gpus_free(self) -> int:
        free = self.gpus_total - self.gpus_alloc
        return max(free, 0)

    @property
    def dominant_gpu_type(self) -> str:
        """The GPU type to display alongside ``max_vram_gb``.

        Prefers the type that actually produced ``max_vram_gb`` (so the
        ``type`` and ``vram`` columns in the survey table always
        correlate). Falls back to the first-seen type on partitions where
        NO GPU type is recognised, so the user still sees what's there.
        """
        if self.max_vram_gpu_type:
            return self.max_vram_gpu_type
        return self.gpu_types[0] if self.gpu_types else ""

    @property
    def has_mixed_gpu_types(self) -> bool:
        return len(self.gpu_types) > 1

    @property
    def vram_known(self) -> bool:
        """True iff at least one GPU type on the partition is in our map."""
        return self.max_vram_gb > 0


def aggregate_partitions(nodes: list[Node]) -> dict[str, Partition]:
    """Aggregate nodes into per-partition scheduling view.

    ``nodes_total`` reflects the physical fleet (including DRAIN/DOWN so
    the user sees the node count they expect from ``sinfo``), but
    ``gpus_total`` / ``gpus_alloc`` are schedulable-only: a drained
    node's GPUs can't be scheduled onto, and counting them would make
    gpus_free look good on a partition whose only usable node is full.
    Per-node ceilings (``max_vram_gb``, ``max_mem_mb``, ``max_cpus``) do
    consider every node because a drained node may come back, and the
    ceiling answer doesn't change by the day.
    """
    parts: dict[str, Partition] = {}
    for n in nodes:
        for p in n.partitions:
            part = parts.setdefault(p, Partition(name=p))
            part.nodes_total += 1
            part.node_names.append(n.name)
            if n.is_usable:
                part.nodes_usable += 1
                part.gpus_total += n.gpu_total
                part.gpus_alloc += n.gpu_alloc
            if n.gpu_type and n.gpu_type not in part.gpu_types:
                part.gpu_types.append(n.gpu_type)
            v = vram_for_gpu_type(n.gpu_type)
            if v > part.max_vram_gb:
                part.max_vram_gb = v
                part.max_vram_gpu_type = n.gpu_type
            # Track unknowns separately — helps render_table flag partitions
            # whose VRAM can't be verified (A5000 / Quadro RTX 5000 / etc.
            # were added post-hoc after a real cluster run surfaced the gap;
            # this list is the "what did we miss" signal for next time).
            if n.gpu_type and v == 0 and n.gpu_type not in part.unknown_gpu_types:
                part.unknown_gpu_types.append(n.gpu_type)
            if n.realmemory_mb > part.max_mem_mb:
                part.max_mem_mb = n.realmemory_mb
            if n.cpus > part.max_cpus:
                part.max_cpus = n.cpus
            if n.gpu_total > part.max_gpu_per_node:
                part.max_gpu_per_node = n.gpu_total
    return parts


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


# ---------------------------------------------------------------------------
# Request parsing — read the requested partition / gpu count / mem from
# the slurm script header + any CLI overrides passed through.
# ---------------------------------------------------------------------------

_SBATCH_DEFAULT_PARTITION = re.compile(
    r"^#SBATCH\s+--partition=(\S+)", re.MULTILINE
)
_SBATCH_DEFAULT_GRES = re.compile(
    r"^#SBATCH\s+--gres=gpu:(\S+)", re.MULTILINE
)
_SBATCH_DEFAULT_MEM = re.compile(r"^#SBATCH\s+--mem=(\S+)", re.MULTILINE)
_SBATCH_DEFAULT_CPUS = re.compile(
    r"^#SBATCH\s+--cpus-per-task=(\d+)", re.MULTILINE
)


@dataclass
class Request:
    partition: str
    gpus: int
    mem_mb: int
    cpus: int
    vram_need_gb: int


def parse_slurm_header(script_text: str) -> dict[str, str]:
    defaults: dict[str, str] = {}
    if m := _SBATCH_DEFAULT_PARTITION.search(script_text):
        defaults["partition"] = m.group(1)
    if m := _SBATCH_DEFAULT_GRES.search(script_text):
        g = m.group(1)
        defaults["gpus"] = g.rsplit(":", 1)[-1]
    if m := _SBATCH_DEFAULT_MEM.search(script_text):
        defaults["mem"] = m.group(1)
    if m := _SBATCH_DEFAULT_CPUS.search(script_text):
        defaults["cpus"] = m.group(1)
    return defaults


def parse_mem_mb(raw: str) -> int:
    """SLURM --mem value (e.g. 120G, 64000, 1T) → megabytes. 0 if unparseable."""
    if not raw:
        return 0
    m = re.match(r"^(\d+)([KMGT]?)$", raw)
    if not m:
        return 0
    n = int(m.group(1))
    unit = (m.group(2) or "").upper()
    if unit == "T":
        return n * 1024 * 1024
    if unit == "G":
        return n * 1024
    if unit == "K":
        return n // 1024
    return n  # bare number = MB in SLURM


# ---------------------------------------------------------------------------
# Fit check + advice.
# ---------------------------------------------------------------------------

@dataclass
class FitStatus:
    """Can a partition run this request?

    - ``shape_ok``: the partition's LARGEST node has enough GPUs, VRAM,
      mem, CPUs. If False, this partition can never run the job no
      matter how long we wait.
    - ``has_capacity_now``: at least ``request.gpus`` GPUs are free on
      some usable node in the partition RIGHT NOW.
    - ``vram_known``: we have VRAM data for at least one GPU type on the
      partition. When False, the partition gets UNKNOWN-VRAM status in
      the survey table and is not recommended as an alternative — we
      can't stand behind it without verifying.
    """

    partition: str
    shape_ok: bool
    has_capacity_now: bool
    vram_known: bool = True
    reasons: list[str] = field(default_factory=list)

    def status_word(self) -> str:
        if not self.shape_ok:
            return "no-fit"
        if not self.has_capacity_now:
            return "contested"
        if not self.vram_known:
            return "unknown-vram"
        return "ready"


def check_fit(part: Partition, req: Request) -> FitStatus:
    reasons: list[str] = []
    shape_ok = True
    # Shape: partition's largest node must have ≥ request in all dims.
    # For GPUs, use max_gpu_per_node (populated during aggregation) —
    # SLURM enforces --gres=gpu:N per-node, not fleet-wide, so the
    # biggest single node is the true ceiling.
    if part.nodes_total == 0:
        shape_ok = False
        reasons.append("partition has no nodes")
    elif req.gpus > part.max_gpu_per_node:
        shape_ok = False
        reasons.append(
            f"--gres=gpu:{req.gpus} exceeds per-node GPU ceiling "
            f"({part.max_gpu_per_node})"
        )
    if req.vram_need_gb and part.max_vram_gb \
            and req.vram_need_gb > part.max_vram_gb:
        shape_ok = False
        reasons.append(
            f"job needs ≥{req.vram_need_gb} GB VRAM, partition tops out at "
            f"{part.max_vram_gb} GB ({part.dominant_gpu_type or 'unknown'})"
        )
    if req.mem_mb and part.max_mem_mb and req.mem_mb > part.max_mem_mb:
        shape_ok = False
        reasons.append(
            f"--mem={req.mem_mb}M exceeds largest node RealMemory "
            f"={part.max_mem_mb}M"
        )
    if req.cpus and part.max_cpus and req.cpus > part.max_cpus:
        shape_ok = False
        reasons.append(
            f"--cpus-per-task={req.cpus} exceeds largest node CPUs "
            f"={part.max_cpus}"
        )
    # Capacity right now: at least ``req.gpus`` free on some usable node?
    # Approximate using partition-wide free count (good enough; submit
    # attempts a real fit at sbatch time anyway).
    has_capacity_now = shape_ok and part.gpus_free >= req.gpus
    # VRAM verifiable iff we know VRAM for at least one type AND there's
    # no unknown type that could under/over-shoot the reported max. If
    # some nodes have unknown VRAM, we can't promise the max is real.
    vram_known = part.vram_known and not part.unknown_gpu_types
    return FitStatus(
        partition=part.name,
        shape_ok=shape_ok,
        has_capacity_now=has_capacity_now,
        vram_known=vram_known,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def _ansi(enable: bool) -> dict[str, str]:
    if not enable:
        return {k: "" for k in ("red", "yel", "grn", "dim", "bold", "reset")}
    return {
        "red":   "\033[31m",
        "yel":   "\033[33m",
        "grn":   "\033[32m",
        "dim":   "\033[2m",
        "bold":  "\033[1m",
        "reset": "\033[0m",
    }


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
        f"{'type':<22}  {'vram':<6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    # Sort: target first, then ready > contested > unknown-vram > no-fit,
    # then by free GPUs desc.
    def sort_key(p: Partition) -> tuple[int, int, int]:
        fit = fits.get(p.name)
        if p.name == req.partition:
            return (0, 0, 0)
        if fit is None:
            return (4, 0, 0)
        rank = {
            "ready": 1,
            "contested": 2,
            "unknown-vram": 3,
            "no-fit": 4,
        }[fit.status_word()]
        return (rank, -p.gpus_free, 0)
    has_mixed_any = False
    for p in sorted(partitions, key=sort_key):
        fit = fits.get(p.name)
        status = fit.status_word() if fit else "?"
        if status == "ready":
            colour_code = c["grn"]
            tag = "READY"
        elif status == "contested":
            colour_code = c["yel"]
            tag = "CONTESTED"
        elif status == "unknown-vram":
            colour_code = c["yel"]
            tag = "UNKNOWN-VRAM"
        elif status == "no-fit":
            colour_code = c["red"]
            tag = "NO-FIT"
        else:
            colour_code = c["dim"]
            tag = "?"
        mark = " *" if p.name == req.partition else "  "
        gpu_cell = f"{p.gpus_free}/{p.gpus_total}"
        gpu_type = p.dominant_gpu_type or "—"
        # Suffix with '+' to signal mixed-hardware partition (the displayed
        # type is the one that produced max_vram_gb; other node types exist).
        if p.has_mixed_gpu_types:
            gpu_type = f"{gpu_type}+"
            has_mixed_any = True
        vram = f"{p.max_vram_gb}G" if p.max_vram_gb else "—"
        lines.append(
            f"{mark}{p.name:<18}  {colour_code}{tag:<13}{c['reset']}  "
            f"{gpu_cell:<18}  {gpu_type:<22}  {vram:<6}"
        )
    if has_mixed_any:
        lines.append(
            f"  {c['dim']}'+' = mixed-hardware partition (vram/type shown "
            f"for the fattest card){c['reset']}"
        )
    return "\n".join(lines)


def pick_recommended(
    partitions: list[Partition],
    fits: dict[str, FitStatus],
    req: Request,
) -> Optional[str]:
    """If the target is contested or no-fit, suggest a ready alternative.

    Prefers partitions with the SMALLEST fitting VRAM — don't send a
    Kiwi-DA job to a100 when nice-project has capacity. Among ties
    prefers the one with MORE free GPUs. Partitions with UNKNOWN-VRAM
    status are NOT recommended: we can't promise they fit without
    verifying, and a blind recommendation is worse than none.
    """
    target_fit = fits.get(req.partition)
    if target_fit and target_fit.has_capacity_now and target_fit.vram_known:
        return None
    candidates = [
        p for p in partitions
        if p.name != req.partition
        and fits[p.name].has_capacity_now
        and fits[p.name].vram_known
        and (not req.vram_need_gb or p.max_vram_gb >= req.vram_need_gb)
    ]
    if not candidates:
        return None
    # Right-size: pick the smallest VRAM that still fits.
    candidates.sort(key=lambda p: (p.max_vram_gb, -p.gpus_free))
    return candidates[0].name


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def build_request(
    args: argparse.Namespace,
    slurm_defaults: dict[str, str],
    vram_need_gb: int,
) -> Request:
    partition = args.partition or slurm_defaults.get("partition", "nice-project")
    gpus = args.gpus or int(slurm_defaults.get("gpus", "1"))
    mem_mb = parse_mem_mb(args.mem or slurm_defaults.get("mem", ""))
    cpus = args.cpus or int(slurm_defaults.get("cpus", "0") or "0")
    return Request(
        partition=partition,
        gpus=gpus,
        mem_mb=mem_mb,
        cpus=cpus,
        vram_need_gb=vram_need_gb,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                   help="run config YAML to infer VRAM need from")
    p.add_argument("--partition", type=str, default=None,
                   help="target partition (default: parse from slurm header)")
    p.add_argument("--gpus", type=int, default=None,
                   help="requested GPU count (default: from slurm header)")
    p.add_argument("--mem", type=str, default=None,
                   help="requested --mem (e.g. 120G); default from slurm header")
    p.add_argument("--cpus", type=int, default=None,
                   help="requested --cpus-per-task; default from slurm header")
    p.add_argument("--slurm-script", type=Path,
                   default=Path("scripts/run_mt_metrix.slurm"),
                   help="slurm script for #SBATCH defaults")
    p.add_argument("--json", action="store_true",
                   help="emit machine-readable JSON")
    p.add_argument("--no-colour", action="store_true",
                   help="disable ANSI colour in the table")
    args = p.parse_args(argv)

    # Slurm-script defaults — parsed FIRST because the requested --gres=gpu:N
    # drives the tp-based scorer filter in infer_vram_need below. Without
    # this the probe over-estimates VRAM need for any full-matrix config
    # that includes tp>1 models (Tower-72B), and routes every submission
    # to a100-only partitions.
    slurm_defaults: dict[str, str] = {}
    if args.slurm_script and args.slurm_script.is_file():
        slurm_defaults = parse_slurm_header(args.slurm_script.read_text())
    gpus_for_inference = args.gpus or int(slurm_defaults.get("gpus", "1"))

    # VRAM need from config.
    vram_need_gb = DEFAULT_VRAM_GB
    vram_detail: Optional[ConfigVRAMNeed] = None
    if args.config and args.config.is_file():
        vram_detail = infer_vram_need(
            args.config.read_text(), gpus=gpus_for_inference,
        )
        vram_need_gb = vram_detail.max_vram_gb

    req = build_request(args, slurm_defaults, vram_need_gb)

    # Probe.
    raw = run_scontrol_show_node()
    if raw is None:
        msg = {
            "error": "scontrol unavailable or failed",
            "hint": "run this on aisurrey-submit01; submit.sh falls back to --test-only",
        }
        if args.json:
            print(json.dumps(msg, indent=2))
        else:
            print(
                f"cluster_probe: {msg['error']} — {msg['hint']}",
                file=sys.stderr,
            )
        return 3

    nodes = [
        n for line in raw.splitlines()
        if (n := parse_scontrol_node_line(line)) is not None
    ]
    if not nodes:
        if args.json:
            print(json.dumps({"error": "no nodes parsed from scontrol"}))
        else:
            print("cluster_probe: scontrol returned no node records",
                  file=sys.stderr)
        return 3
    partitions_map = aggregate_partitions(nodes)
    partitions = sorted(partitions_map.values(), key=lambda p: p.name)
    fits = {p.name: check_fit(p, req) for p in partitions}

    # Target may not even be a known partition.
    target_fit = fits.get(req.partition)
    if target_fit is None:
        if args.json:
            print(json.dumps({
                "error": f"partition '{req.partition}' not found",
                "available": sorted(partitions_map.keys()),
            }, indent=2))
        else:
            print(
                f"cluster_probe: partition '{req.partition}' not found. "
                f"Known: {', '.join(sorted(partitions_map.keys()))}",
                file=sys.stderr,
            )
        return 1

    recommended = pick_recommended(partitions, fits, req)

    if args.json:
        payload = {
            "request": asdict(req),
            "target_fit": asdict(target_fit),
            "partitions": [asdict(p) | {"gpus_free": p.gpus_free} for p in partitions],
            "recommendation": recommended,
            "vram_inference": asdict(vram_detail) if vram_detail else None,
        }
        print(json.dumps(payload, indent=2))
    else:
        use_colour = (not args.no_colour) and sys.stdout.isatty()
        print(render_table(partitions, fits, req, colour=use_colour))
        if vram_detail:
            runnable = [s for s in vram_detail.scorers if not s["skipped"]]
            print()
            print(
                f"  inferred VRAM need: {vram_need_gb} GB "
                f"(from {len(runnable)}/{len(vram_detail.scorers)} scorers "
                f"runnable at --gres=gpu:{req.gpus}; peak = {vram_need_gb} GB "
                f"because scorers run sequentially)"
            )
            if vram_detail.skipped_tp:
                skipped_refs = ", ".join(vram_detail.skipped_tp)
                print(
                    f"  skipped at tp>--gres=gpu:{req.gpus}: {skipped_refs} — "
                    f"runner will skip these at load time; re-submit with more "
                    f"GPUs if you need them."
                )
        # Flag partitions whose VRAM we couldn't verify so the user can
        # check the GPU type manually rather than trusting a blank READY.
        unknown_parts = [p for p in partitions if p.unknown_gpu_types]
        if unknown_parts:
            print()
            for p in unknown_parts:
                types = ", ".join(p.unknown_gpu_types)
                print(
                    f"  note: {p.name} has unmapped GPU type(s): {types} — "
                    f"verify VRAM with `scontrol show node <node>` before "
                    f"submitting here."
                )
        print()
        tf = target_fit
        if not tf.shape_ok:
            print(f"  → {req.partition}: NO-FIT — " + "; ".join(tf.reasons))
            if recommended:
                print(f"    Try: scripts/submit.sh <config> -p {recommended}")
        elif not tf.has_capacity_now:
            print(f"  → {req.partition}: CONTESTED — shape fits but 0 "
                  f"free GPUs right now.")
            if recommended:
                print(f"    Immediately available alternative: "
                      f"scripts/submit.sh <config> -p {recommended}")
            else:
                print("    No ready alternative; sbatch would reject too. "
                      "Wait for the queue to drain.")
        elif not tf.vram_known:
            print(
                f"  → {req.partition}: UNKNOWN-VRAM — partition has GPU "
                f"types we don't have a VRAM entry for. shape-ok, GPUs "
                f"free, but we can't promise the job fits."
            )
            if recommended:
                print(f"    Safer alternative: "
                      f"scripts/submit.sh <config> -p {recommended}")
        else:
            print(f"  → {req.partition}: READY — proceed with pre-flight.")

    # Exit code policy.
    if not target_fit.shape_ok:
        return 1
    if not target_fit.has_capacity_now:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
