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
# ---------------------------------------------------------------------------

GPU_VRAM_GB: list[tuple[str, int]] = [
    ("a100",          80),
    ("h100",          80),
    ("l40s",          48),
    ("a6000",         48),
    ("rtx8000",       48),
    ("quadro_rtx_8000", 48),
    ("3090",          24),
    ("rtx_3090",      24),
    ("rtx3090",       24),
    ("2080",          11),
    ("rtx_2080",      11),
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
# VRAM inference from a run config. The catalogue is rich enough that we
# can't do this with perfect accuracy without importing mt_metrix, but a
# regex pass over ``ref:`` entries gets the right ballpark for every
# scorer in the current catalogue — that's all we need to decide the
# partition.
# ---------------------------------------------------------------------------

# Order matters — first match wins (most specific → least specific).
VRAM_HINTS_GB: list[tuple[re.Pattern[str], int, str]] = [
    # 72B Tower on a100 only (tp=4, 144 GB fp16 total).
    (re.compile(r"(?i)72b"),                 80, "Tower-72B (tp=4 a100 only)"),
    # XXL COMET fp32 state dict + 10.7B backbone → 48 GB floor.
    (re.compile(r"(?i)\bxxl\b|cometkiwi-da-xxl|xcomet-xxl"), 48, "COMET-XXL"),
    # 13B Tower = 26 GB fp16 — 48 GB card.
    (re.compile(r"(?i)13b"),                 48, "Tower-13B"),
    # 9B Tower = 18 GB fp16 — fits 24 GB but tight; 48 GB comfortable.
    (re.compile(r"(?i)\b9b\b|tower-plus-9b"), 24, "Tower-9B"),
    # 7B Tower, Mistral-7B, COMET-XL (3.5B → 7 GB peak).
    (re.compile(r"(?i)\b7b\b|mistral|\bxl\b"), 24, "Tower-7B / COMET-XL"),
    # 2B Tower, Kiwi-DA base, everything smaller.
    (re.compile(r"(?i)\b2b\b"),              11, "Tower-2B"),
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
    """

    max_vram_gb: int
    scorers: list[dict[str, object]] = field(default_factory=list)
    """Per-scorer detail: {ref, matched_rule, vram_gb}."""


def infer_vram_need(config_text: str) -> ConfigVRAMNeed:
    """Scan a run config's ``ref:`` entries and compute the max VRAM peak.

    Returns a default of ``DEFAULT_VRAM_GB`` if no scorer matches a VRAM
    hint — those are all small COMET / sacrebleu metrics.
    """
    refs = re.findall(
        r"^\s*-\s*ref:\s*(\S+)", config_text, flags=re.MULTILINE
    )
    max_vram = DEFAULT_VRAM_GB
    scorers: list[dict[str, object]] = []
    for ref in refs:
        vram, rule = DEFAULT_VRAM_GB, "default (≤8 GB)"
        for pat, gb, label in VRAM_HINTS_GB:
            if pat.search(ref):
                vram, rule = gb, label
                break
        scorers.append({"ref": ref, "rule": rule, "vram_gb": vram})
        if vram > max_vram:
            max_vram = vram
    return ConfigVRAMNeed(max_vram_gb=max_vram, scorers=scorers)


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
    max_mem_mb: int = 0
    max_cpus: int = 0
    # Largest GPU count on any single node in the partition — this is the
    # per-node ceiling SLURM will enforce on --gres=gpu:N requests.
    max_gpu_per_node: int = 0
    node_names: list[str] = field(default_factory=list)

    @property
    def gpus_free(self) -> int:
        free = self.gpus_total - self.gpus_alloc
        return max(free, 0)

    @property
    def dominant_gpu_type(self) -> str:
        return self.gpu_types[0] if self.gpu_types else ""


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
            if n.realmemory_mb > part.max_mem_mb:
                part.max_mem_mb = n.realmemory_mb
            if n.cpus > part.max_cpus:
                part.max_cpus = n.cpus
            if n.gpu_total > part.max_gpu_per_node:
                part.max_gpu_per_node = n.gpu_total
    return parts


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
    """

    partition: str
    shape_ok: bool
    has_capacity_now: bool
    reasons: list[str] = field(default_factory=list)

    def status_word(self) -> str:
        if not self.shape_ok:
            return "no-fit"
        if self.has_capacity_now:
            return "ready"
        return "contested"


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
    return FitStatus(
        partition=part.name,
        shape_ok=shape_ok,
        has_capacity_now=has_capacity_now,
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
        f"  {'partition':<18}  {'status':<10}  {'gpus (free/total)':<18}  "
        f"{'type':<18}  {'vram':<6}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    # Sort: target first, then ready > contested > no-fit, then by free GPUs desc.
    def sort_key(p: Partition) -> tuple[int, int, int]:
        fit = fits.get(p.name)
        if p.name == req.partition:
            return (0, 0, 0)
        if fit is None:
            return (3, 0, 0)
        rank = {"ready": 1, "contested": 2, "no-fit": 3}[fit.status_word()]
        return (rank, -p.gpus_free, 0)
    for p in sorted(partitions, key=sort_key):
        fit = fits.get(p.name)
        status = fit.status_word() if fit else "?"
        if status == "ready":
            colour_code = c["grn"]
            tag = "READY"
        elif status == "contested":
            colour_code = c["yel"]
            tag = "CONTESTED"
        elif status == "no-fit":
            colour_code = c["red"]
            tag = "NO-FIT"
        else:
            colour_code = c["dim"]
            tag = "?"
        mark = " *" if p.name == req.partition else "  "
        gpu_cell = f"{p.gpus_free}/{p.gpus_total}"
        gpu_type = p.dominant_gpu_type or "—"
        vram = f"{p.max_vram_gb}G" if p.max_vram_gb else "—"
        lines.append(
            f"{mark}{p.name:<18}  {colour_code}{tag:<10}{c['reset']}  "
            f"{gpu_cell:<18}  {gpu_type:<18}  {vram:<6}"
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
    prefers the one with MORE free GPUs.
    """
    target_fit = fits.get(req.partition)
    if target_fit and target_fit.has_capacity_now:
        return None
    candidates = [
        p for p in partitions
        if p.name != req.partition
        and fits[p.name].has_capacity_now
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

    # VRAM need from config.
    vram_need_gb = DEFAULT_VRAM_GB
    vram_detail: Optional[ConfigVRAMNeed] = None
    if args.config and args.config.is_file():
        vram_detail = infer_vram_need(args.config.read_text())
        vram_need_gb = vram_detail.max_vram_gb

    # Slurm-script defaults.
    slurm_defaults: dict[str, str] = {}
    if args.slurm_script and args.slurm_script.is_file():
        slurm_defaults = parse_slurm_header(args.slurm_script.read_text())

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
            print()
            print(f"  inferred VRAM need: {vram_need_gb} GB "
                  f"(from {len(vram_detail.scorers)} scorers; "
                  f"peak = {vram_need_gb} GB because scorers run sequentially)")
        print()
        tf = target_fit
        if tf.has_capacity_now:
            print(f"  → {req.partition}: READY — {sum(1 for _ in [0])} "
                  f"proceed with pre-flight.")
        elif not tf.shape_ok:
            print(f"  → {req.partition}: NO-FIT — " + "; ".join(tf.reasons))
            if recommended:
                print(f"    Try: scripts/submit.sh <config> -p {recommended}")
        else:
            print(f"  → {req.partition}: CONTESTED — shape fits but 0 "
                  f"free GPUs right now.")
            if recommended:
                print(f"    Immediately available alternative: "
                      f"scripts/submit.sh <config> -p {recommended}")
            else:
                print("    No ready alternative; sbatch would reject too. "
                      "Wait for the queue to drain.")

    # Exit code policy.
    if not target_fit.shape_ok:
        return 1
    if not target_fit.has_capacity_now:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
