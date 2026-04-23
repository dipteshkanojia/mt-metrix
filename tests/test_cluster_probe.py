"""Unit tests for scripts/cluster_probe.py.

The probe is stdlib-only by design (it runs before ``conda activate`` in
submit.sh), so these tests drive it with synthetic ``scontrol show node
-o`` output rather than a live cluster. The fixture lines mirror the
real AISURREY topology observed in ``scontrol show node aisurrey35`` as
of 2026-04:

    nice-project → aisurrey35 (2× L40s 48 GB, CfgTRES cpu=14 mem=125G gpu=2)
    a100         → aisurreyN (1× A100 80 GB)
    3090         → aisurreyM (1× RTX3090 24 GB)
    drained      → a dead node (verifies we exclude it from capacity sums)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


# The probe is a standalone CLI script, not an importable package. Load it
# as a module once per session so individual tests can exercise its
# internals without shelling out.
REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "cluster_probe.py"


@pytest.fixture(scope="session")
def probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cluster_probe", PROBE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cluster_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic scontrol output. One line per node, matching the real shape.
# ---------------------------------------------------------------------------

SCONTROL_FIXTURE = "\n".join([
    # nice-project: single node, 2× L40s, 1 GPU allocated right now
    "NodeName=aisurrey35 Arch=x86_64 CoreSpecCount=2 CoresPerSocket=8 CPUAlloc=8 CPUTot=16 "
    "RealMemory=128000 MemSpecLimit=8000 "
    "State=MIXED ThreadsPerCore=1 "
    "Gres=gpu:nvidia_l40s:2(IDX:0-1) "
    "Partitions=nice-project "
    "AllocTRES=cpu=8,mem=64G,gres/gpu=1,gres/gpu:nvidia_l40s=1 "
    "CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2",

    # a100: 1 node, 4× A100 80 GB, idle (need 4 so 72B tp=4 has a home in
    # the fit checks below; the prior fixture had 1 which wouldn't admit
    # any tp>1 workload)
    "NodeName=aisurrey01 Arch=x86_64 CoresPerSocket=8 CPUTot=32 "
    "RealMemory=512000 "
    "State=IDLE "
    "Gres=gpu:nvidia_a100:4(IDX:0-3) "
    "Partitions=a100 "
    "AllocTRES= "
    "CfgTRES=cpu=32,mem=500G,gres/gpu=4,gres/gpu:nvidia_a100=4",

    # 3090: 1 node, 1× RTX3090 24 GB, idle
    "NodeName=aisurrey10 Arch=x86_64 CPUTot=16 "
    "RealMemory=64000 "
    "State=IDLE "
    "Gres=gpu:nvidia_rtx_3090:1(IDX:0) "
    "Partitions=3090,3090_risk "
    "AllocTRES= "
    "CfgTRES=cpu=16,mem=62G,gres/gpu=1,gres/gpu:nvidia_rtx_3090=1",

    # rtx_a6000_risk: 1 node, 1× A6000 48 GB, fully allocated (contested)
    "NodeName=aisurrey20 Arch=x86_64 CPUTot=24 "
    "RealMemory=256000 "
    "State=ALLOCATED "
    "Gres=gpu:nvidia_a6000:1(IDX:0) "
    "Partitions=rtx_a6000_risk "
    "AllocTRES=cpu=8,mem=128G,gres/gpu=1,gres/gpu:nvidia_a6000=1 "
    "CfgTRES=cpu=24,mem=252G,gres/gpu=1,gres/gpu:nvidia_a6000=1",

    # drained node: must be excluded from capacity
    "NodeName=aisurrey26 Arch=x86_64 CPUTot=16 "
    "RealMemory=128000 "
    "State=DRAIN+DOWN "
    "Gres=gpu:nvidia_l40s:2 "
    "Partitions=l40s_risk "
    "AllocTRES= "
    "CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2",

    # A usable l40s_risk node so l40s_risk isn't empty after excluding aisurrey26
    "NodeName=aisurrey27 Arch=x86_64 CPUTot=16 "
    "RealMemory=128000 "
    "State=IDLE "
    "Gres=gpu:nvidia_l40s:2(IDX:0-1) "
    "Partitions=l40s_risk "
    "AllocTRES= "
    "CfgTRES=cpu=14,mem=125G,gres/gpu=2,gres/gpu:nvidia_l40s=2",

    # cogvis-project: mixed-hardware partition. One 3090 node (24 GB) and
    # one A6000 node (48 GB) — the display must show the A6000 alongside
    # the 48 GB max, NOT the first-seen 3090. Mirrors the real cluster
    # bug the probe uncovered on 2026-04-23.
    "NodeName=aisurrey50 Arch=x86_64 CPUTot=16 "
    "RealMemory=128000 "
    "State=MIXED "
    "Gres=gpu:nvidia_geforce_rtx_3090:4(IDX:0-3) "
    "Partitions=cogvis-project "
    "AllocTRES=cpu=4,mem=32G,gres/gpu=1,gres/gpu:nvidia_geforce_rtx_3090=1 "
    "CfgTRES=cpu=16,mem=125G,gres/gpu=4,gres/gpu:nvidia_geforce_rtx_3090=4",

    "NodeName=aisurrey51 Arch=x86_64 CPUTot=32 "
    "RealMemory=256000 "
    "State=IDLE "
    "Gres=gpu:nvidia_rtx_a6000:2(IDX:0-1) "
    "Partitions=cogvis-project "
    "AllocTRES= "
    "CfgTRES=cpu=32,mem=252G,gres/gpu=2,gres/gpu:nvidia_rtx_a6000=2",

    # narrative-project: single node, 2× RTX 5000 Ada (32 GB each), idle —
    # exercises the 32 GB Ada entry we added post-hoc.
    "NodeName=aisurrey52 Arch=x86_64 CPUTot=16 "
    "RealMemory=128000 "
    "State=IDLE "
    "Gres=gpu:nvidia_rtx_5000_ada:2(IDX:0-1) "
    "Partitions=narrative-project "
    "AllocTRES= "
    "CfgTRES=cpu=16,mem=125G,gres/gpu=2,gres/gpu:nvidia_rtx_5000_ada=2",

    # unknown-vram partition: GPU type not in our map. Must surface as
    # UNKNOWN-VRAM in the survey so the user knows to verify manually.
    "NodeName=aisurrey99 Arch=x86_64 CPUTot=16 "
    "RealMemory=128000 "
    "State=IDLE "
    "Gres=gpu:nvidia_future_card:2(IDX:0-1) "
    "Partitions=experimental "
    "AllocTRES= "
    "CfgTRES=cpu=16,mem=125G,gres/gpu=2,gres/gpu:nvidia_future_card=2",
])


# ---------------------------------------------------------------------------
# GPU type → VRAM mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gpu_type,expected",
    [
        # 80 GB
        ("nvidia_a100", 80),
        ("NVIDIA_A100_80GB_PCIe", 80),
        ("nvidia_h100", 80),
        # 48 GB
        ("nvidia_l40s", 48),
        ("nvidia_a6000", 48),
        ("nvidia_rtx_a6000", 48),
        ("rtx8000", 48),
        ("quadro_rtx_8000", 48),
        # 32 GB — RTX 5000 Ada (narrative-project on AISURREY)
        ("nvidia_rtx_5000_ada", 32),
        ("rtx_5000_ada_generation", 32),
        # 24 GB — RTX A5000 (debug on AISURREY) and 3090 variants
        ("nvidia_rtx_a5000", 24),
        ("nvidia_a5000", 24),
        ("nvidia_rtx_3090", 24),
        ("nvidia_geforce_rtx_3090", 24),
        # 20 GB — defensive A4500 entry
        ("nvidia_rtx_a4500", 20),
        # 16 GB — Quadro RTX 5000 (rtx5000 on AISURREY); MUST NOT collide
        # with the 32 GB Ada variant above
        ("quadro_rtx_5000", 16),
        ("nvidia_quadro_rtx_5000", 16),
        # 11 GB
        ("nvidia_rtx_2080ti", 11),
        # Unknown
        ("", 0),
        ("unknown_card", 0),
        ("nvidia_future_card", 0),
    ],
)
def test_vram_for_gpu_type(probe, gpu_type, expected):
    assert probe.vram_for_gpu_type(gpu_type) == expected


def test_vram_for_gpu_type_ada_does_not_shadow_quadro(probe):
    """Regression: if ``rtx_5000`` came before ``rtx_5000_ada`` in the map,
    a 32 GB Ada card would be reported as 16 GB. Lock that in."""
    assert probe.vram_for_gpu_type("nvidia_rtx_5000_ada") == 32
    assert probe.vram_for_gpu_type("quadro_rtx_5000") == 16


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


# ---------------------------------------------------------------------------
# Gres / AllocTRES / State parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gres,expected",
    [
        ("gpu:nvidia_l40s:2(IDX:0-1)", ("nvidia_l40s", 2)),
        ("gpu:nvidia_a100:4", ("nvidia_a100", 4)),
        ("gpu:2", ("", 2)),
        ("(null)", ("", 0)),
        ("", ("", 0)),
    ],
)
def test_parse_gres_gpu(probe, gres, expected):
    assert probe._parse_gres_gpu(gres) == expected


@pytest.mark.parametrize(
    "alloc,expected",
    [
        ("cpu=8,mem=64G,gres/gpu=1,gres/gpu:nvidia_l40s=1", 1),
        ("cpu=16,mem=128G,gres/gpu=4,gres/gpu:nvidia_a100=4", 4),
        ("cpu=4,mem=32G", 0),
        ("", 0),
    ],
)
def test_parse_alloc_gpu(probe, alloc, expected):
    assert probe._parse_alloc_gpu(alloc) == expected


@pytest.mark.parametrize(
    "raw,expected_primary,expected_usable",
    [
        ("IDLE", "IDLE", True),
        ("MIXED", "MIXED", True),
        ("ALLOCATED", "ALLOCATED", True),
        ("MIXED+DRAIN", "MIXED", False),   # drain sub-state kills usable
        ("IDLE*", "IDLE", True),            # trailing star (asterisk) OK
        ("DOWN", "DOWN", False),
        ("DRAIN", "DRAIN", False),
        ("", "", False),
    ],
)
def test_normalise_state(probe, raw, expected_primary, expected_usable):
    primary, usable = probe._normalise_state(raw)
    assert primary == expected_primary
    assert usable is expected_usable


@pytest.mark.parametrize(
    "raw,expected_mb",
    [
        ("128000", 128000),
        ("125G", 128000),
        ("1T", 1024 * 1024),
        ("512M", 512),
        ("", 0),
    ],
)
def test_parse_realmemory_mb(probe, raw, expected_mb):
    assert probe._parse_realmemory_mb(raw) == expected_mb


@pytest.mark.parametrize(
    "raw,expected_mb",
    [
        ("120G", 120 * 1024),
        ("256G", 256 * 1024),
        ("64000", 64000),
        ("1T", 1024 * 1024),
        ("", 0),
        ("garbage", 0),
    ],
)
def test_parse_mem_mb_slurm_cli(probe, raw, expected_mb):
    assert probe.parse_mem_mb(raw) == expected_mb


# ---------------------------------------------------------------------------
# Full scontrol line → Node
# ---------------------------------------------------------------------------

def test_parse_scontrol_node_line_nice_project(probe):
    line = SCONTROL_FIXTURE.splitlines()[0]
    n = probe.parse_scontrol_node_line(line)
    assert n is not None
    assert n.name == "aisurrey35"
    assert n.partitions == ["nice-project"]
    assert n.state == "MIXED"
    assert n.is_usable is True
    assert n.gpu_type == "nvidia_l40s"
    assert n.gpu_total == 2
    assert n.gpu_alloc == 1
    # CfgTRES mem=125G takes precedence over RealMemory=128000 M.
    assert n.realmemory_mb == 125 * 1024
    # CfgTRES cpu=14 takes precedence over CPUTot=16 (respects CoreSpecCount=2).
    assert n.cpus == 14


def test_parse_scontrol_node_line_drained(probe):
    drained_line = [l for l in SCONTROL_FIXTURE.splitlines() if "aisurrey26" in l][0]
    n = probe.parse_scontrol_node_line(drained_line)
    assert n is not None
    assert n.is_usable is False
    assert n.gpu_free == 0


def test_parse_scontrol_node_line_skips_non_node(probe):
    assert probe.parse_scontrol_node_line("Some header line") is None
    assert probe.parse_scontrol_node_line("") is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _parse_all(probe):
    nodes = [
        n for line in SCONTROL_FIXTURE.splitlines()
        if (n := probe.parse_scontrol_node_line(line)) is not None
    ]
    return nodes, probe.aggregate_partitions(nodes)


def test_aggregate_partitions_nice_project(probe):
    _, parts = _parse_all(probe)
    p = parts["nice-project"]
    assert p.nodes_total == 1
    assert p.nodes_usable == 1
    assert p.gpus_total == 2
    assert p.gpus_alloc == 1
    assert p.gpus_free == 1
    assert p.dominant_gpu_type == "nvidia_l40s"
    assert p.max_vram_gb == 48


def test_aggregate_partitions_a100(probe):
    _, parts = _parse_all(probe)
    p = parts["a100"]
    assert p.gpus_free == 4           # 4 GPUs, all idle in the fixture
    assert p.max_gpu_per_node == 4    # supports tp=4 workloads
    assert p.max_vram_gb == 80
    assert p.dominant_gpu_type == "nvidia_a100"
    assert p.max_vram_gpu_type == "nvidia_a100"
    assert p.has_mixed_gpu_types is False
    assert p.unknown_gpu_types == []


def test_aggregate_partitions_contested_a6000(probe):
    _, parts = _parse_all(probe)
    p = parts["rtx_a6000_risk"]
    # Single GPU, currently allocated → 0 free.
    assert p.gpus_total == 1
    assert p.gpus_alloc == 1
    assert p.gpus_free == 0


def test_aggregate_partitions_excludes_drained(probe):
    _, parts = _parse_all(probe)
    # l40s_risk has 2 nodes in the fixture (aisurrey26 DRAIN + aisurrey27 IDLE)
    # — usable-node count and free-GPU count must only reflect aisurrey27.
    p = parts["l40s_risk"]
    assert p.nodes_total == 2
    assert p.nodes_usable == 1
    assert p.gpus_free == 2  # only aisurrey27 contributes


def test_aggregate_partitions_mixed_hardware_reports_fattest(probe):
    """cogvis-project has a 3090 node (first-seen, 24 GB) and an A6000 node
    (second-seen, 48 GB). The display pair (type, vram) must correlate —
    if vram=48 GB then the type column shows the A6000, not the first-seen
    3090. This exact mismatch was what the live cluster exposed on
    2026-04-23.
    """
    _, parts = _parse_all(probe)
    p = parts["cogvis-project"]
    assert p.has_mixed_gpu_types is True
    assert p.max_vram_gb == 48
    # Max-VRAM type is the A6000 even though 3090 was first-seen.
    assert p.max_vram_gpu_type == "nvidia_rtx_a6000"
    assert p.dominant_gpu_type == "nvidia_rtx_a6000"
    # Both types are still tracked in gpu_types (for diagnostics).
    assert set(p.gpu_types) == {"nvidia_geforce_rtx_3090", "nvidia_rtx_a6000"}
    # Both types are in our VRAM map, so no unknowns are recorded.
    assert p.unknown_gpu_types == []


def test_aggregate_partitions_narrative_project_uses_ada_vram(probe):
    """Sanity-check a partition whose only GPU type is the 32 GB Ada card
    — before this fix the probe reported it as '—' (unknown)."""
    _, parts = _parse_all(probe)
    p = parts["narrative-project"]
    assert p.max_vram_gb == 32
    assert p.dominant_gpu_type == "nvidia_rtx_5000_ada"
    assert p.unknown_gpu_types == []


def test_aggregate_partitions_experimental_records_unknown(probe):
    """A partition whose only GPU type is unknown to us: VRAM must be 0
    and the unknown type must be listed so render_table can flag it."""
    _, parts = _parse_all(probe)
    p = parts["experimental"]
    assert p.max_vram_gb == 0
    assert p.vram_known is False
    assert p.unknown_gpu_types == ["nvidia_future_card"]
    # dominant_gpu_type falls back to first-seen when no VRAM is known.
    assert p.dominant_gpu_type == "nvidia_future_card"


# ---------------------------------------------------------------------------
# VRAM inference from a run config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_body,gpus,expected_vram",
    [
        # Empty / no scorers → default 8 GB floor.
        ("", 1, 8),
        # Kiwi-DA only → 8 GB default.
        ("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n", 1, 8),
        # Add XL → 24 GB.
        (
            "scorers:\n  - ref: comet/wmt22-cometkiwi-da\n"
            "  - ref: comet/wmt23-cometkiwi-da-xl\n",
            1, 24,
        ),
        # XXL → 48 GB.
        ("scorers:\n  - ref: comet/wmt23-cometkiwi-da-xxl\n", 1, 48),
        # 13B Tower → 48 GB.
        ("scorers:\n  - ref: tower/towerinstruct-13b-v0.1\n", 1, 48),
        # Mistral → 24 GB (7B HF-transformers backend).
        ("scorers:\n  - ref: tower/towerinstruct-mistral-7b-v0.2\n", 1, 24),
        # 72B alone with --gres=gpu:1 → runner SKIPS (tp=4 > 1), so effective
        # VRAM need falls back to the default floor. No single-scorer job.
        ("scorers:\n  - ref: tower/tower-plus-72b\n", 1, 8),
        # Same 72B with --gres=gpu:4 → runnable, 80 GB.
        ("scorers:\n  - ref: tower/tower-plus-72b\n", 4, 80),
        # Full matrix on --gres=gpu:1 → 72B dropped, XXL dominates (48 GB).
        # This is the exact cluster-side defect that made the first probe
        # run report nice-project as NO-FIT.
        (
            "scorers:\n  - ref: comet/wmt22-cometkiwi-da\n"
            "  - ref: tower/tower-plus-2b\n"
            "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
            "  - ref: tower/tower-plus-72b\n",
            1, 48,
        ),
        # Same matrix on --gres=gpu:4 → 72B runs, 80 GB dominates.
        (
            "scorers:\n  - ref: comet/wmt22-cometkiwi-da\n"
            "  - ref: tower/tower-plus-2b\n"
            "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
            "  - ref: tower/tower-plus-72b\n",
            4, 80,
        ),
    ],
)
def test_infer_vram_need(probe, config_body, gpus, expected_vram):
    assert probe.infer_vram_need(config_body, gpus=gpus).max_vram_gb == expected_vram


def test_infer_vram_need_reports_each_scorer(probe):
    cfg = (
        "scorers:\n"
        "  - ref: comet/wmt22-cometkiwi-da\n"
        "  - ref: tower/towerinstruct-13b-v0.1\n"
    )
    need = probe.infer_vram_need(cfg)
    refs = [s["ref"] for s in need.scorers]
    assert refs == ["comet/wmt22-cometkiwi-da", "tower/towerinstruct-13b-v0.1"]
    assert need.scorers[0]["vram_gb"] == 8
    assert need.scorers[1]["vram_gb"] == 48
    # Per-scorer detail now also carries tp + skipped flag.
    assert need.scorers[0]["tp"] == 1
    assert need.scorers[0]["skipped"] is False
    assert need.scorers[1]["tp"] == 1
    assert need.scorers[1]["skipped"] is False


def test_infer_vram_need_records_skipped_tp(probe):
    """72B on --gres=gpu:1 is SKIPPED; report it in ``skipped_tp`` so main()
    can surface it in the footer."""
    cfg = (
        "scorers:\n"
        "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
        "  - ref: tower/tower-plus-72b\n"
    )
    need = probe.infer_vram_need(cfg, gpus=1)
    assert need.max_vram_gb == 48  # XXL wins; 72B not counted
    assert need.skipped_tp == ["tower/tower-plus-72b"]
    # The 72B scorer is still in .scorers but marked skipped.
    skipped = [s for s in need.scorers if s["skipped"]]
    assert len(skipped) == 1
    assert skipped[0]["ref"] == "tower/tower-plus-72b"
    assert skipped[0]["tp"] == 4


def test_infer_vram_need_no_skips_at_gpus_4(probe):
    """Same config on 4 GPUs — nothing is skipped."""
    cfg = (
        "scorers:\n"
        "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
        "  - ref: tower/tower-plus-72b\n"
    )
    need = probe.infer_vram_need(cfg, gpus=4)
    assert need.max_vram_gb == 80  # 72B now runs
    assert need.skipped_tp == []


# ---------------------------------------------------------------------------
# Fit check
# ---------------------------------------------------------------------------

def test_check_fit_ready(probe):
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="a100", gpus=1, mem_mb=0, cpus=0, vram_need_gb=80,
    )
    fit = probe.check_fit(parts["a100"], req)
    assert fit.shape_ok is True
    assert fit.has_capacity_now is True
    assert fit.status_word() == "ready"


def test_check_fit_contested(probe):
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=48,
    )
    fit = probe.check_fit(parts["rtx_a6000_risk"], req)
    assert fit.shape_ok is True
    assert fit.has_capacity_now is False
    assert fit.status_word() == "contested"


def test_check_fit_no_fit_vram(probe):
    """A 72B job → needs 80 GB VRAM → nice-project (48 GB) can't ever run it."""
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="nice-project", gpus=1, mem_mb=0, cpus=0, vram_need_gb=80,
    )
    fit = probe.check_fit(parts["nice-project"], req)
    assert fit.shape_ok is False
    assert fit.status_word() == "no-fit"
    assert any("VRAM" in r for r in fit.reasons)


def test_check_fit_no_fit_gpu_count(probe):
    """Asking for gpu:4 on a partition whose largest node has 2 GPUs → no-fit."""
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="nice-project", gpus=4, mem_mb=0, cpus=0, vram_need_gb=0,
    )
    fit = probe.check_fit(parts["nice-project"], req)
    assert fit.shape_ok is False
    assert any("per-node GPU" in r for r in fit.reasons)


def test_check_fit_no_fit_mem(probe):
    """--mem=256G on nice-project (RealMemory=128000 MB) → no-fit."""
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="nice-project", gpus=1, mem_mb=256 * 1024, cpus=0, vram_need_gb=0,
    )
    fit = probe.check_fit(parts["nice-project"], req)
    assert fit.shape_ok is False
    assert any("RealMemory" in r for r in fit.reasons)


def test_check_fit_unknown_vram(probe):
    """A partition whose GPU type we don't recognise must NOT be reported
    as READY even if it has free GPUs — we can't promise it fits."""
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="experimental", gpus=1, mem_mb=0, cpus=0, vram_need_gb=24,
    )
    fit = probe.check_fit(parts["experimental"], req)
    assert fit.shape_ok is True            # GPUs present, no mem/cpu block
    assert fit.has_capacity_now is True    # 2 free
    assert fit.vram_known is False         # but we don't know the VRAM
    assert fit.status_word() == "unknown-vram"


def test_check_fit_ready_a100_with_tp4(probe):
    """Four-GPU A100 partition must accept tp=4 workload (72B)."""
    _, parts = _parse_all(probe)
    req = probe.Request(
        partition="a100", gpus=4, mem_mb=0, cpus=0, vram_need_gb=80,
    )
    fit = probe.check_fit(parts["a100"], req)
    assert fit.shape_ok is True
    assert fit.has_capacity_now is True
    assert fit.status_word() == "ready"


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_pick_recommended_when_target_contested(probe):
    """Target rtx_a6000_risk is contested; 48 GB partitions with capacity
    exist (nice-project / l40s_risk / cogvis-project). a100 is 80 GB →
    overkill. Right-sizing picks the smallest-VRAM fit; among ties the one
    with MORE free GPUs wins.
    """
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=48,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    rec = probe.pick_recommended(partitions, fits, req)
    # Any of the 48 GB ready partitions. cogvis-project has the most free.
    assert rec in {"nice-project", "l40s_risk", "cogvis-project"}


def test_pick_recommended_none_when_target_ready(probe):
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="a100", gpus=1, mem_mb=0, cpus=0, vram_need_gb=80,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    assert probe.pick_recommended(partitions, fits, req) is None


def test_pick_recommended_skips_unknown_vram(probe):
    """A partition with UNKNOWN-VRAM must NEVER be recommended — we can't
    promise it fits, and a blind recommendation is worse than none.
    """
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    # Target contested; 'experimental' is the only partition with free
    # GPUs that matches 'unknown-vram'. Recommendation must bypass it.
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=48,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    rec = probe.pick_recommended(partitions, fits, req)
    assert rec != "experimental"


def test_pick_recommended_prefers_smallest_fitting(probe):
    """Kiwi-DA (need 8 GB) with target rtx_a6000_risk (contested) → should
    recommend a small card (3090 24 GB), not a100 80 GB."""
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=8,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    rec = probe.pick_recommended(partitions, fits, req)
    # 3090 is 24 GB — smallest of the ready set that fits 8 GB.
    assert rec == "3090"


# ---------------------------------------------------------------------------
# Slurm header parsing
# ---------------------------------------------------------------------------

def test_parse_slurm_header_reads_real_script(probe):
    header = probe.parse_slurm_header(
        (REPO_ROOT / "scripts" / "run_mt_metrix.slurm").read_text()
    )
    assert header["partition"] == "nice-project"
    assert header["gpus"] == "1"
    assert header["mem"] == "120G"
    assert header["cpus"] == "8"


# ---------------------------------------------------------------------------
# End-to-end main() — mock out scontrol via the module's run_scontrol hook.
# ---------------------------------------------------------------------------

def test_main_target_ready_exit_zero(probe, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a100" in out
    assert "READY" in out


def test_main_target_contested_exit_two(probe, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/xcomet-xxl-qe\n")  # 48 GB
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "rtx_a6000_risk", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 2
    out = capsys.readouterr().out
    assert "CONTESTED" in out
    # Recommendation must be shown.
    assert "scripts/submit.sh" in out


def test_main_target_no_fit_exit_one(probe, monkeypatch, capsys, tmp_path):
    """XXL (48 GB need) on 3090 (24 GB) → shape no-fit on VRAM. We use XXL
    rather than 72B because 72B+gpus=1 is now tp-skipped, which would
    collapse vram_need_gb to the 8 GB default and make every partition
    'fit' — that's the bug this suite guards against, not a no-fit case.
    """
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt23-cometkiwi-da-xxl\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "3090", "--gpus", "1",  # 24 GB card, job needs 48
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "NO-FIT" in out
    assert "VRAM" in out


def test_main_target_no_fit_72b_on_nice_project_with_4_gpus(probe, monkeypatch, capsys, tmp_path):
    """72B on --gres=gpu:4 → tp=4 workload, 80 GB need. nice-project has
    2 L40s (48 GB) per node, so:
      - per-node GPU ceiling = 2, --gpus=4 → exceeds
      - VRAM 80 GB > 48 GB → exceeds
    Either condition alone would no-fit; both together prove the probe
    surfaces the tp-scale reality users actually hit.
    """
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: tower/tower-plus-72b\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "nice-project", "--gpus", "4",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "NO-FIT" in out


def test_main_tp_skip_reported_in_footer(probe, monkeypatch, capsys, tmp_path):
    """When a scorer is tp-skipped, the footer must tell the user so they
    can re-submit with more GPUs if they actually wanted that scorer."""
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(
        "scorers:\n"
        "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
        "  - ref: tower/tower-plus-72b\n"
    )
    probe.main([
        "--config", str(cfg),
        "--partition", "nice-project", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    out = capsys.readouterr().out
    assert "tower/tower-plus-72b" in out
    assert "skipped" in out.lower()


def test_main_target_unknown_vram(probe, monkeypatch, capsys, tmp_path):
    """A partition with an unmapped GPU type must surface as UNKNOWN-VRAM
    rather than READY — user may want to verify before trusting the probe.
    """
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "experimental", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    # experimental has free GPUs → shape ok → exit 0, but the status is
    # UNKNOWN-VRAM in the table and the footer flags the unmapped type.
    assert rc == 0
    out = capsys.readouterr().out
    assert "UNKNOWN-VRAM" in out
    assert "nvidia_future_card" in out


def test_main_scontrol_missing_exit_three(probe, monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: None)
    rc = probe.main([
        "--partition", "a100",
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 3


def test_main_json_output_structure(probe, monkeypatch, capsys, tmp_path):
    import json as jsonmod
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: comet/wmt23-cometkiwi-da-xxl\n")
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "a100", "--gpus", "1",
        "--slurm-script", "/dev/null",
        "--json",
    ])
    assert rc == 0
    payload = jsonmod.loads(capsys.readouterr().out)
    assert payload["request"]["partition"] == "a100"
    assert payload["request"]["vram_need_gb"] == 48
    assert payload["target_fit"]["shape_ok"] is True
    # Partitions list should include our whole fixture set.
    part_names = {p["name"] for p in payload["partitions"]}
    assert {"nice-project", "a100", "3090", "rtx_a6000_risk", "l40s_risk"} \
        <= part_names
