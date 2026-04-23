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

    # a100: 1 node, 1× A100 80 GB, idle
    "NodeName=aisurrey01 Arch=x86_64 CoresPerSocket=8 CPUTot=32 "
    "RealMemory=512000 "
    "State=IDLE "
    "Gres=gpu:nvidia_a100:1(IDX:0) "
    "Partitions=a100 "
    "AllocTRES= "
    "CfgTRES=cpu=32,mem=500G,gres/gpu=1,gres/gpu:nvidia_a100=1",

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
])


# ---------------------------------------------------------------------------
# GPU type → VRAM mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gpu_type,expected",
    [
        ("nvidia_a100", 80),
        ("NVIDIA_A100_80GB_PCIe", 80),
        ("nvidia_l40s", 48),
        ("nvidia_a6000", 48),
        ("rtx8000", 48),
        ("nvidia_rtx_3090", 24),
        ("nvidia_rtx_2080ti", 11),
        ("", 0),
        ("unknown_card", 0),
    ],
)
def test_vram_for_gpu_type(probe, gpu_type, expected):
    assert probe.vram_for_gpu_type(gpu_type) == expected


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
    assert p.gpus_free == 1
    assert p.max_vram_gb == 80
    assert p.dominant_gpu_type == "nvidia_a100"


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


# ---------------------------------------------------------------------------
# VRAM inference from a run config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_body,expected_vram",
    [
        # Empty / no scorers → default 8 GB floor.
        ("", 8),
        # Kiwi-DA only → 8 GB default.
        ("scorers:\n  - ref: comet/wmt22-cometkiwi-da\n", 8),
        # Add XL → 24 GB.
        (
            "scorers:\n  - ref: comet/wmt22-cometkiwi-da\n"
            "  - ref: comet/wmt23-cometkiwi-da-xl\n",
            24,
        ),
        # Add XXL → 48 GB.
        ("scorers:\n  - ref: comet/wmt23-cometkiwi-da-xxl\n", 48),
        # Add 13B Tower → 48 GB.
        ("scorers:\n  - ref: tower/towerinstruct-13b-v0.1\n", 48),
        # Add 72B Tower → 80 GB.
        ("scorers:\n  - ref: tower/tower-plus-72b\n", 80),
        # Mistral → 24 GB (7B HF-transformers backend).
        ("scorers:\n  - ref: tower/towerinstruct-mistral-7b-v0.2\n", 24),
        # Mixed full matrix — max wins (72B).
        (
            "scorers:\n  - ref: comet/wmt22-cometkiwi-da\n"
            "  - ref: tower/tower-plus-2b\n"
            "  - ref: comet/wmt23-cometkiwi-da-xxl\n"
            "  - ref: tower/tower-plus-72b\n",
            80,
        ),
    ],
)
def test_infer_vram_need(probe, config_body, expected_vram):
    assert probe.infer_vram_need(config_body).max_vram_gb == expected_vram


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


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_pick_recommended_when_target_contested(probe):
    """Target rtx_a6000_risk is contested; nice-project, a100, 3090 are ready.
    Need 48 GB VRAM → a6000_risk and l40s_risk and nice-project qualify, a100
    also qualifies but 80 GB is overkill — smallest-fitting wins, so
    nice-project or l40s_risk (both 48 GB). Among ties more-free wins.
    """
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="rtx_a6000_risk", gpus=1, mem_mb=0, cpus=0, vram_need_gb=48,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    rec = probe.pick_recommended(partitions, fits, req)
    # Must be a 48 GB partition with capacity, not a100 (80 GB, overkill).
    assert rec in {"nice-project", "l40s_risk"}


def test_pick_recommended_none_when_target_ready(probe):
    _, parts_map = _parse_all(probe)
    partitions = sorted(parts_map.values(), key=lambda p: p.name)
    req = probe.Request(
        partition="a100", gpus=1, mem_mb=0, cpus=0, vram_need_gb=80,
    )
    fits = {p.name: probe.check_fit(p, req) for p in partitions}
    assert probe.pick_recommended(partitions, fits, req) is None


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
    monkeypatch.setattr(probe, "run_scontrol_show_node", lambda: SCONTROL_FIXTURE)
    cfg = tmp_path / "run.yaml"
    cfg.write_text("scorers:\n  - ref: tower/tower-plus-72b\n")  # 80 GB need
    rc = probe.main([
        "--config", str(cfg),
        "--partition", "nice-project", "--gpus", "1",  # 48 GB L40s
        "--slurm-script", "/dev/null",
        "--no-colour",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "NO-FIT" in out


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
