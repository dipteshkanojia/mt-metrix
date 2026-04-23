"""Tests for the SLURM submission glue.

The canonical submit path is ``scripts/submit.sh``. Python submission is a
thin wrapper that shells out to it and parses the job id on success. These
tests cover:

- Path resolution for ``scripts/submit.sh`` and ``scripts/run_mt_metrix.slurm``.
- ``submit_via_wrapper`` happy path (monkeypatches subprocess).
- ``submit_via_wrapper`` sbatch-failure path.
- The wrapper's ``--dry-run`` flag passes through.
- The shell wrapper itself is syntactically valid (``bash -n``).
- The slurm script has the non-negotiable AISURREY safety settings baked in.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mt_metrix.submit.slurm import (
    FLAKY_NODE,
    resolve_run_slurm_script,
    resolve_submit_script,
    submit_via_wrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_submit_script_finds_scripts_submit_sh():
    path = resolve_submit_script(REPO_ROOT)
    assert path == REPO_ROOT / "scripts" / "submit.sh"
    assert path.is_file()


def test_resolve_submit_script_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="submit.sh not found"):
        resolve_submit_script(tmp_path)


def test_resolve_run_slurm_script_finds_template():
    path = resolve_run_slurm_script(REPO_ROOT)
    assert path == REPO_ROOT / "scripts" / "run_mt_metrix.slurm"
    assert path.is_file()


# ---------------------------------------------------------------------------
# submit_via_wrapper — monkeypatched subprocess
# ---------------------------------------------------------------------------

def test_submit_via_wrapper_parses_job_id(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("run:\n  id: x\n")

    def fake_run(cmd, capture_output, text, check):
        assert cmd[0].endswith("submit.sh")
        assert cmd[1] == str(cfg)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="pre-flight OK. Submitting...\nSubmitted batch job 123456\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, job_id = submit_via_wrapper(cfg, repo_root=REPO_ROOT)
    assert rc == 0
    assert job_id == "123456"


def test_submit_via_wrapper_returns_error_on_nonzero(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("run:\n  id: x\n")

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1,
            stdout="",
            stderr="FAIL: partition 'gpu' does not exist\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, job_id = submit_via_wrapper(cfg, repo_root=REPO_ROOT)
    assert rc == 1
    assert job_id is None


def test_submit_via_wrapper_forwards_sbatch_args(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("run:\n  id: x\n")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="Submitted batch job 42\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    submit_via_wrapper(
        cfg,
        sbatch_args=["-p", "rtx_a6000_risk", "--gres=gpu:1", "--time=02:00:00"],
        repo_root=REPO_ROOT,
    )
    cmd = captured["cmd"]
    assert "-p" in cmd and "rtx_a6000_risk" in cmd
    assert "--gres=gpu:1" in cmd and "--time=02:00:00" in cmd


def test_submit_via_wrapper_dry_run_passes_flag(monkeypatch, tmp_path: Path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("run:\n  id: x\n")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="dry-run mode: pre-flight OK\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, job_id = submit_via_wrapper(cfg, dry_run=True, repo_root=REPO_ROOT)
    assert "--dry-run" in captured["cmd"]
    assert rc == 0
    assert job_id is None  # dry-run doesn't print "Submitted batch job"


def test_submit_via_wrapper_raises_on_missing_config(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config not found"):
        submit_via_wrapper(tmp_path / "nope.yaml", repo_root=REPO_ROOT)


# ---------------------------------------------------------------------------
# Shell script content checks — AISURREY safety settings are baked in
# ---------------------------------------------------------------------------

def test_submit_sh_is_syntactically_valid():
    script = resolve_submit_script(REPO_ROOT)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_run_mt_metrix_slurm_is_syntactically_valid():
    script = resolve_run_slurm_script(REPO_ROOT)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    result = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_run_slurm_excludes_flaky_node_via_wrapper_default():
    # The slurm template itself does NOT hardcode --exclude=aisurrey26;
    # submit.sh adds it every submission. Verify the wrapper does so.
    script_txt = resolve_submit_script(REPO_ROOT).read_text()
    assert FLAKY_NODE in script_txt
    assert "--exclude=" in script_txt


def test_run_slurm_has_hf_cache_redirect():
    txt = resolve_run_slurm_script(REPO_ROOT).read_text()
    # HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE must be exported in
    # the sbatch header, not only in .bashrc (aisurrey-deploy.md rule #4).
    assert "HF_HOME=" in txt
    assert "TRANSFORMERS_CACHE=" in txt
    assert "HF_DATASETS_CACHE=" in txt


def test_run_slurm_has_nice_project_default_partition():
    txt = resolve_run_slurm_script(REPO_ROOT).read_text()
    # Default must be a real partition, not "gpu" (aisurrey-deploy.md rule #1).
    # Changed a100 -> nice-project on 2026-04-23: nice-project is NICE-group
    # dedicated (2× L40s 48 GB, zero queue contention) and covers every
    # full-matrix scorer except Tower-72B, which has its own dedicated
    # config (configs/runs/surrey_legal_tower72b.yaml) that overrides
    # -p a100 --gres=gpu:4 at submit time.
    assert "#SBATCH --partition=nice-project" in txt
    assert "--partition=gpu" not in txt
    # Defensive: a100 should NOT be the default any more — it's reserved
    # for the Tower-72B follow-up only.
    assert "#SBATCH --partition=a100" not in txt


def test_run_slurm_activates_scratch_prefix_env():
    txt = resolve_run_slurm_script(REPO_ROOT).read_text()
    # Env lives at $SCRATCH/conda_env (prefix path) not a named env on
    # /mnt/fast/nobackup/users — user volume can't hold torch+vllm+comet deps.
    assert 'conda activate "$SCRATCH/conda_env"' in txt


def test_run_slurm_sets_short_ray_tmpdir():
    """vLLM launches Ray for 13B+ tensor-parallel. Ray builds an AF_UNIX
    socket path under whatever TMPDIR points at, and Linux caps those at
    107 bytes. With TMPDIR=$SCRATCH/tmp (44 chars already) the resulting
    session socket overflows and Tower-13B/9B/72B/9B-mqm all crash with
    ``validate_socket_filename failed`` before a single token is generated.
    The slurm template must override RAY_TMPDIR onto /tmp before the run.
    """
    txt = resolve_run_slurm_script(REPO_ROOT).read_text()
    assert "RAY_TMPDIR=" in txt
    assert "/tmp/ray_" in txt, (
        "RAY_TMPDIR must live under /tmp (short path) to keep the Ray "
        "session socket under the AF_UNIX 107-byte limit"
    )


def test_submit_sh_rejects_partition_gpu_literal():
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # submit.sh must catch the `-p gpu` typo explicitly.
    assert 'PARTITION" == "gpu"' in txt
    assert "no 'gpu' partition on AISURREY" in txt.lower() or \
           "no 'gpu' partition" in txt


def test_submit_sh_does_sbatch_test_only():
    txt = resolve_submit_script(REPO_ROOT).read_text()
    assert "sbatch --test-only" in txt


def test_submit_sh_warns_above_four_gpus():
    txt = resolve_submit_script(REPO_ROOT).read_text()
    assert "soft cap" in txt.lower() or "4-GPU" in txt or "4 GPUs" in txt


# ---------------------------------------------------------------------------
# Pre-flight #5 soft-warn path (transient resource contention vs. genuine
# shape violation). Added 2026-04-23 after nice-project's sole node had both
# L40s allocated and submit.sh hard-failed on a job that was only contended,
# not shape-invalid.
# ---------------------------------------------------------------------------

def test_submit_sh_distinguishes_transient_from_shape_error():
    """Wrapper must branch on the error message so transient contention
    doesn't block a submission that's actually fine shape-wise."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # The grep on SLURM's error phrasing.
    assert "Requested node configuration is not available" in txt, (
        "pre-flight #5 must detect the transient-contention error text"
    )
    # The shape-vs-transient branching logic's own keywords.
    assert "genuine shape violation" in txt, (
        "shape-violation branch must self-label so the red path is obvious"
    )
    assert "transient" in txt.lower(), (
        "transient branch must self-label so the yellow path is obvious"
    )
    # The user escape hatch.
    assert "Ctrl-C within 5s" in txt, (
        "transient branch must give the user 5s to abort before proceeding"
    )


def test_submit_sh_parses_mem_suffixes():
    """The _parse_mem_mb helper must handle the suffixes SLURM accepts."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # Helper must be defined and cover G/M/T/K plus bare numbers.
    assert "_parse_mem_mb()" in txt
    for suffix in ["T", "G", "M", "K"]:
        assert f"[0-9]+){suffix}" in txt, f"--mem parser missing {suffix} suffix"


def test_submit_sh_queries_partition_ceiling_before_failing():
    """Shape check must actually cross-reference the partition's real
    per-node ceiling, not just accept or reject blindly."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # Must read sinfo for RealMemory / CPUs / Gres per node.
    assert "sinfo -h -p \"$PARTITION\" -N -o '%m'" in txt, (
        "shape check must ask sinfo for largest-node RealMemory"
    )
    assert "sinfo   -h -p \"$PARTITION\" -N -o '%c'" in txt, (
        "shape check must ask sinfo for largest-node CPU count"
    )
    assert "sinfo   -h -p \"$PARTITION\" -N -o '%G'" in txt, (
        "shape check must ask sinfo for largest-node Gres"
    )


# ---------------------------------------------------------------------------
# Cluster-probe integration. submit.sh must run scripts/cluster_probe.py as
# pre-flight [5/6] so the user sees live per-partition capacity and a VRAM-
# aware recommendation BEFORE sbatch makes its own decision.
# ---------------------------------------------------------------------------

def test_submit_sh_runs_cluster_probe_before_test_only():
    """Pre-flight [5/6] must invoke cluster_probe.py, and the --test-only
    step must still run afterwards (it's now labelled [6/6])."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # The probe must be invoked.
    assert "scripts/cluster_probe.py" in txt, (
        "submit.sh must call scripts/cluster_probe.py for DETECT+COMPREHEND+ADVISE"
    )
    assert "python3 \"$CLUSTER_PROBE\"" in txt, (
        "probe is stdlib-only and must run via python3 before conda activate"
    )
    # Probe comes BEFORE the --test-only step.
    probe_idx = txt.index("scripts/cluster_probe.py")
    test_only_idx = txt.index("sbatch --test-only")
    assert probe_idx < test_only_idx, (
        "cluster_probe.py must run BEFORE sbatch --test-only"
    )
    # Step counters updated.
    assert "[5/6] cluster probe" in txt
    assert "[6/6] sbatch --test-only" in txt


def test_submit_sh_maps_cluster_probe_exit_codes():
    """submit.sh must distinguish probe exit codes: 0 ready, 1 no-fit
    (hard fail), 2 contested (warn + grace), 3 probe failed (proceed)."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    # All four case arms must be handled explicitly (we branch on $PROBE_RC).
    assert 'PROBE_RC=$?' in txt
    # 1 = no-fit must call fail.
    assert "target partition cannot run this shape" in txt.lower() or \
           "pick a different -p" in txt
    # 2 = contested must have a 5s grace.
    probe_block_start = txt.index("[5/6] cluster probe")
    probe_block_end = txt.index("[6/6] sbatch --test-only")
    probe_block = txt[probe_block_start:probe_block_end]
    assert "fully allocated right now" in probe_block
    assert "read -r -t 5" in probe_block
    # 3 = probe failed must NOT block; it warns and falls through.
    assert "couldn't query scontrol" in probe_block


def test_submit_sh_forwards_overrides_to_probe():
    """CLI overrides (--mem, --cpus-per-task, --gres=gpu:N) must be forwarded
    to the probe so it evaluates the real request, not the slurm-header
    defaults."""
    txt = resolve_submit_script(REPO_ROOT).read_text()
    probe_start = txt.index("[5/6] cluster probe")
    probe_end = txt.index("[6/6] sbatch --test-only")
    probe_block = txt[probe_start:probe_end]
    # Mem override forwarded.
    assert '--mem=*)' in probe_block and 'PROBE_ARGS+=("--mem"' in probe_block
    # Cpus override forwarded.
    assert '--cpus-per-task=*)' in probe_block and 'PROBE_ARGS+=("--cpus"' in probe_block
    # GPU count forwarded.
    assert 'PROBE_ARGS+=("--gpus"' in probe_block


def test_cluster_probe_script_exists_and_is_executable_python():
    """The probe must live at scripts/cluster_probe.py and parse as valid
    Python. We don't import it (stdlib-only by design — no mt_metrix
    coupling), just syntax-check via ``python -c``."""
    import subprocess
    probe = REPO_ROOT / "scripts" / "cluster_probe.py"
    assert probe.is_file(), "scripts/cluster_probe.py must exist"
    # Validate with -m py_compile, which raises SyntaxError if malformed.
    res = subprocess.run(
        ["python3", "-m", "py_compile", str(probe)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"cluster_probe.py failed py_compile: {res.stderr}"
