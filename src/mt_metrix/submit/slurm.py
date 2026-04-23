"""SLURM helpers for AISURREY.

This module is intentionally thin. The canonical path for submitting a job
on AISURREY is ``scripts/submit.sh`` — a bash wrapper that performs the
six pre-flight checks (partition exists, not the nonexistent ``gpu``
partition; conda env present; no duplicate in queue; cluster probe for
live capacity + VRAM fit; ``sbatch --test-only`` accepts the plan) and
submits with ``--exclude=aisurrey26``.

See ``~/Documents/Claude/agent-context/aisurrey-deploy.md`` for the full
SOP that informs this design.

What this module provides:

- :func:`resolve_submit_script` — locate ``scripts/submit.sh`` from code.
- :func:`submit_via_wrapper` — programmatic caller of ``scripts/submit.sh``
  (runs the same pre-flight). Useful when wiring mt-metrix into a larger
  pipeline (e.g. a training loop that launches evaluation jobs).
- :func:`resolve_run_slurm_script` — locate the parameterised
  ``scripts/run_mt_metrix.slurm`` template for callers that need to
  inspect or patch the sbatch directives.

What this module deliberately does NOT provide:

- Size-tiered template selection. We used to pick per-model-size sbatch
  files from scorer model ids; this fought right-sizing (see
  ``docs/AISURREY.md``). Pick the partition and gres on the CLI:
  ``scripts/submit.sh configs/runs/x.yaml -p rtx_a6000_risk --gres=gpu:1``.
- Direct ``sbatch`` invocation bypassing the pre-flight. If you are in a
  context that legitimately cannot call ``scripts/submit.sh`` (e.g. a
  compute-node postprocessor), pass ``run_preflight=False`` and accept
  that the common failure modes will bite harder.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_FLAKY_NODE = "aisurrey26"
_SUBMIT_SCRIPT = "scripts/submit.sh"
_RUN_SLURM = "scripts/run_mt_metrix.slurm"


def resolve_submit_script(repo_root: Path | None = None) -> Path:
    """Return the path to ``scripts/submit.sh`` for this repo."""
    root = Path(repo_root) if repo_root else Path.cwd()
    candidate = root / _SUBMIT_SCRIPT
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{_SUBMIT_SCRIPT} not found relative to {root!s}. "
            f"Run from the mt-metrix repo root or pass repo_root=."
        )
    return candidate


def resolve_run_slurm_script(repo_root: Path | None = None) -> Path:
    """Return the path to ``scripts/run_mt_metrix.slurm`` for this repo."""
    root = Path(repo_root) if repo_root else Path.cwd()
    candidate = root / _RUN_SLURM
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{_RUN_SLURM} not found relative to {root!s}. "
            f"Run from the mt-metrix repo root or pass repo_root=."
        )
    return candidate


def submit_via_wrapper(
    config_path: str | Path,
    *,
    sbatch_args: list[str] | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> tuple[int, str | None]:
    """Submit via ``scripts/submit.sh`` (runs the six pre-flight checks).

    ``sbatch_args`` are forwarded verbatim to ``sbatch`` inside the wrapper,
    e.g. ``["-p", "rtx_a6000_risk", "--gres=gpu:1", "--time=02:00:00"]``.
    The wrapper already adds ``--exclude=aisurrey26``; callers should not
    duplicate that.

    Returns ``(returncode, job_id)``. ``job_id`` is ``None`` on dry-run or
    if sbatch output cannot be parsed.
    """
    script = resolve_submit_script(repo_root)
    config = Path(config_path)
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")

    cmd: list[str] = [str(script), str(config)]
    if dry_run:
        cmd.append("--dry-run")
    if sbatch_args:
        cmd.extend(sbatch_args)

    log.info("invoking %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # submit.sh prints sbatch's stdout verbatim on success: "Submitted batch job NNN"
    job_id: str | None = None
    for line in (res.stdout or "").splitlines():
        parts = line.strip().split()
        if parts and parts[-1].isdigit() and "Submitted" in line:
            job_id = parts[-1]
            break
    if res.returncode != 0:
        log.error("submit.sh failed (rc=%d): %s", res.returncode, res.stderr.strip())
    return res.returncode, job_id


__all__ = [
    "resolve_submit_script",
    "resolve_run_slurm_script",
    "submit_via_wrapper",
    "FLAKY_NODE",
]

FLAKY_NODE = _FLAKY_NODE
