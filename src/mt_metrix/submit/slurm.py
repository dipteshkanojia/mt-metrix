"""SLURM script rendering and submission for AISURREY.

Honours the conventions from ``~/Documents/Claude/agent-context/aisurrey-cluster.md``:

- Scratch root is ``/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix``
- Torch 2.4.1+cu121 pin, conda env ``mt-metrix``
- a100 partition, ``--exclude=aisurrey26`` (flaky node)
- HF cache redirected to scratch (``HF_HOME``)
- COMET cache redirected too (``COMET_CACHE``)
- HF token loaded from ``~/.hf_token`` if present

Size heuristic (chooses template based on scorer model ids in the run config):

- Tower-72B                         → 4× A100, 08:00:00
- Tower-13B / Tower-Plus-9B         → 2× A100, 04:00:00
- COMET-XXL / XCOMET-XXL / cometkiwi-xxl → 1× A100, 04:00:00
- COMET-XL / XCOMET-XL / cometkiwi-xl    → 1× A100, 02:00:00
- Tower-7B / Tower-Plus-2B          → 1× A100, 02:00:00
- everything else (comet-da, cometkiwi-da, BLEU, chrF++, …) → 1× A100, 01:00:00
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mt_metrix.config import RunConfig

log = logging.getLogger(__name__)


@dataclass
class SlurmPlan:
    gpus: int
    time: str
    partition: str
    mem: str
    cpus_per_task: int
    template: str


def plan_for(config: RunConfig, *, partition: str | None, gpus: int | None, time: str | None) -> SlurmPlan:
    """Pick reasonable SLURM resources for this run.

    Explicit CLI overrides win. Otherwise we pick a plan from the largest
    scorer model in the run.
    """
    default = SlurmPlan(
        gpus=1, time="01:00:00", partition="a100", mem="64G",
        cpus_per_task=8, template="comet_small.sbatch",
    )
    plans: list[SlurmPlan] = [default]

    for scorer in config.scorers:
        lower = (scorer.model or "").lower()
        if "72b" in lower:
            plans.append(SlurmPlan(4, "08:00:00", "a100", "256G", 16, "tower_72b.sbatch"))
        elif "13b" in lower or "9b" in lower:
            plans.append(SlurmPlan(2, "04:00:00", "a100", "128G", 12, "tower_13b.sbatch"))
        elif "xxl" in lower:
            plans.append(SlurmPlan(1, "04:00:00", "a100", "96G", 8, "comet_xxl.sbatch"))
        elif "xl" in lower:
            plans.append(SlurmPlan(1, "02:00:00", "a100", "64G", 8, "comet_xl.sbatch"))
        elif "7b" in lower or "2b" in lower:
            plans.append(SlurmPlan(1, "02:00:00", "a100", "64G", 8, "tower_7b.sbatch"))

    # pick by max wall time → max gpus
    def _score(p: SlurmPlan) -> tuple[int, int]:
        return (_time_to_seconds(p.time), p.gpus)

    plan = max(plans, key=_score)

    if partition:
        plan.partition = partition
    if gpus:
        plan.gpus = gpus
    if time:
        plan.time = time

    return plan


def _time_to_seconds(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + sec


SBATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=mtm-{run_id}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --exclude=aisurrey26
#SBATCH --output={log_root}/slurm-%j.out
#SBATCH --error={log_root}/slurm-%j.err

set -euo pipefail

# --- AISURREY conventions ---
export SCRATCH=${{SCRATCH:-/mnt/fast/nobackup/scratch4weeks/$USER/mt-metrix}}
mkdir -p "$SCRATCH"/models "$SCRATCH"/hf-cache "$SCRATCH"/outputs

export HF_HOME="$SCRATCH/hf-cache"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf-cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf-cache"
export COMET_CACHE="$SCRATCH/models/comet"

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN=$(cat "$HOME/.hf_token")
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# --- conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate {conda_env}

# --- run ---
cd {repo_root}
echo "host: $(hostname)"
echo "gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "scratch: $SCRATCH"
echo "HF_HOME: $HF_HOME"

mt-metrix score \\
  --config {config_path} \\
  --output-root "$SCRATCH/outputs"
"""


def render_script(
    config: RunConfig,
    plan: SlurmPlan,
    repo_root: Path,
    config_path: Path,
    conda_env: str = "mt-metrix",
) -> str:
    log_root = repo_root / "outputs" / "slurm-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    return SBATCH_TEMPLATE.format(
        run_id=config.run_id,
        partition=plan.partition,
        gpus=plan.gpus,
        cpus_per_task=plan.cpus_per_task,
        mem=plan.mem,
        time=plan.time,
        log_root=str(log_root),
        conda_env=conda_env,
        repo_root=shlex.quote(str(repo_root)),
        config_path=shlex.quote(str(config_path)),
    )


def render_and_submit(
    config: RunConfig,
    *,
    cluster: str = "aisurrey",
    partition: str | None = None,
    gpus: int | None = None,
    time: str | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> tuple[int, str | None, Path]:
    """Render the sbatch script and (optionally) submit it.

    Returns ``(returncode, job_id, script_path)``. ``job_id`` is ``None`` on
    dry-run or non-zero submission.
    """
    if cluster != "aisurrey":
        raise ValueError(f"unsupported cluster {cluster!r}")

    repo_root = repo_root or Path.cwd().resolve()
    config_path = Path.cwd() / "outputs" / "submitted" / f"{config.run_id}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    from mt_metrix.config import dump_resolved_config

    dump_resolved_config(config, config_path)

    plan = plan_for(config, partition=partition, gpus=gpus, time=time)
    script = render_script(config, plan, repo_root=repo_root, config_path=config_path)
    script_path = repo_root / "outputs" / "submitted" / f"{config.run_id}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    os.chmod(script_path, 0o755)

    if dry_run:
        return 0, None, script_path

    log.info("submitting SLURM job: sbatch %s", script_path)
    res = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True, check=False
    )
    job_id: str | None = None
    if res.returncode == 0:
        # sbatch prints "Submitted batch job NNNN"
        parts = res.stdout.strip().split()
        if parts and parts[-1].isdigit():
            job_id = parts[-1]
    else:
        log.error("sbatch failed: %s", res.stderr.strip())
    return res.returncode, job_id, script_path
