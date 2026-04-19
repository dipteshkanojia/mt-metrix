"""Tests for SLURM plan selection + sbatch script rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from mt_metrix.config import DatasetConfig, OutputConfig, RunConfig
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.submit.slurm import plan_for, render_and_submit, render_script


def _mk_config(model_ids: list[str]) -> RunConfig:
    return RunConfig(
        run_id="test",
        dataset=DatasetConfig(kind="local", params={"path": "tests/fixtures/tiny_with_ref.tsv"}),
        scorers=[
            ScorerConfig(family="tower" if "Tower" in m else "comet",
                         name=m.split("/")[-1].lower(), model=m, params={})
            for m in model_ids
        ],
        output=OutputConfig(),
    )


def test_plan_for_small_model_gets_comet_small():
    cfg = _mk_config(["Unbabel/wmt22-cometkiwi-da"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "comet_small.sbatch"
    assert plan.gpus == 1


def test_plan_for_xl_gets_comet_xl():
    cfg = _mk_config(["Unbabel/wmt23-cometkiwi-da-xl"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "comet_xl.sbatch"


def test_plan_for_xxl_gets_comet_xxl():
    cfg = _mk_config(["Unbabel/XCOMET-XXL"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "comet_xxl.sbatch"


def test_plan_for_tower_7b():
    cfg = _mk_config(["Unbabel/TowerInstruct-7B-v0.2"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "tower_7b.sbatch"


def test_plan_for_tower_13b_gets_2_gpus():
    cfg = _mk_config(["Unbabel/TowerInstruct-13B-v0.1"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "tower_13b.sbatch"
    assert plan.gpus == 2


def test_plan_for_tower_72b_gets_4_gpus():
    cfg = _mk_config(["Unbabel/Tower-Plus-72B"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    assert plan.template == "tower_72b.sbatch"
    assert plan.gpus == 4


def test_plan_for_picks_largest_model_in_mixed_run():
    cfg = _mk_config([
        "Unbabel/wmt22-cometkiwi-da",
        "Unbabel/TowerInstruct-13B-v0.1",
    ])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    # 13B wins over base COMET
    assert plan.gpus == 2


def test_plan_for_honours_cli_overrides():
    cfg = _mk_config(["Unbabel/wmt22-cometkiwi-da"])
    plan = plan_for(cfg, partition="gpu", gpus=3, time="06:00:00")
    assert plan.partition == "gpu"
    assert plan.gpus == 3
    assert plan.time == "06:00:00"


def test_render_script_substitutes_fields(tmp_path: Path):
    cfg = _mk_config(["Unbabel/wmt22-cometkiwi-da"])
    plan = plan_for(cfg, partition=None, gpus=None, time=None)
    script = render_script(
        cfg, plan,
        repo_root=tmp_path,
        config_path=tmp_path / "run.yaml",
    )
    assert "#SBATCH --partition=a100" in script
    assert "#SBATCH --exclude=aisurrey26" in script
    assert "mt-metrix score" in script
    assert "HF_HOME" in script
    assert "COMET_CACHE" in script
    assert "mtm-test" in script  # job name derived from run_id


def test_render_and_submit_dry_run_writes_script(tmp_path: Path, monkeypatch):
    cfg = _mk_config(["Unbabel/wmt22-cometkiwi-da"])
    monkeypatch.chdir(tmp_path)
    rc, job_id, script_path = render_and_submit(
        cfg, cluster="aisurrey", dry_run=True, repo_root=tmp_path,
    )
    assert rc == 0
    assert job_id is None
    assert script_path.is_file()
    assert (tmp_path / "outputs" / "submitted" / "test.yaml").is_file()


def test_render_and_submit_unknown_cluster_raises(tmp_path: Path):
    cfg = _mk_config(["Unbabel/wmt22-cometkiwi-da"])
    with pytest.raises(ValueError, match="unsupported cluster"):
        render_and_submit(cfg, cluster="jade2", dry_run=True, repo_root=tmp_path)
