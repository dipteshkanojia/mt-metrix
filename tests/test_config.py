"""Tests for config loading, !include resolution, and ref: lookups."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mt_metrix.config import _coerce, _set_dotted, load_run_config


def test_load_run_config_with_refs(project_root: Path, tmp_path: Path):
    """Catalogue refs resolve against configs/models/*.yaml."""
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "run": {"id": "test"},
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                    "columns": {"source": "source", "target": "target"},
                },
                "scorers": [{"ref": "sacrebleu/bleu"}],
                "output": {"root": "outputs", "formats": ["tsv"]},
            }
        )
    )
    cfg = load_run_config(cfg_path, catalogue_roots=[project_root])
    assert cfg.run_id == "test"
    assert len(cfg.scorers) == 1
    assert cfg.scorers[0].family == "sacrebleu"
    assert cfg.scorers[0].name == "bleu"


def test_load_run_config_inline_scorer(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [
                    {
                        "family": "sacrebleu",
                        "name": "custom-bleu",
                        "params": {"metric": "bleu", "tokenize": "13a"},
                    }
                ],
            }
        )
    )
    cfg = load_run_config(cfg_path, catalogue_roots=[project_root])
    assert cfg.scorers[0].family == "sacrebleu"
    assert cfg.scorers[0].name == "custom-bleu"
    assert cfg.scorers[0].params["tokenize"] == "13a"


def test_load_run_config_overrides_merge_into_params(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [
                    {"ref": "sacrebleu/bleu", "overrides": {"tokenize": "intl"}}
                ],
            }
        )
    )
    cfg = load_run_config(cfg_path, catalogue_roots=[project_root])
    assert cfg.scorers[0].params["tokenize"] == "intl"
    # smooth_method comes from the catalogue default
    assert cfg.scorers[0].params["smooth_method"] == "exp"


def test_load_run_config_unknown_ref_raises(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [{"ref": "nosuch/model"}],
            }
        )
    )
    with pytest.raises(KeyError, match="Unknown scorer ref"):
        load_run_config(cfg_path, catalogue_roots=[project_root])


def test_load_run_config_missing_dataset(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump({"scorers": [{"ref": "sacrebleu/bleu"}]}))
    with pytest.raises(ValueError, match="missing required 'dataset'"):
        load_run_config(cfg_path, catalogue_roots=[project_root])


def test_load_run_config_no_scorers(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="must list at least one scorer"):
        load_run_config(cfg_path, catalogue_roots=[project_root])


def test_load_run_config_cli_overrides(project_root: Path, tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "run": {"id": "orig"},
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [{"ref": "sacrebleu/bleu"}],
            }
        )
    )
    cfg = load_run_config(
        cfg_path,
        catalogue_roots=[project_root],
        overrides=["run.id=ci-run"],
    )
    assert cfg.run_id == "ci-run"


def test_catalogue_top_level_needs_reference_propagates_into_params(
    project_root: Path, tmp_path: Path
):
    """Catalogue entries declare ``needs_reference`` at the top level (not
    under ``params:``). The resolver must surface that into ``ScorerConfig.
    params`` so scorers picking it up via ``cfg.params.get("needs_reference")``
    see the correct flag. Without this, all COMET-QE variants whose names
    contain ``-qe-`` used to fall through to a buggy heuristic and end up
    marked as needing references — the 2026-04-19 full-matrix regression.
    """
    # wmt20-comet-qe-da has needs_reference: false at the top level
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [{"ref": "comet/wmt20-comet-qe-da"}],
            }
        )
    )
    cfg = load_run_config(cfg_path, catalogue_roots=[project_root])
    assert cfg.scorers[0].params.get("needs_reference") is False, (
        "top-level needs_reference: false in catalogue must propagate to params"
    )

    # wmt22-comet-da has needs_reference: true at the top level — also covered
    cfg_path2 = tmp_path / "run2.yaml"
    cfg_path2.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [{"ref": "comet/wmt22-comet-da"}],
            }
        )
    )
    cfg2 = load_run_config(cfg_path2, catalogue_roots=[project_root])
    assert cfg2.scorers[0].params.get("needs_reference") is True


def test_scorer_overrides_win_over_catalogue_top_level_flag(
    project_root: Path, tmp_path: Path
):
    """If a run config explicitly sets ``needs_reference`` via ``overrides``,
    that must beat the catalogue's top-level value — lets users run e.g.
    XCOMET-XL in a non-default mode from a single run file.
    """
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "kind": "local",
                    "path": str(project_root / "tests/fixtures/tiny_with_ref.tsv"),
                },
                "scorers": [
                    {
                        "ref": "comet/wmt22-comet-da",  # catalogue says needs_reference: true
                        "overrides": {"needs_reference": False},
                    }
                ],
            }
        )
    )
    cfg = load_run_config(cfg_path, catalogue_roots=[project_root])
    assert cfg.scorers[0].params.get("needs_reference") is False


def test_coerce_types():
    assert _coerce("true") is True
    assert _coerce("no") is False
    assert _coerce("42") == 42
    assert _coerce("3.14") == pytest.approx(3.14)
    assert _coerce("hello") == "hello"


def test_set_dotted_creates_nested():
    target: dict = {}
    _set_dotted(target, "a.b.c", 7)
    assert target == {"a": {"b": {"c": 7}}}
