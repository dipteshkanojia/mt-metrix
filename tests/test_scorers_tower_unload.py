"""Regression tests for vLLM distributed-state leak across Tower scorers.

In the 2026-04-20 mt-metrix Surrey full-matrix run, four Tower scorers
(``towerinstruct-13b-v0.1``, ``tower-plus-9b``, ``tower-plus-72b``,
``tower-plus-9b-mqm``) were silently skipped with::

    runtime: tensor parallel group already initialized, but of unexpected
    size: get_tensor_model_parallel_world_size()=1 vs.
    tensor_model_parallel_size=2

Root cause: a prior Tower scorer with ``tp=1`` initialised vLLM's
module-level distributed state (``vllm.distributed.parallel_state``). The
previous ``TowerScorer.unload()`` only released torch memory, leaving that
state live. When the runner advanced to a ``tp>1`` Tower scorer, vLLM's
guard tripped at ``LLM()`` init.

Fix: ``unload()`` now calls ``destroy_model_parallel() +
destroy_distributed_environment()`` (from ``vllm.distributed.parallel_state``)
and ``ray.shutdown()`` when the backend is vLLM. These tests pin that
behaviour without requiring a real vLLM install — we inject fakes via
``sys.modules`` and assert the teardown calls happen.
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.tower import TowerScorer


def _install_fakes(monkeypatch):
    destroy_mp = mock.Mock()
    destroy_env = mock.Mock()
    ray_shutdown = mock.Mock()
    ray_is_init = mock.Mock(return_value=True)

    fake_parallel_state = mock.MagicMock()
    fake_parallel_state.destroy_model_parallel = destroy_mp
    fake_parallel_state.destroy_distributed_environment = destroy_env
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", fake_parallel_state
    )

    fake_ray = mock.MagicMock()
    fake_ray.is_initialized = ray_is_init
    fake_ray.shutdown = ray_shutdown
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    # tower.py's unload() also does `import torch` to call cuda.empty_cache().
    # Dev laptops without torch installed would otherwise raise ImportError
    # inside the outer try/except, skipping the parallel-state teardown we're
    # actually testing here. Stub torch with a CUDA-disabled fake so the call
    # chain completes end-to-end.
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available = mock.Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return destroy_mp, destroy_env, ray_shutdown, ray_is_init


def test_unload_vllm_calls_parallel_state_teardown(monkeypatch):
    """With backend=vllm and a loaded engine, unload() must call
    destroy_model_parallel, destroy_distributed_environment, and
    ray.shutdown in that order."""
    cfg = ScorerConfig(
        family="tower",
        name="tower-unload-test",
        model="Unbabel/TowerInstruct-7B-v0.1",
        params={"backend": "vllm"},
    )
    scorer = TowerScorer(cfg)
    scorer._engine = object()  # pretend a model is loaded

    destroy_mp, destroy_env, ray_shutdown, ray_is_init = _install_fakes(monkeypatch)

    scorer.unload()

    destroy_mp.assert_called_once()
    destroy_env.assert_called_once()
    ray_is_init.assert_called_once()
    ray_shutdown.assert_called_once()
    assert scorer._engine is None


def test_unload_vllm_skips_ray_when_not_initialized(monkeypatch):
    """ray.is_initialized() False → ray.shutdown() must NOT be called."""
    cfg = ScorerConfig(
        family="tower",
        name="tower-no-ray",
        model="Unbabel/TowerInstruct-7B-v0.1",
        params={"backend": "vllm"},
    )
    scorer = TowerScorer(cfg)
    scorer._engine = object()

    destroy_mp, destroy_env, ray_shutdown, ray_is_init = _install_fakes(monkeypatch)
    ray_is_init.return_value = False

    scorer.unload()

    destroy_mp.assert_called_once()
    destroy_env.assert_called_once()
    ray_is_init.assert_called_once()
    ray_shutdown.assert_not_called()


def test_unload_transformers_skips_vllm_teardown(monkeypatch):
    """backend=transformers must NOT touch vLLM's parallel state or Ray —
    the transformers backend doesn't initialise either, so calling their
    teardowns would either no-op or raise."""
    cfg = ScorerConfig(
        family="tower",
        name="tower-transformers",
        model="Unbabel/TowerInstruct-7B-v0.1",
        params={"backend": "transformers"},
    )
    scorer = TowerScorer(cfg)
    scorer._engine = object()

    destroy_mp, destroy_env, ray_shutdown, ray_is_init = _install_fakes(monkeypatch)

    scorer.unload()

    destroy_mp.assert_not_called()
    destroy_env.assert_not_called()
    ray_is_init.assert_not_called()
    ray_shutdown.assert_not_called()


def test_unload_early_returns_when_engine_is_none(monkeypatch):
    """No engine loaded → unload is a no-op (idempotent contract from
    the Scorer protocol in base.py)."""
    cfg = ScorerConfig(
        family="tower",
        name="tower-unloaded",
        model="Unbabel/TowerInstruct-7B-v0.1",
        params={"backend": "vllm"},
    )
    scorer = TowerScorer(cfg)
    # do NOT set scorer._engine — leave it None

    destroy_mp, destroy_env, ray_shutdown, ray_is_init = _install_fakes(monkeypatch)

    scorer.unload()

    destroy_mp.assert_not_called()
    destroy_env.assert_not_called()
    ray_is_init.assert_not_called()
    ray_shutdown.assert_not_called()


def test_unload_survives_teardown_exception(monkeypatch):
    """If destroy_model_parallel raises, unload() must log-and-continue,
    not propagate — the runner's per-scorer try/except relies on unload()
    being defensive so one bad scorer doesn't wedge the whole job."""
    cfg = ScorerConfig(
        family="tower",
        name="tower-teardown-crash",
        model="Unbabel/TowerInstruct-7B-v0.1",
        params={"backend": "vllm"},
    )
    scorer = TowerScorer(cfg)
    scorer._engine = object()

    destroy_mp = mock.Mock(side_effect=RuntimeError("boom"))
    destroy_env = mock.Mock()
    ray_shutdown = mock.Mock()
    ray_is_init = mock.Mock(return_value=True)

    fake_parallel_state = mock.MagicMock()
    fake_parallel_state.destroy_model_parallel = destroy_mp
    fake_parallel_state.destroy_distributed_environment = destroy_env
    monkeypatch.setitem(
        sys.modules, "vllm.distributed.parallel_state", fake_parallel_state
    )

    fake_ray = mock.MagicMock()
    fake_ray.is_initialized = ray_is_init
    fake_ray.shutdown = ray_shutdown
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available = mock.Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Should not raise despite destroy_model_parallel blowing up.
    scorer.unload()

    destroy_mp.assert_called_once()
    # destroy_distributed_environment is in the same try/except — skipped
    # after the first raise. That's fine; the outer "except" still wins.
    assert scorer._engine is None
