"""Regression tests for ``TowerScorer._load_vllm`` kwarg passthrough.

The Tower catalogue (``configs/models/tower.yaml``) carries a handful of
optional vLLM engine kwargs that the scorer must forward verbatim to
``vllm.LLM(...)``. Two recent additions are load-bearing on AISURREY:

- ``max_model_len`` — bypasses Mistral-backbone config quirks by pinning
  a context length instead of letting vLLM derive one.
- ``disable_sliding_window`` — added 2026-04-21 as a belt-and-braces
  workaround for the Mistral ``TypeError: unsupported operand type(s)
  for *: 'int' and 'NoneType'`` crash in ``vllm==0.6.3.post1``. The
  ``max_model_len`` pin wasn't always enough; disabling sliding-window
  short-circuits the broken code path.

These tests pin the passthrough contract without requiring a real vLLM
install — we inject a fake ``vllm`` module via ``sys.modules`` that
captures ``LLM.__init__`` kwargs, then assert on them.
"""
from __future__ import annotations

import sys
from unittest import mock

from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.tower import TowerScorer


def _install_fake_vllm(monkeypatch):
    """Inject a fake ``vllm`` module that records LLM(...) kwargs.

    Returns the mock class so tests can inspect ``.call_args.kwargs``.
    """
    fake_llm_cls = mock.MagicMock(name="FakeLLM")
    fake_vllm = mock.MagicMock()
    fake_vllm.LLM = fake_llm_cls
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    # The _load_vllm method also imports AutoTokenizer from transformers
    # after the LLM is constructed; stub it so the test never hits HF.
    fake_tokenizer = mock.MagicMock()
    fake_tokenizer.from_pretrained = mock.Mock(return_value=mock.MagicMock())
    fake_transformers = mock.MagicMock()
    fake_transformers.AutoTokenizer = fake_tokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    return fake_llm_cls


def test_disable_sliding_window_passthrough_when_true(monkeypatch):
    """``disable_sliding_window: true`` in catalogue → forwarded as
    ``disable_sliding_window=True`` to ``vllm.LLM(...)``."""
    cfg = ScorerConfig(
        family="tower",
        name="mistral-test",
        model="Unbabel/TowerInstruct-Mistral-7B-v0.2",
        params={
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "disable_sliding_window": True,
        },
    )
    fake_llm_cls = _install_fake_vllm(monkeypatch)

    scorer = TowerScorer(cfg)
    scorer.load()

    fake_llm_cls.assert_called_once()
    kwargs = fake_llm_cls.call_args.kwargs
    assert kwargs.get("disable_sliding_window") is True
    assert kwargs.get("max_model_len") == 8192
    assert kwargs.get("tensor_parallel_size") == 1


def test_disable_sliding_window_omitted_when_absent(monkeypatch):
    """No ``disable_sliding_window`` in catalogue → kwarg is NOT passed to
    ``vllm.LLM(...)``. vLLM's default (enable) must be preserved for every
    non-Mistral model. Passing ``disable_sliding_window=False`` explicitly
    would also disable vLLM's own fallback logic — we want absence, not
    a forced False."""
    cfg = ScorerConfig(
        family="tower",
        name="no-mistral-test",
        model="Unbabel/TowerInstruct-7B-v0.2",
        params={
            "backend": "vllm",
            "tensor_parallel_size": 1,
        },
    )
    fake_llm_cls = _install_fake_vllm(monkeypatch)

    scorer = TowerScorer(cfg)
    scorer.load()

    fake_llm_cls.assert_called_once()
    kwargs = fake_llm_cls.call_args.kwargs
    assert "disable_sliding_window" not in kwargs


def test_disable_sliding_window_coerces_to_bool(monkeypatch):
    """Accept truthy non-bool values from YAML (``1``, ``"true"``) and
    coerce to a real ``bool`` before handing to vLLM. YAML parsers sometimes
    surface these as strings depending on quoting."""
    cfg = ScorerConfig(
        family="tower",
        name="coerce-test",
        model="Unbabel/TowerInstruct-Mistral-7B-v0.2",
        params={
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "disable_sliding_window": 1,  # truthy non-bool
        },
    )
    fake_llm_cls = _install_fake_vllm(monkeypatch)

    scorer = TowerScorer(cfg)
    scorer.load()

    kwargs = fake_llm_cls.call_args.kwargs
    assert kwargs.get("disable_sliding_window") is True
    assert isinstance(kwargs.get("disable_sliding_window"), bool)


def test_max_model_len_still_passthrough(monkeypatch):
    """Ensure the existing ``max_model_len`` passthrough hasn't regressed
    after the ``disable_sliding_window`` addition."""
    cfg = ScorerConfig(
        family="tower",
        name="mml-test",
        model="Unbabel/TowerInstruct-7B-v0.2",
        params={
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
        },
    )
    fake_llm_cls = _install_fake_vllm(monkeypatch)

    scorer = TowerScorer(cfg)
    scorer.load()

    kwargs = fake_llm_cls.call_args.kwargs
    assert kwargs.get("max_model_len") == 4096
