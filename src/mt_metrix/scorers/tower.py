"""Tower-family LLM scorer with GEMBA-DA / GEMBA-MQM / Tower-native prompts.

Inference backend selection:

- ``backend: vllm`` (default) — the high-throughput choice, required for
  cluster-scale runs. Supports tensor parallelism for 13B and 72B models.
- ``backend: transformers`` — single-GPU / CPU-slow fallback, used for local
  smoke tests and small Tower-Plus-2B.

All Tower variants are instruction-tuned chat models. We apply the model's
chat template via the HF tokeniser (both backends support this).

Defaults (from GEMBA / Tower papers):

- ``temperature: 0.0`` (deterministic)
- ``top_p: 1.0``
- ``max_tokens: 50`` for DA, ``256`` for MQM
- ``prompt_mode: gemba-da`` (options: ``gemba-da``, ``gemba-mqm``, ``tower-native``)
"""
from __future__ import annotations

import logging
from typing import Any

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.prompts.gemba_da import build_gemba_da_prompt, parse_gemba_da_score
from mt_metrix.prompts.gemba_mqm import build_gemba_mqm_prompt, parse_gemba_mqm_score
from mt_metrix.prompts.tower_native import build_tower_native_messages
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import register_scorer

log = logging.getLogger(__name__)

PROMPT_MODES = {"gemba-da", "gemba-mqm", "tower-native"}


class TowerScorer:
    def __init__(self, cfg: ScorerConfig) -> None:
        self._cfg = cfg
        if not cfg.model:
            raise ValueError(f"Tower scorer {cfg.name!r} requires a 'model' field")
        mode = cfg.params.get("prompt_mode", "gemba-da")
        if mode not in PROMPT_MODES:
            raise ValueError(
                f"Tower scorer 'prompt_mode' must be one of {sorted(PROMPT_MODES)}, got {mode!r}"
            )
        self._mode: str = mode
        self._backend: str = cfg.params.get("backend", "vllm")
        if self._backend not in {"vllm", "transformers"}:
            raise ValueError(
                f"Tower scorer 'backend' must be vllm|transformers, got {self._backend!r}"
            )
        self._engine: Any = None
        self._tokenizer: Any = None
        self.corpus_score: dict[str, Any] | None = None

    @property
    def config(self) -> ScorerConfig:
        return self._cfg

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def family(self) -> str:
        return "tower"

    @property
    def needs_reference(self) -> bool:
        return False

    def load(self) -> None:
        if self._engine is not None:
            return
        if self._backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self) -> None:
        from vllm import LLM

        params = self._cfg.params
        tp = int(params.get("tensor_parallel_size", _auto_tp(self._cfg.model)))
        dtype = params.get("dtype", "auto")
        max_model_len = params.get("max_model_len")
        gpu_mem = float(params.get("gpu_memory_utilization", 0.90))
        download_dir = params.get("download_dir")

        kwargs: dict[str, Any] = dict(
            model=self._cfg.model,
            tensor_parallel_size=tp,
            dtype=dtype,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=bool(params.get("trust_remote_code", False)),
        )
        if max_model_len is not None:
            kwargs["max_model_len"] = int(max_model_len)
        if download_dir is not None:
            kwargs["download_dir"] = str(download_dir)

        log.info("loading Tower model via vLLM: %s (tp=%d)", self._cfg.model, tp)
        self._engine = LLM(**kwargs)

        # vLLM handles tokenisation internally, but we pull the HF tokenizer for
        # chat-template application in tower-native mode.
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.model)

    def _load_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        params = self._cfg.params
        log.info("loading Tower model via transformers: %s", self._cfg.model)
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.model)
        model = AutoModelForCausalLM.from_pretrained(
            self._cfg.model,
            torch_dtype=params.get("torch_dtype", "auto"),
            device_map=params.get("device_map", "auto"),
        )
        self._engine = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            return_full_text=False,
        )

    def score(self, segments: list[Segment]) -> list[SegmentScore]:
        if self._engine is None:
            self.load()

        if self._mode == "gemba-da":
            prompts = [
                self._as_chat(build_gemba_da_prompt(s.source, s.target, s.lang_pair))
                for s in segments
            ]
            parser = _parse_da
            default_max_tokens = 50
        elif self._mode == "gemba-mqm":
            prompts = [
                self._as_chat(build_gemba_mqm_prompt(s.source, s.target, s.lang_pair))
                for s in segments
            ]
            parser = _parse_mqm
            default_max_tokens = 384
        else:  # tower-native
            prompts = [
                self._tokenizer.apply_chat_template(
                    build_tower_native_messages(s.source, s.target, s.lang_pair),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for s in segments
            ]
            parser = _parse_da
            default_max_tokens = 50

        params = self._cfg.params
        temperature = float(params.get("temperature", 0.0))
        top_p = float(params.get("top_p", 1.0))
        max_tokens = int(params.get("max_tokens", default_max_tokens))

        responses: list[str]
        if self._backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=params.get("stop") or None,
            )
            outputs = self._engine.generate(prompts, sp)
            # preserve input order
            responses = [o.outputs[0].text for o in outputs]
        else:
            outputs = self._engine(
                prompts,
                do_sample=temperature > 0.0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                max_new_tokens=max_tokens,
                batch_size=int(params.get("batch_size", 1)),
            )
            # transformers pipeline returns list[list[dict]] or list[dict]
            if outputs and isinstance(outputs[0], list):
                responses = [o[0]["generated_text"] for o in outputs]
            else:
                responses = [o["generated_text"] for o in outputs]  # type: ignore[index]

        out: list[SegmentScore] = []
        for seg, resp in zip(segments, responses, strict=False):
            score, extra = parser(resp)
            extra["prompt_mode"] = self._mode
            extra["raw_response"] = resp
            out.append(
                SegmentScore(
                    segment_id=seg.segment_id,
                    score=score if score is not None else float("nan"),
                    extra=extra,
                )
            )
        return out

    def _as_chat(self, user_prompt: str) -> str:
        """Wrap a plain prompt in the model's chat template."""
        messages = [{"role": "user", "content": user_prompt}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def unload(self) -> None:
        if self._engine is None:
            return
        try:
            import gc

            import torch

            del self._engine
            self._engine = None
            self._tokenizer = None

            # vLLM persists its model-parallel state at module level
            # (``vllm.distributed.parallel_state``). Without explicit
            # teardown, the NEXT Tower scorer in the same process hits
            # "tensor parallel group already initialized, but of unexpected
            # size: get_tensor_model_parallel_world_size()=N vs.
            # tensor_model_parallel_size=M" whenever its tp differs from
            # the previous scorer's. Observed 2026-04-20 in a full-matrix
            # run on AISURREY a100 (4 GPUs): a prior tp=1 Tower-7B pinned
            # the group to world_size=1, and Tower-13B/Tower-Plus-9B/72B
            # (tp=2 / tp=4) all died at LLM() init before generating a
            # single token.
            if self._backend == "vllm":
                try:
                    from vllm.distributed.parallel_state import (
                        destroy_distributed_environment,
                        destroy_model_parallel,
                    )

                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception as e:  # pragma: no cover — defensive
                    log.warning("vLLM parallel-state teardown raised: %s", e)

                # Ray is used by vLLM for tp>1. Leaving it up attaches the
                # next scorer to the existing cluster with the old
                # world_size, which defeats the parallel_state teardown.
                try:
                    import ray

                    if ray.is_initialized():
                        ray.shutdown()
                except ImportError:  # pragma: no cover — vllm pulls ray
                    pass
                except Exception as e:  # pragma: no cover — defensive
                    log.warning("ray.shutdown raised: %s", e)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover
            log.warning("Tower unload raised: %s", e)


def _auto_tp(model_id: str) -> int:
    """Heuristic tensor-parallel size based on model name."""
    lower = model_id.lower()
    if "72b" in lower:
        return 4
    if "13b" in lower or "9b" in lower:
        return 2
    return 1


def _parse_da(response: str) -> tuple[float | None, dict[str, Any]]:
    score, ok = parse_gemba_da_score(response)
    return score, {"parse_ok": ok}


def _parse_mqm(response: str) -> tuple[float | None, dict[str, Any]]:
    score, ok, errors = parse_gemba_mqm_score(response)
    return score, {
        "parse_ok": ok,
        "errors": [
            {"severity": e.severity, "category": e.category, "span": e.span}
            for e in errors
        ],
    }


def _factory(cfg: ScorerConfig) -> TowerScorer:
    return TowerScorer(cfg)


register_scorer("tower", _factory)
