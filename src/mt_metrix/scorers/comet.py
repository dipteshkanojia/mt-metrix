"""COMET-family scorer.

Thin wrapper over ``unbabel-comet``:

1. ``download_model(model_id)`` — downloads to the HF cache (redirected
   via the ``HF_HOME`` env var; set this before running).
2. ``load_from_checkpoint(ckpt_path)`` — loads the PyTorch Lightning model.
3. ``model.predict(samples, batch_size, gpus, num_workers)`` — returns
   segment-level scores, plus extras for XCOMET (error spans).

The same scorer handles every COMET variant — the differences are in
``needs_reference`` (DA-reference vs Kiwi-QE) and in the sample dict shape.

Paper-sourced defaults (COMET paper series, Rei et al. 2020/2022/2023):

* ``batch_size=16`` for XL/XXL, ``64`` for base/large, ``128`` for small.
* ``gpus=1`` (single-GPU default).
* ``num_workers=2``.
* XCOMET adds ``output_seg_err_spans=True`` to populate error-span extras.

See ``docs/PARAMETERS.md`` for citations and per-variant defaults.
"""
from __future__ import annotations

import logging
from typing import Any

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import register_scorer

log = logging.getLogger(__name__)


class CometScorer:
    def __init__(self, cfg: ScorerConfig) -> None:
        self._cfg = cfg
        if not cfg.model:
            raise ValueError(f"COMET scorer {cfg.name!r} requires a 'model' field")
        self._model: Any = None
        self._needs_reference: bool = bool(
            cfg.params.get(
                "needs_reference",
                # heuristic fallback from the model id
                _infer_needs_reference(cfg.model),
            )
        )
        self.corpus_score: dict[str, Any] | None = None

    @property
    def config(self) -> ScorerConfig:
        return self._cfg

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def family(self) -> str:
        return "comet"

    @property
    def needs_reference(self) -> bool:
        return self._needs_reference

    def load(self) -> None:
        if self._model is not None:
            return
        from pathlib import Path

        from comet import download_model, load_from_checkpoint

        log.info("downloading COMET model %s (respecting HF_HOME cache)", self._cfg.model)
        ckpt_path = download_model(self._cfg.model)
        _raise_if_marian_layout(self._cfg.model, Path(ckpt_path))
        log.info("loading checkpoint from %s", ckpt_path)
        self._model = load_from_checkpoint(ckpt_path)

    def score(self, segments: list[Segment]) -> list[SegmentScore]:
        if self._model is None:
            self.load()
        assert self._model is not None

        samples: list[dict[str, str]] = []
        for s in segments:
            sample: dict[str, str] = {"src": s.source, "mt": s.target}
            if self.needs_reference:
                if not s.has_reference():
                    raise ValueError(
                        f"COMET scorer {self.name!r} requires references but "
                        f"segment {s.segment_id!r} has none"
                    )
                sample["ref"] = s.reference or ""
            samples.append(sample)

        params = self._cfg.params
        configured_batch_size = int(params.get("batch_size", 16))
        batch_size = _resolve_xxl_batch_size(
            self._cfg.model, configured_batch_size, _detected_vram_gb()
        )
        if batch_size != configured_batch_size:
            log.warning(
                "COMET model %s: auto-downshifted batch_size %d -> %d because "
                "detected VRAM is below the 60 GB XXL threshold (48 GB L40s OOMs "
                "at batch=8; use A100 80 GB to keep the full batch). Override by "
                "running on an 80 GB card or forcing batch_size explicitly.",
                self._cfg.model,
                configured_batch_size,
                batch_size,
            )
        gpus = int(params.get("gpus", 1))
        num_workers = int(params.get("num_workers", 2))
        progress_bar = bool(params.get("progress_bar", True))
        want_spans = bool(params.get("output_seg_err_spans", _is_xcomet(self._cfg.model)))

        predict_kwargs: dict[str, Any] = dict(
            samples=samples,
            batch_size=batch_size,
            gpus=gpus,
            num_workers=num_workers,
            progress_bar=progress_bar,
        )
        # XCOMET exposes this kwarg; older COMET models ignore it. Try/except
        # shields both cases.
        if want_spans:
            predict_kwargs["output_seg_err_spans"] = True

        try:
            result = self._model.predict(**predict_kwargs)
        except TypeError as e:
            # Older COMET versions reject unknown kwargs. Strip and retry.
            if "output_seg_err_spans" in predict_kwargs:
                log.warning(
                    "COMET model %s doesn't support output_seg_err_spans: %s",
                    self._cfg.model, e,
                )
                predict_kwargs.pop("output_seg_err_spans", None)
                result = self._model.predict(**predict_kwargs)
            else:
                raise

        scores = getattr(result, "scores", None) or result["scores"]
        system_score = getattr(result, "system_score", None)
        err_spans = (
            getattr(result, "metadata", {}).get("error_spans")
            if hasattr(result, "metadata")
            else result.get("error_spans") if isinstance(result, dict) else None
        )

        out: list[SegmentScore] = []
        for i, seg in enumerate(segments):
            extra: dict[str, Any] = {}
            if err_spans is not None and i < len(err_spans):
                extra["error_spans"] = err_spans[i]
            out.append(
                SegmentScore(
                    segment_id=seg.segment_id,
                    score=float(scores[i]),
                    extra=extra,
                )
            )

        if system_score is not None:
            self.corpus_score = {
                "system_score": float(system_score),
                "model": self._cfg.model,
            }
        return out

    def unload(self) -> None:
        if self._model is None:
            return
        try:
            import gc

            import torch

            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("unload of COMET model %s raised: %s", self._cfg.model, e)


def _is_xcomet(model_id: str | None) -> bool:
    return bool(model_id) and "xcomet" in model_id.lower()


def _is_xxl(model_id: str | None) -> bool:
    """Catch every XXL COMET/XCOMET variant by the ``xxl`` substring."""
    return bool(model_id) and "xxl" in model_id.lower()


def _detected_vram_gb() -> float | None:
    """Return the smallest per-device VRAM in GiB across visible CUDA devices,
    or ``None`` if no device is available (CPU-only run, no torch, etc.).

    We pick the minimum rather than the total so mixed-GPU nodes still get
    a safe answer — batch_size is bounded by the smallest card.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — torch is a hard dep
        return None
    if not torch.cuda.is_available():
        return None
    try:
        n = torch.cuda.device_count()
        if n == 0:
            return None
        return min(
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(n)
        )
    except Exception:  # pragma: no cover — defensive
        return None


# 60 GiB is the empirical XXL cut-off: a 48 GiB L40s OOMs at batch=8 with
# XCOMET-XXL / CometKiwi-XXL (XXL-QE peaks at ~41 GiB in fp32 state-dict load
# before moving to GPU, then grows with batch*seq_len during predict); an
# 80 GiB A100 sits comfortably there. Anything in between is treated as a
# safer downshift target — we haven't seen a 60-80 GiB card in production
# on AISURREY, but if one turns up we'd rather err toward the smaller batch.
_XXL_FULL_BATCH_VRAM_GB_THRESHOLD: float = 60.0
_XXL_SAFE_BATCH_SIZE: int = 4


def _resolve_xxl_batch_size(
    model_id: str | None,
    configured_batch_size: int,
    vram_gb: float | None,
) -> int:
    """Auto-downshift XXL COMET batch_size on <60 GiB cards.

    Rationale: the catalogue defaults XXL COMET/XCOMET to batch=8 (paper
    recommendation for 80 GiB A100). On a 48 GiB L40s / RTX8000 the
    pytorch-lightning predict step OOMs at batch=8 with XCOMET-XXL and
    CometKiwi-XXL; batch=4 is safe. Instead of forcing users to override
    batch_size per partition, we detect the real VRAM at runtime and step
    down to 4 transparently. Users on A100 80 GiB get the full batch=8.

    * Non-XXL models are returned unchanged.
    * If the user already asked for batch<=4, we respect that (don't upshift).
    * If VRAM detection fails (CPU-only, no torch, weird runtime), we trust
      the config — no surprise downshifts.
    """
    if not _is_xxl(model_id):
        return configured_batch_size
    if configured_batch_size <= _XXL_SAFE_BATCH_SIZE:
        return configured_batch_size
    if vram_gb is None:
        return configured_batch_size
    if vram_gb >= _XXL_FULL_BATCH_VRAM_GB_THRESHOLD:
        return configured_batch_size
    return _XXL_SAFE_BATCH_SIZE


class UnsupportedMarianCheckpointError(RuntimeError):
    """Raised when a COMET model id resolves to a Marian-format checkpoint
    that modern unbabel-comet cannot load.

    ``download_model()`` hardcodes ``<snapshot>/checkpoints/model.ckpt`` but
    the three ``-marian`` HF repos ship ``<snapshot>/checkpoints/marian.model.bin``
    instead, so ``load_from_checkpoint()`` would otherwise raise a confusing
    ``Invalid checkpoint path`` error. This wrapper swaps in a message that
    names the real root cause.
    """


def _raise_if_marian_layout(model_id: str, ckpt_path: "Path") -> None:  # noqa: F821 — Path imported at call site
    """Preflight: if download_model returned a model.ckpt path that doesn't
    exist but the snapshot holds a marian.model.bin alongside, the repo is a
    Marian-format checkpoint that modern unbabel-comet can't load."""
    if ckpt_path.is_file():
        return
    marian_sibling = ckpt_path.with_name("marian.model.bin")
    if not marian_sibling.is_file():
        return
    raise UnsupportedMarianCheckpointError(
        f"COMET model {model_id!r} is a Marian-format checkpoint "
        f"({marian_sibling}) — modern unbabel-comet only loads "
        f"pytorch-lightning .ckpt files from <snapshot>/checkpoints/model.ckpt. "
        f"Remove it from your catalogue / run config, or pin to an older "
        f"unbabel-comet that still had a Marian loader."
    )


def _infer_needs_reference(model_id: str) -> bool:
    """Heuristic: Kiwi variants are QE (no ref); ``-qe-`` / trailing ``-qe``
    marks a QE variant; everything else is reference-based.

    XCOMET accepts both — treated as reference-using by default; set
    ``needs_reference: false`` in the catalogue entry to run in QE mode.

    The catalogue should be authoritative — this heuristic is only the
    fallback when a scorer is constructed without a ``needs_reference``
    param. It used to buggily split on the bare token ``qe`` and then look
    for ``da`` in the suffix, which flipped ``-qe-da`` and ``-qe-mqm-marian``
    to the wrong class.
    """
    lower = model_id.lower()
    if "kiwi" in lower:
        return False
    if "-qe-" in lower or lower.endswith("-qe"):
        return False
    return True


def _factory(cfg: ScorerConfig) -> CometScorer:
    return CometScorer(cfg)


register_scorer("comet", _factory)
