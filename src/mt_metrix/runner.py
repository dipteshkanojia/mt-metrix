"""Run orchestration.

A Runner takes a :class:`RunConfig`, loads the dataset, instantiates each
scorer via the registry, runs them serially with load/unload around each
one, and writes the outputs.

Key decisions:

- Scorers are loaded/unloaded one at a time so GPU memory doesn't stack up
  (critical when chaining COMET-XXL → Tower-13B in one job).
- Reference-needing scorers are auto-skipped when the dataset has no
  references, with a visible warning and an entry in ``summary.json``'s
  ``skipped_metrics``.
- Parse / OOM errors inside a scorer are caught and recorded per-scorer so
  one bad metric doesn't nuke the whole run.
- Outputs are persisted after **every** scorer (both success and skip) and
  once before the loop. A mid-loop kill (host-RAM OOM, wall-clock,
  SIGTERM) then still leaves a coherent ``segments.tsv`` + ``summary.json``
  reflecting everything scored up to that point, so ``mt-metrix tabulate``
  can aggregate partial runs.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mt_metrix.config import RunConfig, dump_resolved_config
from mt_metrix.io.datasets import load_dataset_from_config
from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.io.writers import write_segments_jsonl, write_segments_tsv, write_summary
from mt_metrix.logging_utils import setup_logging
from mt_metrix.scorers.base import Scorer
from mt_metrix.scorers.registry import _bootstrap, build_scorer

log = logging.getLogger(__name__)


def run(config: RunConfig) -> Path:
    """Execute a run, return the output directory."""
    _bootstrap()

    out_dir = config.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=out_dir / "run.log")
    log.info("mt-metrix run: %s", config.run_id)
    log.info("output dir: %s", out_dir)

    dump_resolved_config(config, out_dir / "config.yaml")

    # --- dataset ---
    t0 = time.time()
    log.info("loading dataset (%s)", config.dataset.kind)
    segments: list[Segment] = load_dataset_from_config(config.dataset)
    log.info("loaded %d segments in %.1fs", len(segments), time.time() - t0)
    has_references = all(s.has_reference() for s in segments)
    has_any_references = any(s.has_reference() for s in segments)
    if has_any_references and not has_references:
        log.warning(
            "partial references detected (%d/%d segments have refs) — "
            "reference-based metrics will score only the subset that has refs",
            sum(1 for s in segments if s.has_reference()),
            len(segments),
        )

    # --- scorers ---
    scores_by_name: dict[str, list[SegmentScore]] = {}
    corpus_scores: dict[str, Any] = {}
    skipped_metrics: list[dict[str, Any]] = []

    # Build metadata once — shelling out to `git rev-parse` on every
    # incremental persist would be wasteful.
    run_metadata = _build_metadata(config)

    # Seed an empty snapshot up-front so even a crash in the first scorer
    # still leaves a parseable summary.json on disk.
    _persist(
        out_dir, config, segments, scores_by_name, skipped_metrics,
        corpus_scores, run_metadata,
    )

    for scorer_cfg in config.scorers:
        name = scorer_cfg.name
        try:
            scorer: Scorer = build_scorer(scorer_cfg)
        except Exception as e:
            log.error("could not build scorer %s: %s", name, e)
            skipped_metrics.append({"name": name, "reason": f"build-failed: {e}"})
            _persist(
                out_dir, config, segments, scores_by_name, skipped_metrics,
                corpus_scores, run_metadata,
            )
            continue

        if scorer.needs_reference and not has_any_references:
            log.warning(
                "skipping %s — it needs references but the dataset has none",
                name,
            )
            skipped_metrics.append(
                {"name": name, "reason": "dataset-has-no-references"}
            )
            _persist(
                out_dir, config, segments, scores_by_name, skipped_metrics,
                corpus_scores, run_metadata,
            )
            continue

        # pick the subset of segments with references for ref-needing scorers
        if scorer.needs_reference:
            usable = [s for s in segments if s.has_reference()]
        else:
            usable = segments

        log.info(
            "running %s (family=%s) on %d segments", name, scorer.family, len(usable)
        )
        t_scorer = time.time()
        try:
            scorer.load()
            results = scorer.score(usable)
        except Exception as e:
            log.exception("scorer %s failed during load/score: %s", name, e)
            skipped_metrics.append({"name": name, "reason": f"runtime: {e}"})
            try:
                scorer.unload()
            except Exception:  # pragma: no cover
                pass
            _persist(
                out_dir, config, segments, scores_by_name, skipped_metrics,
                corpus_scores, run_metadata,
            )
            continue
        finally:
            try:
                scorer.unload()
            except Exception as e:  # pragma: no cover
                log.warning("scorer %s unload raised: %s", name, e)

        # If we scored only a subset (ref-needing on a partially-ref dataset),
        # align results back to the full segments list by segment_id.
        if len(results) != len(segments):
            by_id = {r.segment_id: r for r in results}
            aligned: list[SegmentScore] = []
            for seg in segments:
                if seg.segment_id in by_id:
                    aligned.append(by_id[seg.segment_id])
                else:
                    aligned.append(
                        SegmentScore(
                            segment_id=seg.segment_id,
                            score=float("nan"),
                            extra={"skipped": "no-reference"},
                        )
                    )
            results = aligned

        scores_by_name[name] = results

        # Collect corpus-level payload if the scorer exposed one via its extras
        corpus_payload = getattr(scorer, "corpus_score", None)
        if corpus_payload is not None:
            corpus_scores[name] = corpus_payload

        log.info("  %s done in %.1fs", name, time.time() - t_scorer)

        # Persist after every successful scorer — a mid-loop OOM in a
        # LATER scorer still leaves this one's results on disk.
        _persist(
            out_dir, config, segments, scores_by_name, skipped_metrics,
            corpus_scores, run_metadata,
        )

    log.info("run complete: %s", out_dir)
    return out_dir


def _persist(
    out_dir: Path,
    config: RunConfig,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
    skipped_metrics: list[dict[str, Any]],
    corpus_scores: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    """Snapshot current run state to disk.

    Called before the scorer loop and after every scorer iteration — each
    call fully rewrites the three output files with the current in-memory
    state, so a mid-loop crash leaves a coherent partial output that
    ``mt-metrix tabulate`` can aggregate like any other run.
    """
    formats = set(config.output.formats)
    if "tsv" in formats:
        write_segments_tsv(out_dir / "segments.tsv", segments, scores_by_name)
    if "jsonl" in formats:
        write_segments_jsonl(out_dir / "segments.jsonl", segments, scores_by_name)
    if "summary" in formats:
        write_summary(
            out_dir / "summary.json",
            segments,
            scores_by_name,
            skipped_metrics=skipped_metrics,
            run_metadata=run_metadata,
            corpus_scores=corpus_scores,
        )


def _build_metadata(config: RunConfig) -> dict[str, Any]:
    md: dict[str, Any] = {
        "run_id": config.run_id,
        "python": sys.version.split()[0],
        "scorers": [
            {"family": s.family, "name": s.name, "model": s.model, "params": s.params}
            for s in config.scorers
        ],
        "dataset": {"kind": config.dataset.kind, **config.dataset.params},
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        md["git_sha"] = git_sha
    except Exception:
        md["git_sha"] = None

    try:
        import torch

        md["torch"] = torch.__version__
        md["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            md["gpu_count"] = torch.cuda.device_count()
            md["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        md["torch"] = None

    return md
