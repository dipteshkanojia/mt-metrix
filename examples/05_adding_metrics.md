# 05 — Adding a new metric family

Scorers are plugins. Every family (`comet`, `tower`, `sacrebleu`, …) lives in
one module under `src/mt_metrix/scorers/` and registers itself with the
registry. The runner is family-agnostic.

## The contract

A scorer implements the `Scorer` protocol from
`src/mt_metrix/scorers/base.py`:

```python
class Scorer(Protocol):
    config: ScorerConfig
    name: str
    family: str
    needs_reference: bool

    def load(self) -> None: ...
    def score(self, segments: list[Segment]) -> list[SegmentScore]: ...
    def unload(self) -> None: ...
```

`load` / `unload` frame expensive setup (model download, GPU allocation) so
the runner can chain multiple scorers in one job without memory stacking.
`needs_reference` drives the auto-skip behaviour when a dataset has no
reference column.

## Walkthrough: adding OpenKiwi

1. **Create the module** — `src/mt_metrix/scorers/openkiwi.py`:

```python
"""OpenKiwi scorer — pre-COMET QE baselines (Kepler et al. 2019)."""
from __future__ import annotations

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.scorers.base import ScorerConfig
from mt_metrix.scorers.registry import register_scorer


class OpenKiwiScorer:
    def __init__(self, cfg: ScorerConfig) -> None:
        self._cfg = cfg
        self._model = None

    @property
    def config(self) -> ScorerConfig: return self._cfg
    @property
    def name(self) -> str: return self._cfg.name
    @property
    def family(self) -> str: return "openkiwi"
    @property
    def needs_reference(self) -> bool: return False  # QE is reference-free

    def load(self) -> None:
        import kiwi
        self._model = kiwi.load_model(self._cfg.model)

    def score(self, segments: list[Segment]) -> list[SegmentScore]:
        assert self._model is not None
        raw = self._model.evaluate([
            {"source": s.source, "target": s.target}
            for s in segments
        ])
        return [
            SegmentScore(segment_id=seg.segment_id, score=float(r["score"]))
            for seg, r in zip(segments, raw)
        ]

    def unload(self) -> None:
        self._model = None
        # free GPU memory
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _factory(cfg: ScorerConfig) -> OpenKiwiScorer:
    return OpenKiwiScorer(cfg)


register_scorer("openkiwi", _factory)
```

2. **Register at bootstrap** — `src/mt_metrix/scorers/registry.py`:

```python
def _bootstrap() -> None:
    for module_name in (
        "mt_metrix.scorers.comet",
        "mt_metrix.scorers.tower",
        "mt_metrix.scorers.sacrebleu_scorer",
        "mt_metrix.scorers.openkiwi",   # <-- add
    ):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            log.debug("skipping %s (%s)", module_name, e)
```

The `ImportError` tolerance means users who haven't installed `kiwi` still
get a working mt-metrix; only runs that explicitly reference `openkiwi/...`
will fail, with a clear `Unknown scorer family` error.

3. **Catalogue file** — `configs/models/openkiwi.yaml`:

```yaml
family: openkiwi
models:
  nuqe:
    model: Unbabel/openkiwi-nuqe
    notes: "OpenKiwi NuQE — historical WMT19 QE baseline."
    params:
      batch_size: 32
```

4. **Use it in a run config** — `configs/runs/surrey_legal_openkiwi.yaml`:

```yaml
run: {id: surrey_legal_openkiwi}
dataset: !include ../datasets/surrey_legal.yaml
scorers:
  - ref: openkiwi/nuqe
```

5. **Tests** — add `tests/test_scorers_openkiwi.py`. Mark slow tests that
require model downloads with `pytestmark = pytest.mark.slow`. The runner
test in `tests/test_runner_e2e.py` already covers the end-to-end wiring.

6. **Optional extras for pyproject** — declare the dependency as an optional
extra so the core install stays lean:

```toml
[project.optional-dependencies]
openkiwi = ["openkiwi>=0.3"]
```

## Producing rich extras

The scorer's per-segment `SegmentScore.extra` dict flows straight into
`segments.jsonl` under `scores.<name>.extra`. Use it for anything beyond a
single float:

```python
SegmentScore(
    segment_id=seg.segment_id,
    score=float(result.score),
    extra={
        "word_tags": result.word_tags,          # list of 0/1 tags
        "confidence": result.confidence_interval,
        "model_version": self._model.version,
    },
)
```

Downstream analysis reads `segments.jsonl` and pulls whatever fields it
needs.

## Corpus-level payloads

If your metric has a meaningful corpus-level score (BLEU corpus score,
average-per-domain, …), expose it as `self.corpus_score` on the scorer after
`score()` returns. The runner reads this and writes it to
`summary.json::corpus_scores::<scorer_name>`. See the pattern in
`src/mt_metrix/scorers/sacrebleu_scorer.py`.

## Smoke-testing the plugin

```bash
pytest tests/test_scorers_openkiwi.py -v
pytest tests/test_runner_e2e.py -v     # confirms registry still works end-to-end

MT_METRIX_RUN_SLOW=1 pytest tests/test_scorers_openkiwi.py -v    # full integration
```
