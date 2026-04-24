# Prediction-Space Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route QE-scorer correlations to the ground-truth column that matches each scorer's prediction space (DA-trained scorers correlate against raw `mean`; z-DA-trained scorers correlate against `z_mean`), with a bold stderr warning when the matching gold column is absent and we fall back.

**Architecture:** Extend `Segment` to carry two optional gold columns (`gold_raw`, `gold_z`). Dataset YAML gets new flat keys `gold_raw:` and `gold_z:` (legacy singular `gold:` continues to load into `gold_raw` with a deprecation log). Scorer catalogue entries gain a `prediction_space: Literal["raw_da", "z_da"]` flag, promoted into `ScorerConfig.params` by the catalogue resolver (same pattern as `needs_reference`). `write_summary` and `_per_lang_correlations` look up each scorer's space and pick the matching gold column; when absent, they fall back to the other column and print a bold stderr warning plus a `log.warning`.

**Tech Stack:** Python 3.10, pandas, scipy.stats, pytest, PyYAML.

---

## File Structure

**Code changes:**
- `src/mt_metrix/io/schema.py` — Add `gold_raw`, `gold_z`, `PredictionSpace` literal, and backward-compat `gold` property.
- `src/mt_metrix/io/datasets.py` — Parse new column keys in `_row_to_segment`; emit deprecation log for legacy `gold:`.
- `src/mt_metrix/io/writers.py` — Emit both gold columns in TSV/JSONL; accept `scorer_spaces` parameter in `write_summary` and route correlations per scorer.
- `src/mt_metrix/config.py` — Promote `prediction_space` from catalogue entry into `ScorerConfig.params` in `_resolve_scorer_entry`.
- `src/mt_metrix/runner.py` — Build `scorer_spaces` map from resolved scorers, pass to `write_summary`.
- `src/mt_metrix/reports/tabulate.py` — Look up each scorer's prediction space; pick matching gold column; bold-warn on fallback.

**Config migrations:**
- `configs/datasets/surrey_legal.yaml`, `surrey_general.yaml`, `surrey_health.yaml`, `surrey_tourism.yaml`, `surrey_lowres_qe.yaml` — Replace `gold: z_mean` with `gold_raw: mean` + `gold_z: z_mean`.
- `configs/models/comet.yaml` — Add `prediction_space: z_da` to every Unbabel COMET/CometKiwi/XCOMET entry (they're all trained on z-DA).
- `configs/models/tower.yaml` — Add `prediction_space: raw_da` to every Tower entry (GEMBA-DA + GEMBA-MQM both emit 0–100 raw).
- `configs/models/sacrebleu.yaml` — Add `prediction_space: raw_da` to each entry.

**Tests:**
- `tests/test_schema.py` — Segment with both gold columns; `gold` property fallback; missing fields default to None.
- `tests/test_datasets.py` — Load rows with `gold_raw: X, gold_z: Y` column map; legacy `gold:` with deprecation capture.
- `tests/test_writers.py` — TSV/JSONL have both columns; summary correlations route correctly; fallback path verified.
- `tests/test_config.py` — `prediction_space` propagation from catalogue; default when absent; overrides.
- `tests/test_tabulate.py` — Per-scorer routing; bold-stderr warning captured via `capsys`; legacy `gold` column still works.
- `tests/fixtures/tiny_both_gold.tsv` (NEW) — 10-row fixture with `gold_raw` + `gold_z`.
- `tests/fixtures/tiny_run_both_gold.yaml` (NEW) — Run config pointing at new fixture.

---

### Task 1: Extend Segment with gold_raw + gold_z

**Files:**
- Modify: `src/mt_metrix/io/schema.py:13-33`
- Test: `tests/test_schema.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_schema.py`:

```python
from mt_metrix.io.schema import Segment, PredictionSpace


def test_segment_carries_both_gold_columns():
    s = Segment(source="hello", target="hola", gold_raw=72.5, gold_z=0.8)
    assert s.gold_raw == 72.5
    assert s.gold_z == 0.8


def test_segment_gold_property_prefers_z():
    """Backward compat: `seg.gold` returns gold_z if set, else gold_raw.

    Z-scored gold is the historic default for QE datasets, so existing
    callers that read `seg.gold` keep getting the same signal.
    """
    s_both = Segment(source="a", target="b", gold_raw=70.0, gold_z=0.5)
    assert s_both.gold == 0.5

    s_raw_only = Segment(source="a", target="b", gold_raw=70.0)
    assert s_raw_only.gold == 70.0

    s_z_only = Segment(source="a", target="b", gold_z=0.5)
    assert s_z_only.gold == 0.5

    s_neither = Segment(source="a", target="b")
    assert s_neither.gold is None


def test_segment_has_gold_still_works():
    assert not Segment(source="a", target="b").has_gold()
    assert Segment(source="a", target="b", gold_raw=1.0).has_gold()
    assert Segment(source="a", target="b", gold_z=1.0).has_gold()


def test_prediction_space_literal_values():
    """The PredictionSpace alias must accept the two values the rest of the
    code routes on. Anything else is a typo the type checker should catch."""
    assert PredictionSpace.__args__ == ("raw_da", "z_da")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schema.py -v -k "both_gold or gold_property or prediction_space"`
Expected: FAIL — `AttributeError: 'Segment' object has no attribute 'gold_raw'` (or import error on `PredictionSpace`).

- [ ] **Step 3: Implement in src/mt_metrix/io/schema.py**

Replace lines 10-33 with:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


PredictionSpace = Literal["raw_da", "z_da"]
"""Space a scorer predicts in.

- ``raw_da`` — raw Direct Assessment 0-100 (GEMBA-DA LLM prompt output,
  GEMBA-MQM derived 100 − error-penalty, sacrebleu reference-based scores).
- ``z_da`` — per-rater z-normalised DA (COMET-QE, CometKiwi, XCOMET
  training targets).

Add new spaces here — tabulate's gold-column resolver reads this alias.
"""


@dataclass
class Segment:
    """A single translation instance to be scored."""

    source: str
    target: str
    reference: Optional[str] = None
    gold_raw: Optional[float] = None
    gold_z: Optional[float] = None
    lang_pair: str = ""
    domain: str = "general"
    segment_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gold(self) -> Optional[float]:
        """Backward-compat accessor. Prefer ``gold_z`` over ``gold_raw``.

        Existing code paths that read ``seg.gold`` keep getting a z-scored
        value when available (the historic default for QE datasets). New
        code should read ``gold_raw`` / ``gold_z`` explicitly.
        """
        return self.gold_z if self.gold_z is not None else self.gold_raw

    def has_reference(self) -> bool:
        return self.reference is not None and len(self.reference) > 0

    def has_gold(self) -> bool:
        return self.gold_raw is not None or self.gold_z is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schema.py -v`
Expected: PASS (all schema tests, including the 4 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/mt_metrix/io/schema.py tests/test_schema.py
git commit -m "schema: add gold_raw + gold_z fields and PredictionSpace alias"
```

---

### Task 2: Dataset loader parses gold_raw/gold_z, deprecates singular gold

**Files:**
- Modify: `src/mt_metrix/io/datasets.py:47-84` (`_row_to_segment`)
- Test: `tests/test_datasets.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_datasets.py`:

```python
import logging
import pytest

from mt_metrix.io.datasets import _row_to_segment


def test_row_to_segment_both_gold_columns():
    row = {"src": "hi", "tgt": "hola", "raw": "72.5", "z": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 72.5
    assert seg.gold_z == 0.8


def test_row_to_segment_only_gold_raw():
    row = {"src": "hi", "tgt": "hola", "raw": "72.5"}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 72.5
    assert seg.gold_z is None


def test_row_to_segment_only_gold_z():
    row = {"src": "hi", "tgt": "hola", "z": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw is None
    assert seg.gold_z == 0.8


def test_row_to_segment_legacy_gold_loads_into_gold_raw_with_warning(caplog):
    """Legacy `gold: z_mean` column maps MUST continue to work but should
    emit a deprecation-warning log so users know to migrate. It routes into
    gold_raw by default (conservative: unknown-space → raw)."""
    row = {"src": "hi", "tgt": "hola", "g": "0.8"}
    columns = {"source": "src", "target": "tgt", "gold": "g"}
    with caplog.at_level(logging.WARNING, logger="mt_metrix.io.datasets"):
        seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw == 0.8
    assert seg.gold_z is None
    assert any("legacy `gold:` column key" in r.message for r in caplog.records)


def test_row_to_segment_invalid_gold_values_become_none():
    row = {"src": "hi", "tgt": "hola", "raw": "n/a", "z": ""}
    columns = {"source": "src", "target": "tgt", "gold_raw": "raw", "gold_z": "z"}
    seg = _row_to_segment(row, columns, idx=0, default_lang_pair="en-es", default_domain="news")
    assert seg.gold_raw is None
    assert seg.gold_z is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasets.py -v -k "row_to_segment"`
Expected: FAIL — loader currently only reads `columns.get("gold")` and writes to `Segment.gold`, which is now a read-only property.

- [ ] **Step 3: Implement loader change**

Replace lines 47-84 in `src/mt_metrix/io/datasets.py` with:

```python
def _resolve_gold(
    row: dict[str, Any],
    columns: dict[str, str],
    *,
    key: str,
) -> float | None:
    """Resolve one gold column (``gold_raw`` or ``gold_z``) to a float.

    Invalid / missing / non-numeric values become ``None`` so downstream
    correlation code can mask them out.
    """
    raw = _resolve_column(row, columns.get(key))
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _row_to_segment(
    row: dict[str, Any],
    columns: dict[str, str],
    idx: int,
    default_lang_pair: str,
    default_domain: str,
) -> Segment:
    source = _resolve_column(row, columns.get("source"))
    target = _resolve_column(row, columns.get("target"))
    if not source or not target:
        raise ValueError(
            f"row {idx} missing source or target (after column mapping): {row}"
        )

    reference = _resolve_column(row, columns.get("reference"))

    gold_raw = _resolve_gold(row, columns, key="gold_raw")
    gold_z = _resolve_gold(row, columns, key="gold_z")

    # Backward compat: legacy singular `gold:` key still accepted. We route
    # it into gold_raw (the conservative default) and emit a one-shot-per-
    # dataset deprecation warning so users migrate their configs to the
    # explicit form.
    if "gold" in columns and gold_raw is None and gold_z is None:
        log.warning(
            "dataset config uses legacy `gold:` column key (mapping column %r); "
            "prefer explicit `gold_raw:` and/or `gold_z:` — this row's value "
            "is being loaded as gold_raw. See docs/DATASETS.md.",
            columns["gold"],
        )
        gold_raw = _resolve_gold(row, columns, key="gold")

    lang_pair = _resolve_column(row, columns.get("lang_pair")) or default_lang_pair
    domain = _resolve_column(row, columns.get("domain")) or default_domain
    segment_id_raw = _resolve_column(row, columns.get("segment_id"))
    segment_id = str(segment_id_raw) if segment_id_raw is not None else f"seg_{idx:08d}"

    return Segment(
        source=str(source),
        target=str(target),
        reference=str(reference) if reference else None,
        gold_raw=gold_raw,
        gold_z=gold_z,
        lang_pair=str(lang_pair),
        domain=str(domain),
        segment_id=segment_id,
        meta=row,
    )
```

- [ ] **Step 4: Run all dataset tests**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS (new tests + any existing that survive the schema change).

- [ ] **Step 5: Commit**

```bash
git add src/mt_metrix/io/datasets.py tests/test_datasets.py
git commit -m "datasets: parse gold_raw/gold_z; deprecate legacy gold: key"
```

---

### Task 3: Writers emit both gold columns in TSV + JSONL

**Files:**
- Modify: `src/mt_metrix/io/writers.py:14-73` (`write_segments_tsv` + `write_segments_jsonl`)
- Test: `tests/test_writers.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_writers.py`:

```python
import json

import pandas as pd

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.io.writers import write_segments_jsonl, write_segments_tsv


def test_segments_tsv_has_both_gold_columns(tmp_path):
    segs = [
        Segment(source="a", target="b", gold_raw=70.0, gold_z=0.5, segment_id="s1"),
        Segment(source="c", target="d", gold_raw=80.0, segment_id="s2"),  # no z
        Segment(source="e", target="f", gold_z=-1.0, segment_id="s3"),    # no raw
        Segment(source="g", target="h", segment_id="s4"),                  # neither
    ]
    path = tmp_path / "segments.tsv"
    write_segments_tsv(path, segs, scores_by_name={})
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df["gold_raw"]) == ["70.0", "80.0", "", ""]
    assert list(df["gold_z"]) == ["0.5", "", "-1.0", ""]


def test_segments_jsonl_has_both_gold_fields(tmp_path):
    segs = [
        Segment(source="a", target="b", gold_raw=70.0, gold_z=0.5, segment_id="s1"),
        Segment(source="c", target="d", gold_raw=80.0, segment_id="s2"),
    ]
    path = tmp_path / "segments.jsonl"
    write_segments_jsonl(path, segs, scores_by_name={})
    lines = path.read_text().strip().splitlines()
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["gold_raw"] == 70.0 and r1["gold_z"] == 0.5
    assert r2["gold_raw"] == 80.0 and r2["gold_z"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_writers.py -v -k "both_gold"`
Expected: FAIL — current writer only emits a single `gold` column.

- [ ] **Step 3: Implement writer changes**

In `src/mt_metrix/io/writers.py`, replace lines 14-45 (`write_segments_tsv`) with:

```python
def write_segments_tsv(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
) -> None:
    """Write a flat TSV with one row per segment and one column per metric.

    Columns: ``segment_id, lang_pair, domain, source, target, reference,
    gold_raw, gold_z, <metric1>, <metric2>, ...``.

    Backward-compat note: the single ``gold`` column we used to emit is
    replaced by ``gold_raw`` and ``gold_z``. Tabulate and any downstream
    analysis reads both; the per-scorer prediction_space decides which.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        row: dict[str, Any] = {
            "segment_id": seg.segment_id,
            "lang_pair": seg.lang_pair,
            "domain": seg.domain,
            "source": seg.source,
            "target": seg.target,
            "reference": seg.reference or "",
            "gold_raw": "" if seg.gold_raw is None else seg.gold_raw,
            "gold_z": "" if seg.gold_z is None else seg.gold_z,
        }
        for name, col in scores_by_name.items():
            if i < len(col):
                val = col[i].score
                row[name] = "" if val is None else val
            else:
                row[name] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
```

And replace lines 48-73 (`write_segments_jsonl`) with:

```python
def write_segments_jsonl(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
) -> None:
    """Write one JSON object per line with all scorer extras included."""
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            obj: dict[str, Any] = {
                "segment_id": seg.segment_id,
                "lang_pair": seg.lang_pair,
                "domain": seg.domain,
                "source": seg.source,
                "target": seg.target,
                "reference": seg.reference,
                "gold_raw": seg.gold_raw,
                "gold_z": seg.gold_z,
                "scores": {},
            }
            for name, col in scores_by_name.items():
                if i < len(col):
                    obj["scores"][name] = {
                        "score": col[i].score,
                        "extra": col[i].extra,
                    }
            f.write(json.dumps(obj, ensure_ascii=False, default=_json_default))
            f.write("\n")
```

- [ ] **Step 4: Run writer tests**

Run: `pytest tests/test_writers.py -v`
Expected: PASS (new tests + existing writer tests; any existing that assumed singular `gold` must be updated in Step 5 alongside this commit).

- [ ] **Step 5: Commit**

```bash
git add src/mt_metrix/io/writers.py tests/test_writers.py
git commit -m "writers: emit gold_raw + gold_z in TSV and JSONL"
```

---

### Task 4: Scorer catalogue carries prediction_space flag

**Files:**
- Modify: `src/mt_metrix/config.py:180-203` (`_resolve_scorer_entry`, the `for flag in (…)` loop)
- Test: `tests/test_config.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
from mt_metrix.config import _resolve_scorer_entry


def test_prediction_space_promoted_from_catalogue():
    catalogue = {
        "comet/wmt22-cometkiwi-da": {
            "family": "comet",
            "name": "wmt22-cometkiwi-da",
            "model": "Unbabel/wmt22-cometkiwi-da",
            "needs_reference": False,
            "prediction_space": "z_da",
            "params": {"batch_size": 64},
        }
    }
    cfg = _resolve_scorer_entry({"ref": "comet/wmt22-cometkiwi-da"}, catalogue)
    assert cfg.params["prediction_space"] == "z_da"


def test_prediction_space_defaults_to_raw_da_when_absent():
    """Catalogue entries without an explicit prediction_space get the
    conservative default 'raw_da'. This keeps legacy entries working without
    forcing every catalogue to be migrated atomically."""
    catalogue = {
        "tower/some-untagged-scorer": {
            "family": "tower",
            "name": "some-untagged-scorer",
            "model": "Unbabel/Whatever",
            "params": {},
        }
    }
    cfg = _resolve_scorer_entry({"ref": "tower/some-untagged-scorer"}, catalogue)
    assert cfg.params["prediction_space"] == "raw_da"


def test_prediction_space_can_be_overridden():
    catalogue = {
        "comet/wmt22-cometkiwi-da": {
            "family": "comet",
            "name": "wmt22-cometkiwi-da",
            "model": "Unbabel/wmt22-cometkiwi-da",
            "prediction_space": "z_da",
            "params": {},
        }
    }
    cfg = _resolve_scorer_entry(
        {"ref": "comet/wmt22-cometkiwi-da", "overrides": {"prediction_space": "raw_da"}},
        catalogue,
    )
    assert cfg.params["prediction_space"] == "raw_da"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v -k "prediction_space"`
Expected: FAIL — `KeyError: 'prediction_space'` or similar.

- [ ] **Step 3: Implement in src/mt_metrix/config.py**

In `_resolve_scorer_entry` (around lines 195-197), replace the flag-promotion loop with:

```python
        # Promote catalogue top-level flags that scorers read from params. Without
        # this, a catalogue entry like `needs_reference: false` at the entry's
        # top level (the documented schema) would be silently dropped, and COMET
        # scorers would fall back to the model-id heuristic — which
        # misclassified `Unbabel/XCOMET-XL` (QE mode) and `wmt20-comet-qe-da`
        # during the 2026-04-19 full-matrix run.
        for flag in ("needs_reference", "prediction_space"):
            if flag in base and flag not in overrides:
                params.setdefault(flag, base[flag])

        # Prediction-space default: if the catalogue didn't declare it, assume
        # raw_da (the conservative default — matches sacrebleu and GEMBA-DA
        # Tower prompts). Downstream tabulate routes z_da scorers to gold_z
        # and raw_da scorers to gold_raw; see docs/PREDICTION_SPACES.md.
        params.setdefault("prediction_space", "raw_da")
```

- [ ] **Step 4: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mt_metrix/config.py tests/test_config.py
git commit -m "config: promote prediction_space from catalogue into ScorerConfig.params"
```

---

### Task 5: Summary writer routes correlations per scorer

**Files:**
- Modify: `src/mt_metrix/io/writers.py:76-132` (`write_summary`)
- Modify: `src/mt_metrix/runner.py` (call site — build scorer_spaces map)
- Test: `tests/test_writers.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_writers.py`:

```python
import json
import sys

from mt_metrix.io.schema import Segment, SegmentScore
from mt_metrix.io.writers import write_summary


def _write_summary_and_load(tmp_path, segs, scores_by_name, scorer_spaces):
    path = tmp_path / "summary.json"
    write_summary(
        path=path,
        segments=segs,
        scores_by_name=scores_by_name,
        skipped_metrics=[],
        run_metadata={},
        scorer_spaces=scorer_spaces,
    )
    return json.loads(path.read_text())


def test_summary_routes_correlations_by_prediction_space(tmp_path):
    """Per-scorer correlations use the gold column matching the scorer's
    prediction_space.

    The fixture is rigged so raw vs z golds are strictly anti-correlated:
    a scorer that predicts the raw gold perfectly gets a Pearson of +1.0
    against gold_raw and −1.0 against gold_z. Routing matters and is
    testable on 4 rows.
    """
    segs = [
        Segment(source="a", target="b", gold_raw=10.0, gold_z=-1.0, segment_id="s1"),
        Segment(source="c", target="d", gold_raw=20.0, gold_z=-0.5, segment_id="s2"),
        Segment(source="e", target="f", gold_raw=30.0, gold_z= 0.5, segment_id="s3"),
        Segment(source="g", target="h", gold_raw=40.0, gold_z= 1.0, segment_id="s4"),
    ]
    # Perfect predictor of raw gold
    raw_preds = [SegmentScore(segment_id=s.segment_id, score=s.gold_raw) for s in segs]
    # Perfect predictor of z gold
    z_preds = [SegmentScore(segment_id=s.segment_id, score=s.gold_z) for s in segs]

    payload = _write_summary_and_load(
        tmp_path,
        segs,
        {"raw_scorer": raw_preds, "z_scorer": z_preds},
        scorer_spaces={"raw_scorer": "raw_da", "z_scorer": "z_da"},
    )

    assert payload["metrics"]["raw_scorer"]["correlation_vs_gold"]["pearson"] == 1.0
    assert payload["metrics"]["z_scorer"]["correlation_vs_gold"]["pearson"] == 1.0
    # Sanity: the gold column used is recorded so downstream analysis can
    # cite it. A cross-space correlation would not produce +1.0 here.
    assert payload["metrics"]["raw_scorer"]["correlation_vs_gold"]["gold_column"] == "gold_raw"
    assert payload["metrics"]["z_scorer"]["correlation_vs_gold"]["gold_column"] == "gold_z"


def test_summary_falls_back_with_bold_warning_when_matching_gold_absent(tmp_path, capsys):
    """z_da scorer on a dataset with only gold_raw must fall back to
    gold_raw and emit a BOLD stderr warning + log.warning."""
    segs = [
        Segment(source="a", target="b", gold_raw=10.0, segment_id="s1"),
        Segment(source="c", target="d", gold_raw=20.0, segment_id="s2"),
        Segment(source="e", target="f", gold_raw=30.0, segment_id="s3"),
    ]
    preds = [SegmentScore(segment_id=s.segment_id, score=s.gold_raw) for s in segs]

    payload = _write_summary_and_load(
        tmp_path,
        segs,
        {"z_scorer_no_z_gold": preds},
        scorer_spaces={"z_scorer_no_z_gold": "z_da"},
    )
    captured = capsys.readouterr()
    # Bold ANSI + explicit PREDICTION-SPACE FALLBACK label
    assert "\033[1;33m" in captured.err
    assert "PREDICTION-SPACE FALLBACK" in captured.err
    assert "z_scorer_no_z_gold" in captured.err
    # Correlation still computed — just against the fallback column
    assert payload["metrics"]["z_scorer_no_z_gold"]["correlation_vs_gold"]["pearson"] == 1.0
    assert payload["metrics"]["z_scorer_no_z_gold"]["correlation_vs_gold"]["gold_column"] == "gold_raw"
    assert payload["metrics"]["z_scorer_no_z_gold"]["correlation_vs_gold"]["fallback"] is True


def test_summary_no_gold_at_all_is_not_an_error(tmp_path):
    """If neither gold column is present, correlation_vs_gold is None and
    no warning fires. (This is the existing behaviour — not a regression.)"""
    segs = [
        Segment(source="a", target="b", segment_id="s1"),
        Segment(source="c", target="d", segment_id="s2"),
    ]
    preds = [SegmentScore(segment_id=s.segment_id, score=1.0) for s in segs]

    payload = _write_summary_and_load(
        tmp_path, segs, {"some_scorer": preds},
        scorer_spaces={"some_scorer": "raw_da"},
    )
    assert payload["metrics"]["some_scorer"]["correlation_vs_gold"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_writers.py -v -k "summary_"`
Expected: FAIL — `write_summary` does not accept `scorer_spaces` and does not route by space.

- [ ] **Step 3: Implement write_summary change**

In `src/mt_metrix/io/writers.py`, replace lines 76-132 with:

```python
# ANSI bold yellow prefix for the prediction-space fallback banner. Printed
# to stderr in addition to log.warning so the user sees it even when logs
# are silenced.
_FALLBACK_BOLD = "\033[1;33m"
_FALLBACK_RESET = "\033[0m"


def _pick_gold_column(
    scorer_name: str,
    prediction_space: str,
    has_gold_raw: bool,
    has_gold_z: bool,
) -> tuple[str | None, bool]:
    """Return (chosen_column, is_fallback).

    ``chosen_column`` is ``"gold_raw"``, ``"gold_z"``, or ``None`` if no
    gold column exists. ``is_fallback`` is True when the chosen column
    does NOT match the scorer's prediction_space (caller must warn).
    """
    preferred = {"raw_da": "gold_raw", "z_da": "gold_z"}.get(prediction_space, "gold_raw")
    present = {"gold_raw": has_gold_raw, "gold_z": has_gold_z}
    if present.get(preferred):
        return preferred, False
    # Fallback: pick the other column if available
    other = "gold_z" if preferred == "gold_raw" else "gold_raw"
    if present.get(other):
        return other, True
    return None, False


def _emit_fallback_warning(
    scorer_name: str,
    declared_space: str,
    preferred_col: str,
    fallback_col: str,
) -> None:
    """Print a bold stderr warning + log.warning for a space-mismatch fallback.

    The stderr write uses ANSI bold-yellow so it's hard to miss even when
    scrolling through a long run log.
    """
    banner = (
        f"{_FALLBACK_BOLD}⚠️  PREDICTION-SPACE FALLBACK{_FALLBACK_RESET}: "
        f"scorer {scorer_name!r} declared prediction_space={declared_space!r} "
        f"but dataset has no {preferred_col!r} column — falling back to "
        f"{fallback_col!r}. Rank correlations remain valid; absolute-score "
        f"interpretations will not.\n"
    )
    import sys
    sys.stderr.write(banner)
    log.warning(
        "prediction-space fallback: scorer=%s declared=%s preferred=%s "
        "fallback=%s",
        scorer_name, declared_space, preferred_col, fallback_col,
    )


def write_summary(
    path: Path,
    segments: list[Segment],
    scores_by_name: dict[str, list[SegmentScore]],
    skipped_metrics: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    corpus_scores: dict[str, Any] | None = None,
    scorer_spaces: dict[str, str] | None = None,
) -> None:
    """Write a one-file summary JSON with aggregates and correlations.

    Per-scorer ``correlation_vs_gold`` is computed against the gold column
    matching that scorer's ``prediction_space`` (from ``scorer_spaces``). If
    the matching column is absent on this dataset, the writer falls back to
    the other column AND emits a bold stderr warning. See
    ``docs/PREDICTION_SPACES.md``.
    """
    import numpy as np

    scorer_spaces = scorer_spaces or {}

    gold_raw_arr = np.array(
        [s.gold_raw if s.gold_raw is not None else np.nan for s in segments],
        dtype=float,
    )
    gold_z_arr = np.array(
        [s.gold_z if s.gold_z is not None else np.nan for s in segments],
        dtype=float,
    )
    has_gold_raw = bool(np.isfinite(gold_raw_arr).any())
    has_gold_z = bool(np.isfinite(gold_z_arr).any())

    metrics_summary: dict[str, Any] = {}
    for name, col in scores_by_name.items():
        preds = np.array([c.score if c.score is not None else np.nan for c in col], dtype=float)
        finite = np.isfinite(preds)
        summary = {
            "n": int(finite.sum()),
            "n_nan": int((~finite).sum()),
            "mean": _safe_float(preds[finite].mean()) if finite.any() else None,
            "std": _safe_float(preds[finite].std()) if finite.any() else None,
            "min": _safe_float(preds[finite].min()) if finite.any() else None,
            "max": _safe_float(preds[finite].max()) if finite.any() else None,
        }

        declared_space = scorer_spaces.get(name, "raw_da")
        chosen_col, is_fallback = _pick_gold_column(
            name, declared_space, has_gold_raw, has_gold_z
        )

        if chosen_col is None:
            summary["correlation_vs_gold"] = None
        else:
            gold_arr = gold_raw_arr if chosen_col == "gold_raw" else gold_z_arr
            gold_mask = np.isfinite(gold_arr)
            usable = finite & gold_mask
            if usable.sum() >= 2:
                corr = _correlations(preds[usable], gold_arr[usable])
                corr["gold_column"] = chosen_col
                corr["fallback"] = is_fallback
                summary["correlation_vs_gold"] = corr
            else:
                summary["correlation_vs_gold"] = None

            if is_fallback:
                preferred = {"raw_da": "gold_raw", "z_da": "gold_z"}[declared_space]
                _emit_fallback_warning(name, declared_space, preferred, chosen_col)

        metrics_summary[name] = summary

    payload: dict[str, Any] = {
        "run_metadata": run_metadata,
        "metrics": metrics_summary,
        "corpus_scores": corpus_scores or {},
        "skipped_metrics": skipped_metrics,
        "n_segments": len(segments),
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
```

- [ ] **Step 4: Update runner to pass scorer_spaces**

In `src/mt_metrix/runner.py`, find the `write_summary(...)` call. Add the `scorer_spaces` kwarg by building it from the resolved scorers. Use this snippet near the other writer calls:

```python
    scorer_spaces = {
        s.name: s.params.get("prediction_space", "raw_da")
        for s in run.scorers
    }
    write_summary(
        path=out_dir / "summary.json",
        segments=segments,
        scores_by_name=scores_by_name,
        skipped_metrics=skipped_metrics,
        run_metadata=run_metadata,
        corpus_scores=corpus_scores,
        scorer_spaces=scorer_spaces,
    )
```

(If the runner call currently uses positional args, convert them to kwargs so the new kwarg doesn't break the signature mid-refactor.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_writers.py tests/test_runner_e2e.py -v`
Expected: PASS. If `test_runner_e2e.py` fails because fixtures use the legacy `gold:` column key, update those fixtures to `gold_raw:` in Task 8 (keep failures red until then is acceptable — note the red test in your commit message).

- [ ] **Step 6: Commit**

```bash
git add src/mt_metrix/io/writers.py src/mt_metrix/runner.py tests/test_writers.py
git commit -m "writers: per-scorer correlation routing + bold fallback warning"
```

---

### Task 6: Tabulate picks gold column per scorer

**Files:**
- Modify: `src/mt_metrix/reports/tabulate.py:164-203` (`_per_lang_correlations`)
- Modify: `src/mt_metrix/reports/tabulate.py` — `collect_records` to thread scorer spaces from the catalogue.
- Test: `tests/test_tabulate.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_tabulate.py`:

```python
import pandas as pd
import pytest

from mt_metrix.reports.tabulate import _per_lang_correlations


def test_per_lang_correlations_routes_by_prediction_space(tmp_path):
    """Given a segments.tsv with both gold_raw and gold_z, each scorer's
    correlation is computed against the column matching its declared space."""
    df = pd.DataFrame(
        {
            "segment_id": ["s1", "s2", "s3", "s4"],
            "lang_pair": ["en-gu"] * 4,
            "gold_raw": [10.0, 20.0, 30.0, 40.0],
            "gold_z":   [-1.0, -0.5, 0.5, 1.0],
            "raw_scorer": [10.0, 20.0, 30.0, 40.0],   # perfect vs gold_raw
            "z_scorer":   [-1.0, -0.5, 0.5, 1.0],     # perfect vs gold_z
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df.to_csv(run_dir / "segments.tsv", sep="\t", index=False)

    result = _per_lang_correlations(
        run_dir,
        scorer_names=["raw_scorer", "z_scorer"],
        scorer_spaces={"raw_scorer": "raw_da", "z_scorer": "z_da"},
    )
    assert result[("raw_scorer", "en-gu")]["pearson"] == 1.0
    assert result[("raw_scorer", "en-gu")]["gold_column"] == "gold_raw"
    assert result[("z_scorer", "en-gu")]["pearson"] == 1.0
    assert result[("z_scorer", "en-gu")]["gold_column"] == "gold_z"


def test_per_lang_correlations_bold_warning_on_fallback(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "segment_id": ["s1", "s2", "s3"],
            "lang_pair": ["en-gu"] * 3,
            "gold_raw": [10.0, 20.0, 30.0],
            # no gold_z column
            "some_scorer": [10.0, 20.0, 30.0],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df.to_csv(run_dir / "segments.tsv", sep="\t", index=False)

    result = _per_lang_correlations(
        run_dir,
        scorer_names=["some_scorer"],
        scorer_spaces={"some_scorer": "z_da"},
    )
    captured = capsys.readouterr()
    assert "\033[1;33m" in captured.err
    assert "PREDICTION-SPACE FALLBACK" in captured.err
    assert "some_scorer" in captured.err
    assert result[("some_scorer", "en-gu")]["gold_column"] == "gold_raw"
    assert result[("some_scorer", "en-gu")]["fallback"] is True


def test_per_lang_correlations_legacy_gold_column(tmp_path):
    """Old segments.tsv outputs (pre-migration) have a single `gold` column.

    Tabulate must still work on those runs without crashing — it treats
    `gold` as `gold_raw` (the conservative default), matching the dataset-
    loader's migration rule."""
    df = pd.DataFrame(
        {
            "segment_id": ["s1", "s2", "s3"],
            "lang_pair": ["en-gu"] * 3,
            "gold": [10.0, 20.0, 30.0],  # legacy singular column
            "some_scorer": [10.0, 20.0, 30.0],
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df.to_csv(run_dir / "segments.tsv", sep="\t", index=False)

    result = _per_lang_correlations(
        run_dir,
        scorer_names=["some_scorer"],
        scorer_spaces={"some_scorer": "raw_da"},
    )
    assert result[("some_scorer", "en-gu")]["pearson"] == 1.0
    assert result[("some_scorer", "en-gu")]["gold_column"] == "gold"  # legacy name preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tabulate.py -v -k "per_lang_correlations"`
Expected: FAIL — current `_per_lang_correlations` takes no `scorer_spaces` kwarg and always reads a single `gold` column.

- [ ] **Step 3: Implement tabulate routing**

In `src/mt_metrix/reports/tabulate.py`, replace lines 164-203 with:

```python
def _per_lang_correlations(
    run_dir: Path,
    scorer_names: list[str],
    scorer_spaces: dict[str, str] | None = None,
) -> dict[tuple[str, str], dict[str, float | None]]:
    """Return {(scorer_name, lang_pair): {pearson, ..., gold_column, fallback}}.

    Pivots ``segments.tsv`` on ``lang_pair`` and invokes
    :func:`mt_metrix.io.writers._correlations` per group. Each scorer's
    gold column is chosen by its entry in ``scorer_spaces`` (``"raw_da"`` →
    ``gold_raw``, ``"z_da"`` → ``gold_z``). If the preferred column is
    absent, falls back to the other column (or to legacy singular
    ``gold``) and emits a bold stderr warning via
    :func:`mt_metrix.io.writers._emit_fallback_warning`.

    Backward-compat: runs produced before prediction-space routing have a
    single ``gold`` column; we accept it as a raw_da-equivalent so old
    outputs keep tabulating.
    """
    import pandas as pd

    from mt_metrix.io.writers import _correlations, _emit_fallback_warning, _pick_gold_column

    scorer_spaces = scorer_spaces or {}

    tsv_path = run_dir / "segments.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    out: dict[tuple[str, str], dict[str, float | None]] = {}

    has_gold_raw = "gold_raw" in df.columns
    has_gold_z = "gold_z" in df.columns
    has_legacy_gold = "gold" in df.columns and not (has_gold_raw or has_gold_z)

    if not (has_gold_raw or has_gold_z or has_legacy_gold) or "lang_pair" not in df.columns:
        return out

    for lp, sub_df in df.groupby("lang_pair", sort=False):
        lp_canon = canonicalise_lang_pair(str(lp))

        for col in scorer_names:
            if col not in sub_df.columns:
                continue

            declared_space = scorer_spaces.get(col, "raw_da")

            if has_legacy_gold:
                # Old-format run: single `gold` column serves as raw.
                chosen_col = "gold"
                is_fallback = False
            else:
                chosen_col, is_fallback = _pick_gold_column(
                    col, declared_space, has_gold_raw, has_gold_z
                )

            if chosen_col is None:
                out[(col, lp_canon)] = {
                    "pearson": None, "spearman": None, "kendall": None,
                    "spa": None, "n": 0, "gold_column": None, "fallback": False,
                }
                continue

            sub_gold = pd.to_numeric(sub_df[chosen_col], errors="coerce")
            pred = pd.to_numeric(sub_df[col], errors="coerce")
            mask = pred.notna() & sub_gold.notna()

            if is_fallback:
                preferred = {"raw_da": "gold_raw", "z_da": "gold_z"}[declared_space]
                _emit_fallback_warning(col, declared_space, preferred, chosen_col)

            if mask.sum() < 2:
                out[(col, lp_canon)] = {
                    "pearson": None, "spearman": None, "kendall": None,
                    "spa": None, "n": int(mask.sum()),
                    "gold_column": chosen_col, "fallback": is_fallback,
                }
                continue

            corr = _correlations(pred[mask].to_numpy(), sub_gold[mask].to_numpy())
            corr["gold_column"] = chosen_col
            corr["fallback"] = is_fallback
            out[(col, lp_canon)] = corr
    return out
```

- [ ] **Step 4: Thread scorer_spaces into collect_records**

In `collect_records` (around line 243 of `src/mt_metrix/reports/tabulate.py`), replace:

```python
        per_lang = _per_lang_correlations(run_dir, scored_names) if scored_names else {}
```

with:

```python
        # Prediction-space per scorer, as recorded in run_metadata.scorers.
        scorer_spaces = {
            s["name"]: (s.get("params") or {}).get("prediction_space", "raw_da")
            for s in scorer_md
        }
        per_lang = (
            _per_lang_correlations(run_dir, scored_names, scorer_spaces)
            if scored_names
            else {}
        )
```

- [ ] **Step 5: Run tabulate tests**

Run: `pytest tests/test_tabulate.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mt_metrix/reports/tabulate.py tests/test_tabulate.py
git commit -m "tabulate: route correlations by scorer prediction_space + bold fallback warning"
```

---

### Task 7: Migrate Surrey dataset configs to gold_raw + gold_z

**Files:**
- Modify: `configs/datasets/surrey_lowres_qe.yaml`
- Modify: `configs/datasets/surrey_legal.yaml`
- Modify: `configs/datasets/surrey_general.yaml`
- Modify: `configs/datasets/surrey_health.yaml`
- Modify: `configs/datasets/surrey_tourism.yaml`

No new tests — the existing dataset loader tests from Task 2 cover the parsing.

- [ ] **Step 1: Update surrey_lowres_qe.yaml**

Replace the `columns:` block (lines 40-45 in the current file) with:

```yaml
columns:
  source: original
  target: translation
  gold_raw: mean
  gold_z: z_mean
  lang_pair: "@from:lang_pair"
  domain: "@constant:lowres_qe"
```

Update the surrounding comment block: the "Target: z_mean" line now becomes two lines explaining both columns route to their respective gold fields. Example replacement for lines 19-22:

```yaml
# Targets:
#   gold_raw ← mean    (raw DA 0-100; matches Tower GEMBA-DA / GEMBA-MQM
#                      prediction space)
#   gold_z   ← z_mean  (per-rater z-normalised DA; matches COMET-QE and
#                      CometKiwi training signal — the ALOPE paper headline)
# Tabulate will pick per scorer based on each scorer's prediction_space
# (see configs/models/comet.yaml and tower.yaml).
```

- [ ] **Step 2: Update the four domain configs the same way**

For each of `surrey_legal.yaml`, `surrey_general.yaml`, `surrey_health.yaml`, `surrey_tourism.yaml`:

Find the `gold: z_mean` line and replace with:

```yaml
  gold_raw: mean
  gold_z: z_mean
```

Note: `surrey_general.yaml` documents that one of its subsets (en-Marathi) lacks the `mean` column — that's fine, those rows get `gold_raw: None` and z_da scorers route cleanly to `gold_z`. Add a comment near the new `gold_raw` line reminding readers:

```yaml
  # Note: en-Marathi subset lacks `mean`; those rows get gold_raw=None
  # and raw_da scorers will show NaN for that pair. z_da scorers are
  # unaffected (use gold_z).
  gold_raw: mean
  gold_z: z_mean
```

- [ ] **Step 3: Smoke-test via existing dataset loader tests**

Run: `pytest tests/test_datasets.py -v`
Expected: PASS (no regressions).

If the repo has fixture datasets that use `gold: z_mean`, update those too — search first:

```bash
grep -r "^\s*gold:\s*z_mean" configs/ tests/fixtures/ || echo "none found"
```

Fix any hits by migrating them the same way.

- [ ] **Step 4: Commit**

```bash
git add configs/datasets/surrey_legal.yaml configs/datasets/surrey_general.yaml \
        configs/datasets/surrey_health.yaml configs/datasets/surrey_tourism.yaml \
        configs/datasets/surrey_lowres_qe.yaml
git commit -m "datasets: migrate Surrey configs to gold_raw + gold_z"
```

---

### Task 8: Annotate model catalogues with prediction_space

**Files:**
- Modify: `configs/models/comet.yaml` — add `prediction_space: z_da` under each model entry
- Modify: `configs/models/tower.yaml` — add `prediction_space: raw_da` under each model entry
- Modify: `configs/models/sacrebleu.yaml` — add `prediction_space: raw_da` under each model entry
- Test: `tests/test_registry.py` (append a smoke test)

- [ ] **Step 1: Write failing smoke test**

Append to `tests/test_registry.py`:

```python
from pathlib import Path

from mt_metrix.config import _find_catalogues


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_every_catalogue_entry_declares_prediction_space():
    """Every scorer in every shipped catalogue must carry prediction_space.

    A missing value would silently default to raw_da at resolution time —
    fine as a loader default, but we want the catalogues themselves to be
    explicit so reviewers see the research decision for each model.
    """
    catalogues = _find_catalogues([REPO_ROOT])
    missing = [
        key for key, entry in catalogues.items()
        if "prediction_space" not in entry
    ]
    assert not missing, (
        "catalogue entries without prediction_space: " + ", ".join(missing)
    )


def test_expected_space_assignments():
    """Spot-check: the headline QE scorers route to the expected gold column.

    COMET family → z_da (trained on z-normalised DA).
    Tower family → raw_da (GEMBA-DA prompts ask for 0-100 raw; GEMBA-MQM
                   derived scores live in the same 0-100 range).
    sacrebleu   → raw_da (no training signal; use raw as the closer-scale
                  fallback).
    """
    catalogues = _find_catalogues([REPO_ROOT])
    assert catalogues["comet/wmt22-cometkiwi-da"]["prediction_space"] == "z_da"
    assert catalogues["comet/xcomet-xl-qe"]["prediction_space"] == "z_da"
    assert catalogues["tower/tower-plus-72b"]["prediction_space"] == "raw_da"
    assert catalogues["tower/tower-plus-72b-mqm"]["prediction_space"] == "raw_da"
    assert catalogues["sacrebleu/chrf"]["prediction_space"] == "raw_da"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v -k "prediction_space or space_assignments"`
Expected: FAIL — catalogue entries don't carry prediction_space yet.

- [ ] **Step 3: Annotate configs/models/comet.yaml**

For every entry under `models:` in `configs/models/comet.yaml`, add `prediction_space: z_da` as a top-level key on the entry (alongside `model:` and `needs_reference:`). Example — for `wmt22-cometkiwi-da` (currently at line 65):

```yaml
  wmt22-cometkiwi-da:
    model: Unbabel/wmt22-cometkiwi-da
    needs_reference: false
    prediction_space: z_da
    notes: "CometKiwi (WMT22) — reference-free, InfoXLM-large. Gated."
    params:
      batch_size: 64
      gpus: 1
      num_workers: 2
      progress_bar: true
```

Do this for ALL 13 entries in the file, including the BROKEN Marian entries (assign `z_da` to each — broken entries don't run but the catalogue should be complete).

- [ ] **Step 4: Annotate configs/models/tower.yaml**

For every entry under `models:` in `configs/models/tower.yaml`, add `prediction_space: raw_da`. This includes all DA variants AND all MQM variants — per the project discussion, GEMBA-MQM's 100 − error-penalty output occupies the same 0-100 raw space as GEMBA-DA, and no MQM gold exists in our datasets.

- [ ] **Step 5: Annotate configs/models/sacrebleu.yaml**

For every entry under `models:` in `configs/models/sacrebleu.yaml`, add `prediction_space: raw_da`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/models/comet.yaml configs/models/tower.yaml configs/models/sacrebleu.yaml tests/test_registry.py
git commit -m "models: annotate every catalogue entry with prediction_space"
```

---

### Task 9: End-to-end integration test

**Files:**
- Create: `tests/fixtures/tiny_both_gold.tsv`
- Create: `tests/fixtures/tiny_run_both_gold.yaml`
- Test: `tests/test_runner_e2e.py` (append)

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/tiny_both_gold.tsv` with this exact content:

```
source	target	gold_raw	gold_z	lang_pair
Hello world	Hola mundo	80.0	1.0	en-es
The book is on the table	El libro está sobre la mesa	85.0	1.5	en-es
I like coffee	Me gusta el café	75.0	0.5	en-es
She is reading	Ella está leyendo	70.0	0.0	en-es
They went to the park	Ellos fueron al parque	65.0	-0.5	en-es
```

- [ ] **Step 2: Create the run config**

Create `tests/fixtures/tiny_run_both_gold.yaml`:

```yaml
run:
  id: test_both_gold

dataset:
  kind: local
  path: tests/fixtures/tiny_both_gold.tsv
  lang_pair: en-es
  domain: test
  columns:
    source: source
    target: target
    gold_raw: gold_raw
    gold_z: gold_z
    lang_pair: lang_pair

scorers:
  - ref: sacrebleu/chrf

output:
  root: outputs
  formats: [tsv, jsonl, summary]
```

- [ ] **Step 3: Write failing end-to-end test**

Append to `tests/test_runner_e2e.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_e2e_both_gold_columns_round_trip(tmp_path, monkeypatch):
    """Run a tiny local-TSV scoring job end-to-end and verify:

    1. segments.tsv carries gold_raw AND gold_z columns
    2. segments.jsonl has both fields
    3. summary.json correlation_vs_gold has a gold_column field
    4. Tabulate picks the right gold column per scorer's prediction_space

    Uses sacrebleu/chrf (prediction_space=raw_da) so correlation routes to
    gold_raw. The fixture is constructed so gold_raw and gold_z are not
    linearly related, making the routing check meaningful.
    """
    import pandas as pd

    from mt_metrix.config import load_run_config
    from mt_metrix.runner import run

    monkeypatch.chdir(REPO_ROOT)
    out_root = tmp_path / "outputs"
    cfg = load_run_config("tests/fixtures/tiny_run_both_gold.yaml")
    cfg.output.root = str(out_root)
    run_dir = run(cfg)  # runs chrf, writes all formats; returns the output dir
    tsv = pd.read_csv(run_dir / "segments.tsv", sep="\t")
    assert "gold_raw" in tsv.columns
    assert "gold_z" in tsv.columns
    assert list(tsv["gold_raw"]) == [80.0, 85.0, 75.0, 70.0, 65.0]
    assert list(tsv["gold_z"]) == [1.0, 1.5, 0.5, 0.0, -0.5]

    jsonl_line = (run_dir / "segments.jsonl").read_text().splitlines()[0]
    obj = json.loads(jsonl_line)
    assert obj["gold_raw"] == 80.0
    assert obj["gold_z"] == 1.0

    summary = json.loads((run_dir / "summary.json").read_text())
    corr = summary["metrics"]["chrf"]["correlation_vs_gold"]
    assert corr is not None
    # chrf is raw_da → should route to gold_raw, no fallback
    assert corr["gold_column"] == "gold_raw"
    assert corr["fallback"] is False
```

The entry function is `run(config: RunConfig) -> Path` at `src/mt_metrix/runner.py:42` and returns the output directory, which is what the test binds to `run_dir`.

- [ ] **Step 4: Run the E2E test**

Run: `pytest tests/test_runner_e2e.py::test_e2e_both_gold_columns_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/tiny_both_gold.tsv tests/fixtures/tiny_run_both_gold.yaml tests/test_runner_e2e.py
git commit -m "tests: e2e test for gold_raw/gold_z routing"
```

---

### Task 10: Documentation

**Files:**
- Create: `docs/PREDICTION_SPACES.md`
- Modify: `docs/DESIGN.md` (append a short section with a link to the new doc)

- [ ] **Step 1: Create docs/PREDICTION_SPACES.md**

```markdown
# Prediction-Space Routing

Every scorer predicts in a named *prediction space*. Every dataset may carry
gold columns for one or more spaces. At tabulate time mt-metrix picks, per
scorer, the gold column whose space matches — so COMET-QE correlates against
z-normalised gold and Tower-GEMBA-DA correlates against raw 0-100 gold.

## Supported spaces

| Space      | Meaning                                                | Gold column |
|------------|--------------------------------------------------------|-------------|
| `raw_da`   | Raw Direct Assessment, 0-100                           | `gold_raw`  |
| `z_da`     | Per-rater z-normalised DA                              | `gold_z`    |

Add a new space by:
1. Extending `PredictionSpace` in `src/mt_metrix/io/schema.py`.
2. Extending the gold-column map in `_pick_gold_column`
   (`src/mt_metrix/io/writers.py`).
3. Declaring the new gold column in the relevant dataset configs.

## Declaring per scorer

Catalogue entry (e.g. `configs/models/comet.yaml`):

```yaml
wmt22-cometkiwi-da:
  model: Unbabel/wmt22-cometkiwi-da
  needs_reference: false
  prediction_space: z_da
  params:
    batch_size: 64
```

Missing `prediction_space:` defaults to `raw_da` (conservative).

## Declaring per dataset

Dataset config (e.g. `configs/datasets/surrey_lowres_qe.yaml`):

```yaml
columns:
  source: original
  target: translation
  gold_raw: mean      # raw DA column in the HF dataset
  gold_z: z_mean      # z-normalised column in the HF dataset
  lang_pair: "@from:lang_pair"
```

A dataset may declare either column, both, or neither. Tabulate picks the
matching one per scorer; when the matching column is absent it falls back
to the other, emitting a bold stderr warning.

## Why correlations are scale-invariant anyway

Pearson/Spearman/Kendall/SPA are all scale-invariant to linear transforms of
the predictor — so correlating raw predictions against `gold_z` still gives
a meaningful rank-agreement number. The point of explicit routing is:

1. **Training-signal alignment**: a COMET-QE model trained on z-normalised
   DA is evaluated against z-normalised gold, not a per-rater-averaged raw
   number that's algebraically different.
2. **Explicit research decision**: each scorer's target space is visible in
   the catalogue, not inferred from the model name.
3. **Downstream safety**: if we ever report MAE or absolute-score agreement
   (not today, but future), routing already lives in the code — no silent
   breakage when we add it.

## Legacy support

Dataset configs using the old singular `gold:` key still load — it routes
into `gold_raw` (with a one-shot deprecation log per load). Output runs
produced before the migration have a single `gold` column in
`segments.tsv`; tabulate treats that column as raw_da-equivalent so old
outputs keep working.
```

- [ ] **Step 2: Link from docs/DESIGN.md**

At the bottom of `docs/DESIGN.md`, append:

```markdown
## Prediction-space routing

Per-scorer routing of correlations to the matching gold column is documented
in [PREDICTION_SPACES.md](PREDICTION_SPACES.md). TL;DR: COMET-QE correlates
against `gold_z` (z-normalised DA); Tower GEMBA-DA/MQM correlates against
`gold_raw` (raw 0-100 DA). Library warns on fallback.
```

- [ ] **Step 3: Commit**

```bash
git add docs/PREDICTION_SPACES.md docs/DESIGN.md
git commit -m "docs: prediction-space routing reference"
```

---

## Final verification

After all tasks are complete:

- [ ] **Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. If `tests/test_datasets.py::test_load_hf_*` still shows pre-existing `ModuleNotFoundError` failures unrelated to this work, note them in the merge commit body and leave them for a follow-up task.

- [ ] **Push**

```bash
git push origin main
```

- [ ] **Re-tabulate any pre-existing runs to pick up the new routing**

```bash
# e.g. for a previously-landed Legal run
mt-metrix tabulate 'outputs/surrey_legal_*' --out docs/reports/legal.md
```

The per-scorer routing applies retroactively to old outputs because the
tabulate logic reads segments.tsv columns, not the scorer's behaviour at
scoring time. Pre-migration runs (single `gold` column) still work — see
the legacy-compat test in Task 6.
