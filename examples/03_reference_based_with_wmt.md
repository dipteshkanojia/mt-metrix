# 03 — Using a dataset with references (so BLEU / chrF++ actually run)

Surrey domain QE datasets have no reference column, so reference-based
metrics (BLEU, chrF++, TER, `wmt22-comet-da`, XCOMET-XL) are auto-skipped.
For studies that DO need reference-based scores, point at a dataset that
carries references — e.g. WMT general-translation or WMT QE DA sets.

## Example: WMT 2024 general-translation

```yaml
# configs/runs/wmt24_full.yaml
run:
  id: wmt24_full

dataset:
  kind: huggingface
  repo: google/wmt24pp            # any WMT-style HF repo with ref column
  split: train
  columns:
    source: source
    target: target
    reference: reference
    gold: score                   # or whatever the dataset calls DA/MQM
    lang_pair: "@from:lp"
    domain: "@constant:wmt24"

scorers:
  - ref: comet/wmt22-comet-da
  - ref: comet/wmt22-cometkiwi-da
  - ref: comet/xcomet-xl
  - ref: sacrebleu/bleu
  - ref: sacrebleu/chrf++
  - ref: sacrebleu/ter

output:
  root: outputs
  formats: [tsv, jsonl, summary]
```

Run:

```bash
mt-metrix score --config configs/runs/wmt24_full.yaml
```

Because every row has `reference`, nothing ends up in `skipped_metrics`. You
get per-segment BLEU/chrF++/TER alongside per-segment COMET, plus corpus-
level BLEU/chrF++/TER under `summary.json::corpus_scores`.

## Column mapping, in one place

The `columns:` mapping in a dataset YAML is the one piece of glue you have to
maintain when adapting to a new dataset shape. Three directive styles:

| Value                   | Meaning                                    |
|-------------------------|--------------------------------------------|
| `some_col`              | copy from that column in the raw data       |
| `@constant:health`      | literal value `"health"` for every row      |
| `@from:language_pair`   | copy from `language_pair` (synonym of bare) |

The `@constant` directive is useful when a single dataset is one language
pair or one domain — you don't need a whole column.

## Using gyroQE's processed TSVs

If you already have gyroQE's unified TSVs, point at them with `kind: gyroqe`:

```yaml
dataset:
  kind: gyroqe
  path: /path/to/gyroQE
  year: wmt23
  lang_pair: en-de
  split: test
```

This resolves to
`{path}/data/processed/{year}/{lang_pair}/{split}.tsv` and applies
gyroQE's standard column mapping (`source`, `target`, `reference`, `z_mean`,
`lang_pair`, `domain`).

## Partial references

If only some rows have a reference, the runner:

1. Runs reference-based scorers on the subset that has one.
2. Fills the non-reference rows with `NaN` (serialised as empty string in TSV,
   `null` in JSONL), with `extra: {"skipped": "no-reference"}`.
3. Logs a warning like `partial references detected (147/300 segments have refs)`.
4. Reports correlations only over the rows where both the score and gold are
   present.
