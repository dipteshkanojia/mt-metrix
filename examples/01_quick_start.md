# 01 — Quick start

The shortest path from "nothing installed" to "scores on screen".

## 0. Install

```bash
git clone https://github.com/dipteshkanojia/mt-metrix.git
cd mt-metrix

python -m venv .venv && source .venv/bin/activate
pip install -e ".[comet]"          # core + unbabel-comet
pip install sacrebleu>=2.4         # already pulled in by core
```

This is enough for COMET + sacrebleu. Tower (vLLM) brings in heavier deps —
install `.[tower]` when you need it.

## 1. Set your HF token

Required for CometKiwi, XCOMET-XL/XXL, and all Tower models (all gated).

```bash
echo "hf_xxx..." > ~/.hf_token
chmod 600 ~/.hf_token
export HF_TOKEN=$(cat ~/.hf_token)
```

Visit each model's HF page (`https://huggingface.co/Unbabel/wmt22-cometkiwi-da`
etc.) once and click "Agree and access".

## 2. Run the smoke test

```bash
mt-metrix score --config configs/runs/example_quick.yaml
```

This loads 20 rows of Surrey Legal-QE, runs `eamt22-cometinho-da`,
`wmt22-cometkiwi-da`, and `sacrebleu/chrf`. Expected wall time:
1–2 minutes on a single A100, longer on first run (downloads).

## 3. Read the outputs

```
outputs/example_quick/
├── config.yaml        # snapshot of what ran
├── run.log            # what happened
├── segments.tsv       # flat, human-readable
├── segments.jsonl     # rich, with per-scorer extras
└── summary.json       # aggregates + correlations
```

Have a look at `summary.json`:

```json
{
  "n_segments": 20,
  "metrics": {
    "cometkiwi-da": {
      "n": 20,
      "mean": 0.614,
      "correlation_vs_gold": {
        "pearson": 0.43,
        "spearman": 0.41,
        "kendall": 0.29,
        "spa": 0.64,
        "n": 20
      }
    }
  },
  "skipped_metrics": [
    {"name": "cometinho-da", "reason": "dataset-has-no-references"},
    {"name": "chrf", "reason": "dataset-has-no-references"}
  ],
  "corpus_scores": {}
}
```

The ref-based metrics (cometinho, chrf) appear in `skipped_metrics` because
Surrey Legal-QE has no reference column. That's the auto-detect behaviour at
work.

## 4. Re-compute correlations without re-running inference

If you want Pearson on a different subset, or SPA with different tie
handling, you can regenerate correlations from the TSV:

```bash
mt-metrix correlate --run outputs/example_quick
```

…prints a JSON with per-metric correlations to stdout.

## Next

- `02_add_more_models.md` — customise the run to add more COMET / Tower variants.
- `03_reference_based_with_wmt.md` — use a WMT dataset that DOES have references, so BLEU/chrF++ actually run.
- `04_aisurrey_submission.md` — run the full-matrix recipe on AISURREY.
- `05_adding_metrics.md` — plug in a new metric family.
