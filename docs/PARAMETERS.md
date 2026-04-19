# Parameter provenance

Defaults in `configs/models/*.yaml` trace back to the published papers and
official reproducibility notes. This doc exists so you know why each value
is what it is before overriding it.

## COMET

### Reference-based (`comet-da`, COMET-22)

Source: **Rei et al. 2020**, "COMET: A Neural Framework for MT Evaluation"
(arXiv:2009.09025). **Rei et al. 2022**, "COMET-22: Unbabel-IST 2022
Submission for the Metrics Shared Task" (arXiv:2209.06243).

- Architecture: XLM-R-large encoder with a feed-forward regressor head.
- Output range: typically [−0.5, 1.5], loosely calibrated to DA z-scores.
- Recommended `batch_size`: 64 on A100; drop to 32 on smaller GPUs.
- `gpus: 1` is the default — the library parallelises within-batch; multi-GPU
  gives diminishing returns for most dataset sizes.

### CometKiwi (reference-free QE)

Source: **Rei et al. 2022b**, "CometKiwi: IST-Unbabel 2022 Submission for the
Quality Estimation Shared Task" (arXiv:2209.06243).

- Base (`wmt22-cometkiwi-da`): InfoXLM-large backbone, ~550M params.
- Batch size 64 is stable on one A100 80GB; halve on smaller cards.

### CometKiwi-XL / XXL

Source: **Rei et al. 2023**, "Scaling Up CometKiwi" (arXiv:2306.11925).

- XL: XLM-R-XL 3.5B params. `batch_size: 16`.
- XXL: XLM-R-XXL 10.7B params. `batch_size: 8` on A100 80GB; will OOM on 40GB.
- Both are **gated** on HF Hub.

### XCOMET

Source: **Guerreiro et al. 2023**, "xCOMET: Transparent MT Evaluation through
Fine-grained Error Detection" (arXiv:2310.10482).

- Produces error spans in addition to a score; we surface them in
  `segments.jsonl` via the `output_seg_err_spans: true` parameter.
- XCOMET-XL (~3.5B), XCOMET-XXL (~10.7B). Batch sizes: 16 / 8.
- We expose a QE mode (`xcomet-xl-qe`, `xcomet-xxl-qe`) that omits the
  reference — the same checkpoint, different call.

## Tower

Source: **Alves et al. 2024**, "Tower: An Open Multilingual LLM for Translation-
Related Tasks" (arXiv:2402.17733). **Pombal et al. 2025**, "Tower-Plus" (Unbabel
blog/report).

- Generation config for evaluation: `temperature=0.0`, `top_p=1.0`,
  `max_tokens=50` for GEMBA-DA (single score),
  `max_tokens=384` for GEMBA-MQM (multi-line errors).
- **Tensor parallel size:** auto-picked by scorer based on param count —
  1 for ≤9B, 2 for 13B, 4 for 72B. Override via `tensor_parallel_size`.
- Tower-Plus tokenisers differ between 2B/9B (Gemma2) and 72B (Qwen2) —
  chat-template handling is done by `vllm`/`transformers`; we don't hand-roll.

## GEMBA prompts

Source: **Kocmi & Federmann 2023**, "Large Language Models Are State-of-the-Art
Evaluators of Translation Quality" (EAMT 2023).

### GEMBA-DA (`prompts/gemba_da.py`)

- Asks the LLM for a continuous score 0–100.
- Parser: regex over the response, first integer/decimal in [0, 100]
  wins.
- `parse_ok=False` is surfaced in `segments.jsonl` extras when no valid number
  is found — use this to filter noisy Tower outputs.

### GEMBA-MQM (`prompts/gemba_mqm.py`)

- Asks the LLM to list errors with `<severity> - <category> - "<span>"`.
- Severity weights follow Freitag et al. MQM practice:
  - critical = −25
  - major = −5
  - minor = −1
  - no-error = 0
- Final score = `max(0, min(100, 100 + sum(weights)))`.
- Parser tolerates formatting drift (dashes vs. colons, quote styles, trailing
  punctuation). Fallback: recognise severity-only lines.

## Reference-based lexical metrics (sacrebleu)

Source: **Post 2018**, "A Call for Clarity in Reporting BLEU Scores"
(arXiv:1804.08771). Signatures in the `corpus_score` payload document the
exact configuration so results are reproducible.

### BLEU — Papineni et al. 2002

- `tokenize: "13a"` (WMT default; not appropriate for CJK — use `"ja-mecab"`,
  `"zh"`, or `"char"` where relevant).
- `smooth_method: "exp"` (exponential smoothing; handles 0-count n-grams).
- `lowercase: false` (always — case is a feature of the target language).

### chrF / chrF++ — Popović 2017

- `char_order: 6` — 1..6 character n-grams.
- `word_order: 0` for plain chrF; `word_order: 2` for chrF++ (WMT default).
- `beta: 2` — recall weighted more than precision (WMT practice).
- `use_effective_order: true` — avoid 0 scores when short references have
  fewer n-grams than `char_order`.

### TER — Snover et al. 2006

- `normalized: false` — raw sacrebleu TER.
- `no_punct: false` — keep punctuation as its own edits.
- `asian_support: true` — enables sentence splitting for CJK.
- `case_sensitive: true`.

## Correlation metrics (`io/writers.py`)

Reported in `summary.json::metrics::<name>::correlation_vs_gold`:

- **Pearson** — linear correlation.
- **Spearman** — rank-based, robust to non-linear monotonic relationships.
- **Kendall** — pairwise rank, more conservative.
- **SPA (Soft Pairwise Accuracy)** — fraction of ordered pairs where the
  sign of the metric difference matches the sign of the gold difference,
  with ties contributing 0.5. Follows Thompson et al.'s WMT-style reporting.

## When to override

- **OOM on a target GPU** → lower `batch_size`. Start by halving.
- **Non-WMT tokeniser needed** (e.g. ja/zh target) → set
  `sacrebleu/bleu` overrides: `tokenize: "ja-mecab"` etc.
- **Low parse_ok on Tower** → try `prompt_mode: gemba-mqm` (the longer output
  tends to parse more reliably) or bump `max_tokens`.
- **Different temperature** — don't. Evaluation is deterministic scoring, not
  generation. Keep `temperature: 0.0`.
