# Surrey IndicQE Dataset Family

Reference document for the forthcoming **IndicQE** paper (Kanojia et al.).
Consolidates every Surrey-published sentence-level MT quality-estimation
dataset that touches an Indic language pair plus the related low-resource
(Estonian/Nepali/Sinhala ↔ English) set, organised **by language pair**
rather than by domain.

The goal: when the IndicQE paper is drafted, every number, every pair, and
every reproducibility command lives here, so the Methods and Data sections
write themselves from this doc + the `mt-metrix tabulate` outputs.

Scope for the paper: ALOPE follow-up, Surrey QE stack, 2026.

---

## 1. At a glance

Five HuggingFace datasets, all QE-only (no references), all carrying
per-rater Direct-Assessment scores plus a z-normalised gold (`z_mean`).

| # | Dataset | Repo | Domain | Pairs | Gold | Licence |
|---|---------|------|--------|-------|------|---------|
| 1 | Legal-QE         | [surrey-nlp/Legal-QE](https://hf.co/datasets/surrey-nlp/Legal-QE)                 | legal      | en→{gu,ta,te}                 | `z_mean` | AFL-3.0 |
| 2 | General-QE       | [surrey-nlp/General-QE](https://hf.co/datasets/surrey-nlp/General-QE)             | general    | en→{gu,hi,mr,ta,te}           | `z_mean` | AFL-3.0 |
| 3 | Health-QE        | [surrey-nlp/health-QE](https://hf.co/datasets/surrey-nlp/health-QE)               | health     | en→{gu,hi,mr,ta}              | `z_mean` | AFL-3.0 |
| 4 | Tourism-QE       | [surrey-nlp/Tourism-QE](https://hf.co/datasets/surrey-nlp/Tourism-QE)             | tourism    | en→{hi,mr,te}                 | `z_mean` | AFL-3.0 |
| 5 | Low-resource-QE-DA | [surrey-nlp/Low-resource-QE-DA-dataset](https://hf.co/datasets/surrey-nlp/Low-resource-QE-DA-dataset) | low-resource | en→{gu,hi,mr,ta,te}, {et,ne,si}→en | `z_mean` | CC |

Dataset 5 is the ALOPE-paper release: **Sindhujan, A., Qian, S., Matthew,
C.C.C., Orasan, C., and Kanojia, D.** _ALOPE: Adaptive Layer Optimization
for Translation Quality Estimation using Large Language Models._ COLM 2024.
[arXiv:2508.07484](https://arxiv.org/abs/2508.07484).

Datasets 1–4 are the domain-specific Surrey QE releases; they share a
common schema (`source_text` / `target_text` / `language_pair` / `z_mean`)
and per-row `language_pair` tagging, which makes pair-wise pivots free.

---

## 2. Master language-pair × domain matrix

Test-split row counts (approximate; `mt-metrix tabulate` emits the exact
numbers). `—` means the pair is absent from that dataset.

| Language pair | Family     | Direction  | Legal | General | Health | Tourism | Low-res | Total stacked (approx) |
|---------------|------------|------------|-------|---------|--------|---------|---------|------------------------|
| en → gu       | Indo-Aryan | en→Indic   | ~900  | ~1,000  | ~1,000 | —       | ~1,000  | ~3,900                 |
| en → hi       | Indo-Aryan | en→Indic   | —     | ~1,000  | ~1,000 | ~1,000  | ~1,000  | ~4,000                 |
| en → mr       | Indo-Aryan | en→Indic   | —     | ~1,000  | ~1,000 | ~1,000  | ~700    | ~3,700                 |
| en → ta       | Dravidian  | en→Indic   | ~900  | ~1,000  | ~1,000 | —       | ~1,000  | ~3,900                 |
| en → te       | Dravidian  | en→Indic   | ~900  | ~1,000  | —      | ~1,000  | ~1,000  | ~3,900                 |
| et → en       | Uralic¹    | X→en       | —     | —       | —      | —       | ~1,000  | ~1,000                 |
| ne → en       | Indo-Aryan | X→en       | —     | —       | —      | —       | ~1,000  | ~1,000                 |
| si → en       | Indo-Aryan | X→en       | —     | —       | —      | —       | ~1,000  | ~1,000                 |
| **Total**     |            |            | ~2,700 | ~5,000  | ~4,000 | ~3,000  | ~7,700  | **~22.4k**             |

¹ Estonian is Finno-Ugric, not Indic. It is retained in the dataset family
because ALOPE's low-resource scope included it; for the IndicQE paper
proper it can be reported as an out-of-family control or dropped.

**Indic pairs proper (paper scope):** en↔gu, en↔hi, en↔mr, en↔ta, en↔te,
ne↔en, si↔en — 7 pairs, 5 Indo-Aryan + 2 Dravidian, 21.4k test rows across
all domains.

---

## 3. Per-pair fact-sheets

Each pair shows which datasets cover it, the HF subset names (which differ
across datasets — see §6), and the per-domain test size.

### en → Gujarati (gu)
Indic family: Indo-Aryan. Script: Gujarati. Direction: en→gu.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Legal-QE        | `en-gujarati`   | ~900      | No        | `z_mean` |
| General-QE      | `en-gujarati`   | ~1,000    | No        | `z_mean` |
| Health-QE       | `en-gujarati`   | ~1,000    | No        | `z_mean` |
| Low-resource-QE-DA (multilingual) | row where `lang_pair = engu` | ~1,000 | No | `z_mean` |

### en → Hindi (hi)
Indic family: Indo-Aryan. Script: Devanagari. Direction: en→hi.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| General-QE      | `en-hindi`      | ~1,000    | No        | `z_mean` |
| Health-QE       | `en-hindi`      | ~1,000    | No        | `z_mean` |
| Tourism-QE      | `en-hindi`      | ~1,000    | No        | `z_mean` |
| Low-resource-QE-DA (multilingual) | row where `lang_pair = enhi` | ~1,000 | No | `z_mean` |

### en → Marathi (mr)
Indic family: Indo-Aryan. Script: Devanagari. Direction: en→mr.
**Data quirk:** General-QE's `en-Marathi` subset is missing the `mean`
column and uses space-separated score lists; the dataset config already
defaults to `z_mean` (which is present) so this is handled.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| General-QE      | `en-marathi`    | ~1,000    | No        | `z_mean` |
| Health-QE       | `en-marathi`    | ~1,000    | No        | `z_mean` |
| Tourism-QE      | `en-marathi`    | ~1,000    | No        | `z_mean` |
| Low-resource-QE-DA (multilingual) | row where `lang_pair = enmr` | ~700   | No | `z_mean` |

### en → Tamil (ta)
Indic family: Dravidian. Script: Tamil. Direction: en→ta.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Legal-QE        | `en-tamil`      | ~900      | No        | `z_mean` |
| General-QE      | `en-tamil`      | ~1,000    | No        | `z_mean` |
| Health-QE       | `en-tamil`      | ~1,000    | No        | `z_mean` |
| Low-resource-QE-DA (multilingual) | row where `lang_pair = enta` | ~1,000 | No | `z_mean` |

### en → Telugu (te)
Indic family: Dravidian. Script: Telugu. Direction: en→te.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Legal-QE        | `en-telugu`     | ~900      | No        | `z_mean` |
| General-QE      | `en-telugu`     | ~1,000    | No        | `z_mean` |
| Tourism-QE      | `en-telugu`     | ~1,000    | No        | `z_mean` |
| Low-resource-QE-DA (multilingual) | row where `lang_pair = ente` | ~1,000 | No | `z_mean` |

### et → English (en)
**Not Indic** (Uralic/Finno-Ugric). Retained from ALOPE for low-resource
context. Consider out-of-family in the IndicQE paper.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Low-resource-QE-DA (multilingual) | row where `lang_pair = eten` | ~1,000 | No | `z_mean` |

### ne → English (en)
Indic family: Indo-Aryan. Script: Devanagari. Direction: ne→en.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Low-resource-QE-DA (multilingual) | row where `lang_pair = neen` | ~1,000 | No | `z_mean` |

### si → English (en)
Indic family: Indo-Aryan. Script: Sinhala. Direction: si→en.

| Dataset         | HF subset       | Test rows | Has refs? | Target   |
|-----------------|-----------------|-----------|-----------|----------|
| Low-resource-QE-DA (multilingual) | row where `lang_pair = sien` | ~1,000 | No | `z_mean` |

---

## 4. Target variable & correlation convention

All five datasets publish per-rater Direct Assessment scores with the
following derived columns:

- `mean` (scalar) — arithmetic mean of the three rater scores per segment.
- `z_mean` (scalar) — mean of per-rater z-normalised scores, i.e. each
  rater's scores are first standardised (subtract their mean, divide by
  their stddev) and the three z-scores are then averaged. This controls
  for per-rater scale drift.

**Target for every run: `z_mean`.** It is the ALOPE paper's headline
signal and the WMT QE shared-task convention. `mean` is retained in the
segment metadata for optional secondary analysis but not used for
correlation reporting.

`summary.json` records Pearson, Spearman, and Kendall-τ correlations per
scorer against `z_mean`. `mt-metrix tabulate` pivots those three
correlations × `lang_pair × domain`, producing the IndicQE paper's
headline table directly.

---

## 5. Available metrics landscape

All five datasets are **reference-free** (QE-only). The runner
auto-detects this and skips reference-based scorers with a visible warning
(recorded in `summary.json::skipped_metrics`).

Applicable scorer families (defined in `configs/runs/*_full_matrix.yaml`):

- **COMET QE (reference-free)** — CometKiwi-22, CometKiwi-23-XL,
  CometKiwi-23-XXL, XCOMET-XL-QE, XCOMET-XXL-QE, WMT20-COMET-QE-DA,
  WMT21-COMET-QE-DA, WMT21-COMET-QE-MQM. (8 scorers.)
- **Tower GEMBA-DA** — tower-plus-2b, towerinstruct-7b-v0.2,
  towerinstruct-mistral-7b-v0.2, towerinstruct-wmt24-chat-7b,
  towerinstruct-13b-v0.1, tower-plus-9b, tower-plus-72b. (7 scorers.)
- **Tower GEMBA-MQM** — towerinstruct-7b-v0.2-mqm, tower-plus-9b-mqm.
  (2 scorers.)

**Not applicable (will be listed in skipped_metrics):**

- COMET reference-based — wmt22-comet-da, eamt22-cometinho-da, xcomet-xl,
  xcomet-xxl.
- sacrebleu — BLEU, chrF, chrF++, TER.

If the IndicQE paper requires any reference-based comparison, a separate
corpus with target-side references is needed (neither Legal nor General
nor Health nor Tourism nor Low-resource-QE-DA ships references). Options:
WMT shared-task dev/test sets, FLORES-200 devtest (has refs), IndicCorp.

---

## 6. Cross-dataset `lang_pair` code conventions

**Heads up:** the HF datasets do not share a single `lang_pair` convention.

| Dataset            | `lang_pair` column values |
|--------------------|---------------------------|
| Legal-QE           | `en-gu`, `en-ta`, `en-te` |
| General-QE         | `en-gu`, `en-hi`, `en-mr`, `en-ta`, `en-te` |
| Health-QE          | `en-gu`, `en-hi`, `en-mr`, `en-ta` |
| Tourism-QE         | `en-hi`, `en-mr`, `en-te` |
| Low-resource-QE-DA (multilingual) | `engu`, `enhi`, `enmr`, `enta`, `ente`, `eten`, `neen`, `sien` |

The subset names are even less aligned: Legal/General/Health/Tourism use
full-language hyphenated names (`en-gujarati`), while Low-resource uses
compact codes (`engu`).

**Implication for aggregation:** `mt-metrix tabulate` pivots on whatever
`lang_pair` strings appear verbatim, so a naive aggregation across all
five datasets will produce two rows for every en→Indic pair (e.g. `en-gu`
and `engu` as separate rows). Before producing paper-ready tables, either
normalise with a post-hoc sed/awk over the per-dataset `segments.tsv`
outputs, or add a `lang_pair_map` option to the loader (planned
enhancement; tracked separately). Recommended canonical form for the
paper: **ISO 639-1 hyphenated, src–tgt direction preserved**
(`en-gu`, `et-en`, `ne-en`, `si-en`).

---

## 7. Running the full IndicQE matrix

Each dataset has a paired full-matrix config. On AISURREY, via the
pre-flight submit wrapper:

```bash
# Domain-specific (en→Indic)
scripts/submit.sh configs/runs/surrey_legal_full_matrix.yaml   --mem=256G --time=24:00:00
scripts/submit.sh configs/runs/surrey_general_full_matrix.yaml --mem=256G --time=24:00:00
scripts/submit.sh configs/runs/surrey_health_full_matrix.yaml  --mem=256G --time=24:00:00
scripts/submit.sh configs/runs/surrey_tourism_full_matrix.yaml --mem=256G --time=24:00:00

# Low-resource (mixed directions, includes Nepali/Sinhala/Estonian)
scripts/submit.sh configs/runs/surrey_lowres_qe_full_matrix.yaml --mem=256G --time=24:00:00
```

`--mem=256G` is mandatory for the full matrix — it chains CometKiwi-XXL
(~40 GB fp32 state dict, loaded into CPU RAM before GPU transfer) AND
Tower-13B / Tower-72B-awq (vLLM CPU copy before sharding). The slurm
template's default `--mem=128G` covers any _single_ XXL scorer, not the
full chain. See `reference_aisurrey_deploy_pattern` memory for the
per-family RAM-floor table and the OOM post-mortem that motivated the
256 GB figure.

Smoke before the full matrix (CometKiwi-DA only, ~15–30 min each):

```bash
scripts/submit.sh configs/runs/surrey_legal_cometkiwi.yaml      -p nice-project --mem=32G --time=01:00:00
scripts/submit.sh configs/runs/surrey_general_cometkiwi.yaml    -p nice-project --mem=32G --time=01:00:00
scripts/submit.sh configs/runs/surrey_health_cometkiwi.yaml     -p nice-project --mem=32G --time=01:00:00
scripts/submit.sh configs/runs/surrey_tourism_cometkiwi.yaml    -p nice-project --mem=32G --time=01:00:00
scripts/submit.sh configs/runs/surrey_lowres_qe_cometkiwi.yaml  -p nice-project --mem=32G --time=01:00:00
```

After all five runs land, the pair-wise / domain-wise matrix:

```bash
mt-metrix tabulate \
  --run outputs/surrey_legal_full_matrix_*/ \
  --run outputs/surrey_general_full_matrix_*/ \
  --run outputs/surrey_health_full_matrix_*/ \
  --run outputs/surrey_tourism_full_matrix_*/ \
  --run outputs/surrey_lowres_qe_full_matrix_*/ \
  --pivot lang_pair,domain \
  --metric spearman \
  --out reports/indic_qe_matrix
```

This emits `reports/indic_qe_matrix.{csv,md,tex}` — the third being
booktabs-ready for the paper.

---

## 8. Compute budget (rough)

For a clean run of every full matrix on AISURREY:

| Dataset         | Rows | --mem | Partition       | Expected wall-time | Comments                              |
|-----------------|------|-------|-----------------|--------------------|---------------------------------------|
| Legal-QE        | ~2,700 | 256G | a100           | 8–14 h             | 3 pairs; XXL + Tower-72B dominate     |
| General-QE      | ~5,000 | 256G | a100           | 12–18 h            | 5 pairs                               |
| Health-QE       | ~4,000 | 256G | a100           | 10–16 h            | 4 pairs                               |
| Tourism-QE      | ~3,000 | 256G | a100           | 8–14 h             | 3 pairs                               |
| Low-resource-QE-DA | ~7,700 | 256G | a100          | 16–24 h            | ~3× Legal; split if wall-time tight   |
| **Total**       | ~22,400 |     |                 | **~54–86 h**        | Queue sequentially, not in parallel   |

Tower-72B-awq is the single biggest cost per dataset; if wall-time is
tight, drop it and note in the paper that 72B was excluded for budget.
Alternatively split each full-matrix into `light` (everything up to XL)
and `heavy` (XXL + 13B+) tiers — the runner's per-scorer persistence
makes the split safe (either half can fail without wasting the other).

---

## 9. Reproducibility checklist

For the IndicQE paper's artefact submission:

- [ ] HF repos cited verbatim (§1 table).
- [ ] `z_mean` defined explicitly (§4).
- [ ] Run scripts: five `*_full_matrix.yaml` files live at
      `configs/runs/surrey_{legal,general,health,tourism,lowres_qe}_full_matrix.yaml`.
- [ ] Environment: `torch==2.4.0+cu121`, `vllm 0.6.3.post1`,
      `unbabel-comet` (version from `pip freeze`), AISURREY a100 nodes.
- [ ] Each `outputs/<run_id>/` is a reproducibility artefact: contains
      `config.yaml`, `segments.tsv`, `segments.jsonl`, `summary.json`,
      `run.log`. Bundle the five for the paper's supplementary material.
- [ ] `git rev-parse HEAD` at run time is in `summary.json::git_sha`.
- [ ] Gated-model licences (CometKiwi-22, XCOMET-*, all Tower models)
      accepted on the HF UI for the account whose token sits at
      `~/.hf_token` on the cluster.

---

## 10. Open questions / followups for the paper

These need answering before the paper's Methods section stabilises:

1. **IndicQE scope** — does Estonian→English stay in, as out-of-family
   control, or is it dropped? (Affects pair count: 7 vs 8.)
2. **Reference-based baselines** — do we claim "QE-only" or produce a
   reference-based comparison on a held-out set (e.g. FLORES-200)?
3. **`lang_pair` code canonicalisation** — run the tabulate step on
   un-normalised codes (ugly LaTeX) or add a `lang_pair_map` to the
   loader and re-run. Leaning towards loader enhancement; tracked.
4. **72B inclusion** — budget permitting, Tower-72B-awq is included; if
   the wall-time exceeds 24 h on a100, drop it and document.
5. **Direction reporting** — en→Indic vs X→en correlations are likely to
   diverge. Report separately or pooled? Current recommendation: both
   (pooled for headline, split for error analysis).

---

## 11. Change log

- **2026-04-19** — Document created. Inventory + pair matrix + per-pair
  fact sheets + submit recipes. Subject to update once the first full-
  matrix runs complete and exact row counts / wall-times are observed.
