# Model catalogue

All models currently registered in mt-metrix. Run
`mt-metrix list-models [--family <family>]` for the live list.

## COMET (`configs/models/comet.yaml`)

Reference-based DA metrics (`needs_reference: true`):

| ref                         | HF model                              | Notes                                    |
|-----------------------------|---------------------------------------|------------------------------------------|
| `comet/wmt22-comet-da`      | `Unbabel/wmt22-comet-da`              | WMT22 SOTA DA; XLM-R-large               |
| `comet/eamt22-cometinho-da` | `Unbabel/eamt22-cometinho-da`         | Distilled; cheap enough for smoke tests  |
| `comet/wmt20-comet-da`      | `Unbabel/wmt20-comet-da`              | Historical baseline                      |
| `comet/wmt21-comet-da`      | `Unbabel/wmt21-comet-da-marian`       | Marian port                              |

Reference-free QE (`needs_reference: false`):

| ref                               | HF model                            | Notes                         |
|-----------------------------------|-------------------------------------|-------------------------------|
| `comet/wmt22-cometkiwi-da`        | `Unbabel/wmt22-cometkiwi-da`        | InfoXLM-large. **Gated.**     |
| `comet/wmt23-cometkiwi-da-xl`     | `Unbabel/wmt23-cometkiwi-da-xl`     | XLM-R-XL 3.5B. **Gated.**     |
| `comet/wmt23-cometkiwi-da-xxl`    | `Unbabel/wmt23-cometkiwi-da-xxl`    | XLM-R-XXL 10.7B. **Gated.**   |
| `comet/wmt20-comet-qe-da`         | `Unbabel/wmt20-comet-qe-da`         | WMT20 QE baseline             |
| `comet/wmt21-comet-qe-da`         | `Unbabel/wmt21-comet-qe-da-marian`  |                               |
| `comet/wmt21-comet-qe-mqm`        | `Unbabel/wmt21-comet-qe-mqm-marian` | MQM-trained                   |

Explainable (XCOMET — returns error spans):

| ref                   | HF model              | Mode        | Notes                            |
|-----------------------|-----------------------|-------------|----------------------------------|
| `comet/xcomet-xl`     | `Unbabel/XCOMET-XL`   | ref-based   | **Gated.** `output_seg_err_spans`|
| `comet/xcomet-xxl`    | `Unbabel/XCOMET-XXL`  | ref-based   | **Gated.**                       |
| `comet/xcomet-xl-qe`  | `Unbabel/XCOMET-XL`   | QE (no ref) | Same checkpoint, ref omitted     |
| `comet/xcomet-xxl-qe` | `Unbabel/XCOMET-XXL`  | QE (no ref) |                                  |

### Gated models — licence acceptance required

CometKiwi and XCOMET checkpoints are gated. For each, visit the model page on
HuggingFace Hub, click "Agree and access" while logged in as the account
whose token sits in `~/.hf_token`, then confirm with
`huggingface-cli whoami`.

## Tower (`configs/models/tower.yaml`)

All Tower models are **CC-BY-NC-4.0** — research use only. `prompt_mode`
switches between the three parsers; each variant is a separate catalogue
entry so a single run can include e.g. both `towerinstruct-7b-v0.2` (GEMBA-DA)
and `towerinstruct-7b-v0.2-mqm` (GEMBA-MQM).

### GEMBA-DA (single 0–100 float, `max_tokens=50`)

| ref                                 | HF model                                   | Size | TP | Notes                             |
|-------------------------------------|--------------------------------------------|------|----|-----------------------------------|
| `tower/tower-plus-2b`               | `Unbabel/Tower-Plus-2B`                    | 2B   | 1  | Gemma2 backbone; smoke test       |
| `tower/towerinstruct-7b-v0.1`       | `Unbabel/TowerInstruct-7B-v0.1`            | 7B   | 1  | Jan 2024                          |
| `tower/towerinstruct-7b-v0.2`       | `Unbabel/TowerInstruct-7B-v0.2`            | 7B   | 1  | **Recommended 7B default**        |
| `tower/towerinstruct-mistral-7b-v0.2`| `Unbabel/TowerInstruct-Mistral-7B-v0.2`   | 7B   | 1  | Mistral base                      |
| `tower/towerinstruct-wmt24-chat-7b` | `Unbabel/TowerInstruct-WMT24-Chat-7B`      | 7B   | 1  | WMT24 variant                     |
| `tower/towerbase-7b-v0.1`           | `Unbabel/TowerBase-7B-v0.1`                | 7B   | 1  | NOT instruction-tuned             |
| `tower/towerinstruct-13b-v0.1`      | `Unbabel/TowerInstruct-13B-v0.1`           | 13B  | 2  |                                   |
| `tower/towerbase-13b-v0.1`          | `Unbabel/TowerBase-13B-v0.1`               | 13B  | 2  |                                   |
| `tower/tower-plus-9b`               | `Unbabel/Tower-Plus-9B`                    | 9B   | 2  | Gemma2 backbone; **recommended**  |
| `tower/tower-plus-72b`              | `Unbabel/Tower-Plus-72B`                   | 72B  | 4  | Qwen2 backbone; 4× A100 80GB      |

### GEMBA-MQM (severity-weighted errors, `max_tokens=384`)

| ref                                     | HF model                                 |
|-----------------------------------------|------------------------------------------|
| `tower/towerinstruct-7b-v0.2-mqm`       | `Unbabel/TowerInstruct-7B-v0.2`          |
| `tower/towerinstruct-13b-v0.1-mqm`      | `Unbabel/TowerInstruct-13B-v0.1`         |
| `tower/tower-plus-9b-mqm`               | `Unbabel/Tower-Plus-9B`                  |
| `tower/tower-plus-72b-mqm`              | `Unbabel/Tower-Plus-72B`                 |

### Tower-native (system+user chat framing)

| ref                                    | HF model                          |
|----------------------------------------|-----------------------------------|
| `tower/towerinstruct-7b-v0.2-native`   | `Unbabel/TowerInstruct-7B-v0.2`   |

## sacrebleu (`configs/models/sacrebleu.yaml`)

All reference-based — auto-skipped if the dataset has no reference column.

| ref                    | Notes                                                   |
|------------------------|---------------------------------------------------------|
| `sacrebleu/bleu`       | WMT defaults: `tokenize=13a`, `smooth_method=exp`       |
| `sacrebleu/chrf`       | `char_order=6`, `word_order=0`, `beta=2`                |
| `sacrebleu/chrf++`     | `char_order=6`, `word_order=2` (WMT default)            |
| `sacrebleu/ter`        | `normalized=false`, `case_sensitive=true`               |

## Adding a new entry

Edit the catalogue file under `configs/models/<family>.yaml`:

```yaml
family: comet
models:
  my-custom-comet:
    model: MyOrg/my-custom-comet
    needs_reference: true
    notes: "In-house COMET fine-tune on domain X."
    params:
      batch_size: 32
      gpus: 1
```

Then reference it in a run config as `comet/my-custom-comet`.

For a whole new family (e.g. OpenKiwi), create
`src/mt_metrix/scorers/openkiwi.py` conforming to the `Scorer` protocol, add
its import to `_bootstrap()` in `scorers/registry.py`, and create
`configs/models/openkiwi.yaml`.
