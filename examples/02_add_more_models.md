# 02 — Adding models to a run

Every entry in `configs/models/*.yaml` is addressable as
`<family>/<name>`. To add an already-registered model to your run, drop a
line like `- ref: comet/wmt22-cometkiwi-da` under `scorers:`.

## Pick your own subset

Rather than running the full matrix, pick a curated list of comparable
models. Copy `configs/runs/surrey_legal_recommended.yaml` as a template:

```yaml
run:
  id: my_subset

dataset: !include ../datasets/surrey_legal.yaml

scorers:
  - ref: comet/wmt22-cometkiwi-da
  - ref: comet/wmt23-cometkiwi-da-xl
    overrides:
      batch_size: 8                  # drop if you're on a 40GB card
  - ref: comet/xcomet-xl-qe          # gives error spans in extras
  - ref: tower/towerinstruct-7b-v0.2
  - ref: tower/tower-plus-9b
  - ref: tower/tower-plus-9b-mqm     # MQM flavour of the same model

output:
  root: outputs
  formats: [tsv, jsonl, summary]
```

Run:

```bash
mt-metrix score --config configs/runs/my_subset.yaml
```

## List what's available

```bash
mt-metrix list-models                  # everything
mt-metrix list-models --family tower   # just Tower
mt-metrix list-models --family comet   # just COMET
```

The first column is what you use under `ref:`; the second is the HF model
ID each one maps to.

## Inline scorers (no catalogue entry needed)

If you want to point at a model that isn't in the catalogue, specify it
inline under `scorers:`:

```yaml
scorers:
  - family: comet
    name: my-finetune
    model: MyOrg/comet-domain-tuned
    params:
      batch_size: 32
      gpus: 1
      progress_bar: true

  - family: tower
    name: my-tower
    model: MyOrg/my-tower
    params:
      prompt_mode: gemba-da
      backend: vllm
      tensor_parallel_size: 1
```

Any per-family knob documented in `configs/models/<family>.yaml` is
accepted under `params:`.

## Overrides

`overrides:` merges into the catalogue entry's `params:`, which is handy for
one-off changes without editing the catalogue file.

```yaml
- ref: comet/wmt23-cometkiwi-da-xxl
  overrides:
    batch_size: 4      # OOM-proofing on a 40GB card
```

## CLI-level overrides

You can splat values over the top-level config from the command line:

```bash
mt-metrix score \
  --config configs/runs/surrey_legal_cometkiwi.yaml \
  --override run.id=my-custom-id \
  --override output.root=/tmp/mt-metrix-out
```

These stack on top of whatever's in the YAML and land in the snapshot
written to `outputs/<run_id>/config.yaml`.
