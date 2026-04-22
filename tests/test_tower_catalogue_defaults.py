"""Pin the default ``tensor_parallel_size`` for every Tower catalogue entry.

Background (2026-04-21 AISURREY full-matrix regression): four Tower
scorers were silently skipped on a ``gpu:1`` allocation with::

    The number of required GPUs exceeds the total number of available
    GPUs in the placement group.

Root cause: the catalogue had ``tensor_parallel_size: 2`` for 13B and 9B
Tower variants. These fit comfortably at tp=1 on any 48 GB+ card (13B fp16
≈ 26 GB, 9B ≈ 18 GB) and don't *need* sharding at all on 80 GB A100.
Over-sharding in the catalogue made the runner request resources the
cluster rarely grants.

Policy going forward (captured as these assertions):

- 7B / 9B / 13B Tower variants: tp=1. They fit single-GPU on the cluster's
  default 48 GB and 80 GB cards. Anyone submitting on a 24 GB card for
  7B-only work can still shard via a per-run override.
- 72B: tp=4. Genuinely won't fit at tp<4 anywhere we have (needs 144 GB
  fp16; tp=2 on 2× 80 GB leaves no headroom for KV cache).

These tests parse the YAML directly so the catalogue is the single source
of truth and nobody has to remember to update a Python constant too.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TOWER_CATALOGUE = REPO_ROOT / "configs" / "models" / "tower.yaml"


@pytest.fixture(scope="module")
def catalogue() -> dict:
    with TOWER_CATALOGUE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["family"] == "tower"
    return data["models"]


# Entries that MUST ship tp=1. Keyed by catalogue entry name.
# If you add a new 7B/9B/13B Tower entry, add it here and default to tp=1.
SINGLE_GPU_ENTRIES = {
    "towerbase-7b-v0.1",
    "towerbase-13b-v0.1",
    "towerinstruct-7b-v0.1",
    "towerinstruct-7b-v0.2",
    "towerinstruct-13b-v0.1",
    "towerinstruct-mistral-7b-v0.2",
    "towerinstruct-wmt24-chat-7b",
    "tower-plus-2b",
    "tower-plus-9b",
    "towerinstruct-7b-v0.2-mqm",
    "towerinstruct-13b-v0.1-mqm",
    "tower-plus-9b-mqm",
    "towerinstruct-7b-v0.2-native",
}

# Entries that genuinely need tp=4.
QUAD_GPU_ENTRIES = {
    "tower-plus-72b",
    "tower-plus-72b-mqm",
}


@pytest.mark.parametrize("entry_name", sorted(SINGLE_GPU_ENTRIES))
def test_single_gpu_entry_defaults_to_tp1(catalogue, entry_name):
    entry = catalogue.get(entry_name)
    assert entry is not None, (
        f"Catalogue missing expected entry {entry_name!r}. If the entry was "
        "removed intentionally, drop it from SINGLE_GPU_ENTRIES in this test."
    )
    tp = entry.get("params", {}).get("tensor_parallel_size")
    assert tp == 1, (
        f"{entry_name}: tensor_parallel_size should be 1 (7B/9B/13B models "
        f"fit tp=1 on 48 GB+ cards), got {tp!r}. Raising tp here causes "
        f"silent skips on gpu:1 allocations — see "
        f"tests/test_tower_catalogue_defaults.py docstring."
    )


@pytest.mark.parametrize("entry_name", sorted(QUAD_GPU_ENTRIES))
def test_large_entry_keeps_tp4(catalogue, entry_name):
    entry = catalogue.get(entry_name)
    assert entry is not None
    tp = entry.get("params", {}).get("tensor_parallel_size")
    assert tp == 4, (
        f"{entry_name}: tensor_parallel_size should be 4 (72B fp16 ≈ 144 GB "
        f"genuinely needs 4× 80 GB A100), got {tp!r}."
    )


def test_every_entry_covered(catalogue):
    """No Tower catalogue entry is left untyped by this policy."""
    all_entries = set(catalogue.keys())
    classified = SINGLE_GPU_ENTRIES | QUAD_GPU_ENTRIES
    uncovered = all_entries - classified
    assert not uncovered, (
        f"New catalogue entries not classified by tp policy: {sorted(uncovered)}. "
        f"Add them to SINGLE_GPU_ENTRIES or QUAD_GPU_ENTRIES in this test."
    )


def test_mistral_has_disable_sliding_window(catalogue):
    """2026-04-21 regression: ``disable_sliding_window: true`` must be set
    on the Mistral-backbone Tower entry to bypass vLLM 0.6.x's TypeError.
    Without it, the scorer raises at LLM() init and the runner skips it."""
    entry = catalogue["towerinstruct-mistral-7b-v0.2"]
    params = entry.get("params", {})
    assert params.get("disable_sliding_window") is True, (
        "towerinstruct-mistral-7b-v0.2 must set disable_sliding_window: true — "
        "see configs/models/tower.yaml header and "
        "src/mt_metrix/scorers/tower.py::_load_vllm."
    )
    # max_model_len pin MUST match vLLM's sliding-window-derived cap
    # (4096). Setting it higher trips _get_and_verify_max_len's validator
    # even when disable_sliding_window is on — vLLM still uses
    # sliding_window to compute the derived bound. 4096 is 5x the
    # GEMBA-DA prompt length, 2.5x the GEMBA-MQM prompt length.
    assert params.get("max_model_len") == 4096, (
        "towerinstruct-mistral-7b-v0.2 must pin max_model_len=4096 — "
        "higher values trip vLLM's sliding-window-derived cap validator. "
        "See configs/models/tower.yaml for the full rationale."
    )
