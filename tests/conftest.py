"""Pytest config and shared fixtures for mt-metrix tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(autouse=True)
def _chdir_to_root(monkeypatch):
    """Tests reference fixture paths relative to the project root."""
    monkeypatch.chdir(ROOT)


def pytest_collection_modifyitems(config, items):
    """Skip `slow` tests unless MT_METRIX_RUN_SLOW=1 is set.

    Slow tests download large models (COMET, Tower) and are unsuitable for
    CI or the default local loop.
    """
    if os.environ.get("MT_METRIX_RUN_SLOW") == "1":
        return
    skip_slow = pytest.mark.skip(reason="slow test; set MT_METRIX_RUN_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
