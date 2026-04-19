"""Project-root conftest — ensures tests can import `mt_metrix` without an editable install.

Placed at the repo root so pytest picks it up before collecting any tests.
CI or users who have run `pip install -e .` won't rely on this; it just makes
the first-run experience smoother.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
