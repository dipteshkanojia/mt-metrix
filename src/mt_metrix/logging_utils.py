"""Project-wide logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logging with a rich handler for stdout and an optional file handler."""
    root = logging.getLogger()
    root.handlers.clear()

    level_int = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(level_int)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from rich.logging import RichHandler

        stream_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=False,
        )
    except ImportError:  # pragma: no cover
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(fmt)

    stream_handler.setLevel(level_int)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level_int)
        fh.setFormatter(fmt)
        root.addHandler(fh)
