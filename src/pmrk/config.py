"""Runtime configuration shared by all modules."""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "PMRK_DATA_DIR"
USER_AGENT = "polymarket-research-kit/0.1"


def data_dir() -> Path:
    """Root of all downloaded and derived data: `$PMRK_DATA_DIR`, default `./data`."""
    return Path(os.environ.get(ENV_DATA_DIR, "data")).expanduser()
