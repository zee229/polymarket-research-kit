"""Every test runs offline against a fresh temporary data dir."""

import pytest

from pmrk.config import ENV_DATA_DIR


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    return tmp_path
