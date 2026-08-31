"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest

from zepto.core.settings import get_core_settings


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate every test from the developer's machine.

    Two sources of contamination are removed:
      1. ZEPTO_* variables exported in the shell.
      2. A local .env file, which pydantic-settings resolves relative to the
         current working directory -- so we run from an empty temp directory.

    Without this, a test could pass on one machine and fail on another purely
    because of local environment state.
    """
    for key in list(os.environ):
        if key.startswith("ZEPTO_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.chdir(tmp_path)
    get_core_settings.cache_clear()
