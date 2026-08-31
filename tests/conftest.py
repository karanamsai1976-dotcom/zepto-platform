"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest

from zepto.analytics.settings import get_analytics_settings
from zepto.assistant.settings import get_assistant_settings
from zepto.core.settings import get_core_settings
from zepto.ingestion.settings import get_ingestion_settings

#: Every cached settings accessor. Each caches the parsed environment for the
#: process, so all of them must be reset between tests or one test's overrides
#: leak into the next.
SETTINGS_CACHES = (
    get_core_settings,
    get_ingestion_settings,
    get_analytics_settings,
    get_assistant_settings,
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate every test from the developer's machine.

    Three sources of contamination are removed:
      1. ZEPTO_* variables exported in the shell.
      2. A local .env file, which pydantic-settings resolves relative to the
         current working directory -- so we run from an empty temp directory.
      3. Settings cached from an earlier test.

    Without this, a test could pass on one machine and fail on another purely
    because of local state, or pass alone and fail in a suite.
    """
    for key in list(os.environ):
        if key.startswith("ZEPTO_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.chdir(tmp_path)

    for cache in SETTINGS_CACHES:
        cache.cache_clear()
