"""Core application settings.

Configuration is read from the environment exactly once, here, rather than
through scattered os.environ lookups. Module-specific settings live alongside
their module (for example, zepto.ingestion.settings) so this package never
needs to know which modules exist.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]


class CoreSettings(BaseSettings):
    """Settings shared by every module in the platform."""

    model_config = SettingsConfigDict(
        env_prefix="ZEPTO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Field(
        default="development",
        description="Deployment environment. Invalid values fail at startup.",
    )
    log_level: str = Field(
        default="INFO",
        description="Minimum severity emitted by the logger.",
    )
    log_json: bool = Field(
        default=False,
        description="Emit machine-readable JSON logs. Enable in production, "
        "leave off locally for human-readable output.",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for local data artifacts.",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_core_settings() -> CoreSettings:
    """Return the process-wide settings, parsed from the environment once.

    Tests can reset this with get_core_settings.cache_clear().
    """
    return CoreSettings()
