"""Zepto Platform: ingestion, analytics, and a retrieval-augmented assistant."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zepto-platform")
except PackageNotFoundError:  # pragma: no cover - source checkout, not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
