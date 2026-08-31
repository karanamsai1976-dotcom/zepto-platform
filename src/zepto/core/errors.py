"""Typed exception hierarchy for the platform.

Every error raised by this codebase derives from ZeptoError, so callers can
distinguish our failures from genuine bugs (which stay as built-in exceptions)
and handle them at the right level of granularity.

Each error carries an optional structured context, which pairs with structured
logging: details are attached as queryable fields rather than interpolated into
a message string that has to be parsed back out later.
"""

from __future__ import annotations

from typing import Any


class ZeptoError(Exception):
    """Base class for every error raised by this platform."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({details})"


class ConfigurationError(ZeptoError):
    """Configuration is missing or invalid. Unrecoverable at startup."""


# --- Ingestion ---


class IngestionError(ZeptoError):
    """Base class for ingestion pipeline failures."""


class FetchError(IngestionError):
    """A remote resource could not be retrieved, including after retries."""


class ParseError(IngestionError):
    """A resource was retrieved but its structure did not match expectations.

    Distinct from FetchError: the network worked, the page shape did not.
    """


class StorageError(ZeptoError):
    """Reading from or writing to a datastore failed."""


# --- Assistant ---


class AssistantError(ZeptoError):
    """Base class for support-assistant failures."""


class RetrievalError(AssistantError):
    """The vector store could not be queried."""


class GenerationError(AssistantError):
    """An answer could not be generated, including after retries."""
