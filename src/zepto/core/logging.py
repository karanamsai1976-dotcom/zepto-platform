"""Structured logging configuration.

v1 used print() for all output: no severity, no timestamp, no context, not
machine-parseable. Structured logging emits key-value events instead, so
production logs can be queried rather than grepped.

Call configure_logging() once at process startup. Use get_logger(__name__)
everywhere else.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import Processor

from zepto.core.settings import get_core_settings


def configure_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Configure structlog and bridge it to the standard library logger.

    Explicit arguments override settings, which is useful for tests and
    one-off scripts. Called with no arguments, behaviour comes entirely from
    the environment.
    """
    settings = get_core_settings()
    resolved_level = (level or settings.log_level).upper()
    resolved_json = settings.log_json if json_output is None else json_output

    # force=True so re-configuring actually takes effect; basicConfig is
    # otherwise a no-op once the root logger has handlers.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=resolved_level, force=True)

    processors: list[Processor] = [
        # Attaches context bound via bind_contextvars to every event emitted
        # downstream -- the basis for per-request tracing in Phase 3.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if resolved_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[resolved_level]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to the given name, conventionally __name__."""
    return structlog.stdlib.get_logger(name)
