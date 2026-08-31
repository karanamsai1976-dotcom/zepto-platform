"""Tests for logging configuration and structured event emission."""

import structlog

from zepto.core.logging import configure_logging, get_logger


def test_configure_logging_runs_with_defaults() -> None:
    configure_logging()


def test_configure_logging_accepts_overrides() -> None:
    configure_logging(level="DEBUG", json_output=True)
    configure_logging(level="INFO", json_output=False)


def test_events_carry_structured_fields() -> None:
    """Fields are emitted as data, not interpolated into a message string."""
    with structlog.testing.capture_logs() as logs:
        get_logger("test").info("book_scraped", title="Soumission", price_gbp=50.10)

    assert logs[0]["event"] == "book_scraped"
    assert logs[0]["title"] == "Soumission"
    assert logs[0]["price_gbp"] == 50.10


def test_bound_context_appears_on_every_subsequent_event() -> None:
    with structlog.testing.capture_logs() as logs:
        logger = get_logger("test").bind(run_id="abc123")
        logger.info("run_started")
        logger.info("run_finished")

    assert len(logs) == 2
    assert all(entry["run_id"] == "abc123" for entry in logs)


def test_configuration_includes_contextvars_merging() -> None:
    """Our processor chain must include merge_contextvars -- this is what makes
    per-request context propagate without threading it through arguments."""
    configure_logging()

    processors = structlog.get_config()["processors"]
    assert structlog.contextvars.merge_contextvars in processors


def test_merge_contextvars_attaches_bound_values() -> None:
    """The mechanism itself: values bound to the context appear in the event dict.

    Tested by invoking the processor directly rather than through
    structlog.testing.capture_logs(), because capture_logs replaces the whole
    processor chain and so bypasses the behaviour under test.
    """
    structlog.contextvars.clear_contextvars()
    try:
        structlog.contextvars.bind_contextvars(request_id="req-1")
        event_dict = structlog.contextvars.merge_contextvars(
            None, "info", {"event": "request_handled"}
        )
    finally:
        structlog.contextvars.clear_contextvars()

    assert event_dict["request_id"] == "req-1"


def test_json_mode_selects_json_renderer() -> None:
    configure_logging(json_output=True)

    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.processors.JSONRenderer)


def test_console_mode_selects_console_renderer() -> None:
    configure_logging(json_output=False)

    processors = structlog.get_config()["processors"]
    assert isinstance(processors[-1], structlog.dev.ConsoleRenderer)
