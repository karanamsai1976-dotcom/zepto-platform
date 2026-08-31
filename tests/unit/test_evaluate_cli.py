"""Tests for the evaluation console output."""

from __future__ import annotations

from pathlib import Path

import pytest

from zepto.assistant.evaluate_cli import format_retrieval, format_routing, main, run
from zepto.assistant.evaluation import CaseOutcome, RetrievalReport, RoutingReport
from zepto.assistant.settings import AssistantSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
CASES_PATH = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"


def test_retrieval_report_shows_the_headline_metrics() -> None:
    report = RetrievalReport(
        cases=29,
        k=3,
        hit_rate_at_1=0.862,
        hit_rate_at_k=0.966,
        mean_reciprocal_rank=0.914,
    )

    rendered = format_retrieval(report)

    assert "86.2%" in rendered
    assert "96.6%" in rendered
    assert "0.914" in rendered


def test_retrieval_report_lists_failing_cases() -> None:
    """A score says something regressed; the cases say what to do about it."""
    report = RetrievalReport(
        cases=1,
        k=3,
        hit_rate_at_1=0.0,
        hit_rate_at_k=1.0,
        mean_reciprocal_rank=0.5,
        failures=[
            CaseOutcome(
                query="How fast will my groceries arrive?",
                expected=("doc_01",),
                retrieved=("doc_02", "doc_01"),
                rank=2,
                routed_intent="",
            )
        ],
    )

    rendered = format_retrieval(report)

    assert "How fast will my groceries arrive?" in rendered
    assert "doc_01" in rendered
    assert "rank=2" in rendered


def test_retrieval_report_marks_a_document_that_never_appeared() -> None:
    report = RetrievalReport(
        cases=1,
        k=3,
        hit_rate_at_1=0.0,
        hit_rate_at_k=0.0,
        mean_reciprocal_rank=0.0,
        failures=[
            CaseOutcome(
                query="Is there a number I can ring?",
                expected=("doc_08",),
                retrieved=("doc_07", "doc_01"),
                rank=None,
                routed_intent="",
            )
        ],
    )

    assert "rank=absent" in format_retrieval(report)


def test_routing_report_shows_both_recalls() -> None:
    report = RoutingReport(
        cases=34,
        accuracy=0.176,
        in_scope_recall=0.034,
        out_of_scope_recall=1.0,
    )

    rendered = format_routing(report)

    assert "in-scope recall" in rendered
    assert "3.4%" in rendered
    assert "100.0%" in rendered


def test_routing_report_lists_misrouted_cases() -> None:
    report = RoutingReport(
        cases=1,
        accuracy=0.0,
        in_scope_recall=0.0,
        out_of_scope_recall=0.0,
        misrouted=[
            CaseOutcome(
                query="How much does shipping cost?",
                expected=("doc_01",),
                retrieved=(),
                rank=None,
                routed_intent="general_question",
            )
        ],
    )

    rendered = format_routing(report)

    assert "How much does shipping cost?" in rendered
    assert "expected=policy" in rendered


def test_run_evaluates_the_committed_set_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = AssistantSettings(corpus_dir=CORPUS_DIR, chroma_dir=tmp_path / "chroma")

    run(settings=settings, cases_path=CASES_PATH)

    output = capsys.readouterr().out
    assert "Retrieval (in-scope cases only)" in output
    assert "Routing (keyword classifier)" in output
    assert "Abstention (out-of-scope cases)" in output


def test_console_entry_point_configures_logging_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zepto.assistant import evaluate_cli

    events: list[str] = []
    monkeypatch.setattr(
        evaluate_cli, "configure_logging", lambda level=None: events.append("configured")
    )
    monkeypatch.setattr(evaluate_cli, "run", lambda: events.append("ran"))

    main()

    assert events == ["configured", "ran"]
