"""Tests for the evaluation harness itself.

An evaluation that is wrong is worse than none, because it produces confident
numbers. These verify the metrics compute what they claim on cases with known
answers, before any conclusion is drawn from them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zepto.assistant.evaluation import (
    EvalCase,
    evaluate_abstention,
    evaluate_retrieval,
    evaluate_routing,
    load_cases,
    rank_of_first_expected,
)
from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings
from zepto.core.errors import DatasetError

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"


class ScriptedSearcher:
    """Returns a fixed ranking per query, so metrics have known answers."""

    def __init__(self, rankings: dict[str, list[tuple[str, float]]]) -> None:
        self.rankings = rankings

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        ranked = self.rankings.get(query, [])
        return [
            RetrievedChunk(
                document_id=document_id,
                chunk_id=f"{document_id}#0",
                text="text",
                relevance=relevance,
            )
            for document_id, relevance in ranked[: top_k or len(ranked)]
        ]


# --- loading ---


def test_committed_evaluation_set_loads() -> None:
    cases = load_cases(CASES_PATH)

    assert len(cases) >= 30
    assert any(case.is_in_scope for case in cases)
    assert any(not case.is_in_scope for case in cases)


def test_every_expected_document_exists_in_the_corpus() -> None:
    """A label pointing at a document that is not indexed would score as a
    permanent failure and quietly misrepresent retrieval quality."""
    corpus_ids = {path.stem for path in (REPO_ROOT / "data" / "corpus").glob("*.txt")}
    cases = load_cases(CASES_PATH)

    for case in cases:
        for document_id in case.expected:
            assert document_id in corpus_ids, f"{case.query} labels unknown {document_id}"


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_cases(tmp_path / "absent.jsonl")


def test_malformed_line_names_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "ok", "expected": []}\nnot json\n', encoding="utf-8")

    with pytest.raises(DatasetError) as exc_info:
        load_cases(path)

    assert exc_info.value.context["line"] == 2


def test_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_cases(path)


# --- ranking primitive ---


def test_rank_is_one_based_and_finds_the_first_acceptable_document() -> None:
    retrieved = [
        RetrievedChunk("doc_05", "doc_05#0", "t", 0.7),
        RetrievedChunk("doc_01", "doc_01#0", "t", 0.5),
    ]

    assert rank_of_first_expected(retrieved, ("doc_01",)) == 2
    assert rank_of_first_expected(retrieved, ("doc_05",)) == 1
    assert rank_of_first_expected(retrieved, ("doc_09",)) is None


def test_any_listed_document_counts_as_a_hit() -> None:
    """Where the corpus genuinely overlaps, either document is a correct answer."""
    retrieved = [RetrievedChunk("doc_02", "doc_02#0", "t", 0.6)]

    assert rank_of_first_expected(retrieved, ("doc_06", "doc_02")) == 1


# --- retrieval metrics ---


def test_perfect_retrieval_scores_one() -> None:
    cases = [EvalCase("a", ("doc_01",)), EvalCase("b", ("doc_02",))]
    searcher = ScriptedSearcher({"a": [("doc_01", 0.9)], "b": [("doc_02", 0.9)]})

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report.hit_rate_at_1 == 1.0
    assert report.hit_rate_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.failures == []


def test_second_place_hits_count_for_k_but_not_for_one() -> None:
    cases = [EvalCase("a", ("doc_01",))]
    searcher = ScriptedSearcher({"a": [("doc_09", 0.9), ("doc_01", 0.5)]})

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report.hit_rate_at_1 == 0.0
    assert report.hit_rate_at_k == 1.0
    assert report.mean_reciprocal_rank == pytest.approx(0.5)
    assert report.failures[0].rank == 2


def test_complete_miss_scores_zero() -> None:
    cases = [EvalCase("a", ("doc_01",))]
    searcher = ScriptedSearcher({"a": [("doc_09", 0.9)]})

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report.hit_rate_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0
    assert report.failures[0].rank is None


def test_out_of_scope_cases_are_excluded_from_retrieval_scoring() -> None:
    """There is no correct document to find, so including them would penalise
    behaviour that is measured correctly elsewhere."""
    cases = [EvalCase("a", ("doc_01",)), EvalCase("b", ())]
    searcher = ScriptedSearcher({"a": [("doc_01", 0.9)], "b": [("doc_03", 0.1)]})

    report = evaluate_retrieval(searcher, cases, k=3)

    assert report.cases == 1
    assert report.hit_rate_at_1 == 1.0


def test_evaluating_with_no_in_scope_cases_is_refused() -> None:
    with pytest.raises(DatasetError):
        evaluate_retrieval(ScriptedSearcher({}), [EvalCase("a", ())], k=3)


# --- routing metrics ---


def test_routing_reports_both_recalls_separately() -> None:
    """The two error types cost different things, so one accuracy number hides
    which one is happening."""
    cases = [
        EvalCase("What is the delivery fee?", ("doc_01",)),
        EvalCase("How much does shipping cost?", ("doc_01",)),
        EvalCase("Who won the world cup?", ()),
    ]

    report = evaluate_routing(cases, ("delivery",))

    assert report.in_scope_recall == pytest.approx(0.5)
    assert report.out_of_scope_recall == 1.0
    assert len(report.misrouted) == 1
    assert report.misrouted[0].query == "How much does shipping cost?"


def test_routing_accuracy_is_over_all_cases() -> None:
    cases = [EvalCase("delivery question", ("doc_01",)), EvalCase("unrelated", ())]

    report = evaluate_routing(cases, ("delivery",))

    assert report.accuracy == 1.0


# --- abstention ---


def test_out_of_scope_below_the_floor_counts_as_declined() -> None:
    cases = [EvalCase("unrelated", ())]
    searcher = ScriptedSearcher({"unrelated": [("doc_01", 0.05)]})

    rate, leaked = evaluate_abstention(searcher, cases, min_relevance=0.25)

    assert rate == 1.0
    assert leaked == []


def test_out_of_scope_above_the_floor_is_reported_as_a_leak() -> None:
    """A floor set too low provides no protection, and this is how you find out."""
    cases = [EvalCase("unrelated", ())]
    searcher = ScriptedSearcher({"unrelated": [("doc_01", 0.8)]})

    rate, leaked = evaluate_abstention(searcher, cases, min_relevance=0.25)

    assert rate == 0.0
    assert leaked[0].query == "unrelated"


def test_abstention_is_vacuously_perfect_without_out_of_scope_cases() -> None:
    rate, leaked = evaluate_abstention(
        ScriptedSearcher({}), [EvalCase("a", ("doc_01",))], min_relevance=0.25
    )

    assert rate == 1.0
    assert leaked == []


def test_settings_supply_the_default_keywords() -> None:
    report = evaluate_routing(
        [EvalCase("What is the delivery fee?", ("doc_01",))],
        AssistantSettings().policy_keywords,
    )

    assert report.in_scope_recall == 1.0
