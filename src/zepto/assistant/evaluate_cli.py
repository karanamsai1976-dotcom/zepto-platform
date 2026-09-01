"""Console entry point for the retrieval evaluation: zepto-eval.

Prints a report to stdout rather than logging it, because the audience is a
person comparing this run against a previous one, not a log aggregator.
"""

from __future__ import annotations

from pathlib import Path

from zepto.assistant.evaluation import (
    DEV_SPLIT,
    TEST_SPLIT,
    EvalCase,
    RetrievalReport,
    ScopeReport,
    evaluate_retrieval,
    evaluate_scope,
    load_cases,
)
from zepto.assistant.retrieval import VectorStore, load_corpus
from zepto.assistant.settings import AssistantSettings, get_assistant_settings
from zepto.core.logging import configure_logging

DEFAULT_CASES = Path("data/eval/retrieval_cases.jsonl")


def format_retrieval(report: RetrievalReport) -> str:
    """Render retrieval quality, listing anything not ranked first."""
    lines = [
        "Retrieval (in-scope cases only)",
        f"  cases                {report.cases}",
        f"  hit rate @1          {report.hit_rate_at_1:.1%}",
        f"  hit rate @{report.k}          {report.hit_rate_at_k:.1%}",
        f"  mean reciprocal rank {report.mean_reciprocal_rank:.3f}",
    ]
    if report.failures:
        lines.append(f"  not ranked first     {len(report.failures)}")
        for failure in report.failures:
            position = failure.rank if failure.rank else "absent"
            lines.append(
                f"    - {failure.query[:56]:<56} expected={'/'.join(failure.expected)} "
                f"got={'/'.join(failure.retrieved[:2])} rank={position}"
            )
    return "\n".join(lines)


def format_scope(report: ScopeReport) -> str:
    """Render the scope decision, separating the two kinds of mistake."""
    lines = [
        "Scope decision (relevance floor)",
        f"  cases                {report.cases}",
        f"  floor                {report.floor}",
        f"  accuracy             {report.accuracy:.1%}",
        f"  in-scope answered    {report.in_scope_recall:.1%}",
        f"  out-of-scope declined {report.out_of_scope_recall:.1%}",
    ]
    if report.wrongly_declined:
        lines.append(f"  wrongly declined     {len(report.wrongly_declined)}")
        for case in report.wrongly_declined:
            lines.append(f"    - {case.query[:70]}")
    if report.wrongly_answered:
        lines.append(f"  wrongly answered     {len(report.wrongly_answered)}")
        for case in report.wrongly_answered:
            lines.append(f"    - {case.query[:70]}")
    return "\n".join(lines)


def run(settings: AssistantSettings | None = None, cases_path: Path | None = None) -> None:
    """Evaluate retrieval and the scope decision, and print the results.

    Reported per split. dev is what thresholds and design choices were chosen
    against; test was held back from all of them, so it is the only number that
    estimates performance on questions nobody tuned for. Reporting only the
    combined figure would hide the difference, which is the one thing worth
    knowing.
    """
    resolved = settings or get_assistant_settings()
    path = cases_path if cases_path is not None else DEFAULT_CASES

    cases = load_cases(path)
    store = VectorStore(settings=resolved)
    if store.count() == 0:
        store.ingest(load_corpus(resolved.corpus_dir))

    print(f"\nEvaluation set: {path} ({len(cases)} cases)")

    rule = "=" * 62
    for split in (DEV_SPLIT, TEST_SPLIT):
        subset = [case for case in cases if case.split == split]
        if not subset:
            continue
        print(f"\n{rule}\n{split.upper()} split ({len(subset)} cases)\n{rule}\n")
        report_split(store, subset, resolved)
    print()


def report_split(store: VectorStore, cases: list[EvalCase], settings: AssistantSettings) -> None:
    """Print both reports for one subset of cases."""
    retrieval = evaluate_retrieval(store, cases, k=settings.top_k)
    scope = evaluate_scope(store, cases, min_relevance=settings.min_relevance, k=settings.top_k)

    print(format_retrieval(retrieval))
    print()
    print(format_scope(scope))


def main() -> None:
    """Console entry point: zepto-eval."""
    configure_logging(level="WARNING")
    run()
