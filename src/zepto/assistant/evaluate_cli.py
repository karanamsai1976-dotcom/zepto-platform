"""Console entry point for the retrieval evaluation: zepto-eval.

Prints a report to stdout rather than logging it, because the audience is a
person comparing this run against a previous one, not a log aggregator.
"""

from __future__ import annotations

from pathlib import Path

from zepto.assistant.evaluation import (
    RetrievalReport,
    RoutingReport,
    evaluate_abstention,
    evaluate_retrieval,
    evaluate_routing,
    load_cases,
)
from zepto.assistant.retrieval import VectorStore, load_corpus
from zepto.assistant.settings import AssistantSettings, get_assistant_settings
from zepto.core.logging import configure_logging

DEFAULT_CASES = Path("data/eval/retrieval_cases.jsonl")


def format_retrieval(report: RetrievalReport) -> str:
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
                f"    - {failure.query[:58]:<58} expected={'/'.join(failure.expected)} "
                f"got={'/'.join(failure.retrieved[:2])} rank={position}"
            )
    return "\n".join(lines)


def format_routing(report: RoutingReport) -> str:
    lines = [
        "Routing (keyword classifier)",
        f"  cases                {report.cases}",
        f"  accuracy             {report.accuracy:.1%}",
        f"  in-scope recall      {report.in_scope_recall:.1%}",
        f"  out-of-scope recall  {report.out_of_scope_recall:.1%}",
    ]
    if report.misrouted:
        lines.append(f"  misrouted            {len(report.misrouted)}")
        for case in report.misrouted:
            expected = "policy" if case.expected else "general"
            lines.append(
                f"    - {case.query[:58]:<58} expected={expected} got={case.routed_intent}"
            )
    return "\n".join(lines)


def run(settings: AssistantSettings | None = None, cases_path: Path | None = None) -> None:
    """Evaluate retrieval, routing, and abstention, and print the results."""
    resolved = settings or get_assistant_settings()
    path = cases_path if cases_path is not None else DEFAULT_CASES

    cases = load_cases(path)
    store = VectorStore(settings=resolved)
    if store.count() == 0:
        store.ingest(load_corpus(resolved.corpus_dir))

    retrieval = evaluate_retrieval(store, cases, k=resolved.top_k)
    routing = evaluate_routing(cases, resolved.policy_keywords)
    abstain_rate, leaked = evaluate_abstention(
        store, cases, min_relevance=resolved.min_relevance, k=resolved.top_k
    )

    print(f"\nEvaluation set: {path} ({len(cases)} cases)\n")
    print(format_retrieval(retrieval))
    print()
    print(format_routing(routing))
    print()
    print("Abstention (out-of-scope cases)")
    print(f"  correctly declined   {abstain_rate:.1%}")
    print(f"  relevance floor      {resolved.min_relevance}")
    for case in leaked:
        print(f"    - would have answered: {case.query[:60]}")
    print()


def main() -> None:
    """Console entry point: zepto-eval."""
    configure_logging(level="WARNING")
    run()
