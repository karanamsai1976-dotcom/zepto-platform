"""Measured evaluation of retrieval and routing quality.

Until now the only evidence this system worked was a handful of queries checked
by hand. That is enough to catch a system that is broken, and useless for
telling whether a change made it better or worse. This module turns that into
numbers against a labelled set.

Three things are measured separately, because they fail for different reasons
and have different fixes.

Retrieval asks: when a question reaches the index, does the right document come
back? Reported as hit rate at 1, hit rate at k, and mean reciprocal rank.

Routing asks: does the keyword classifier send in-scope questions to retrieval
and out-of-scope questions away from it? Retrieval can be excellent and still
answer nothing useful if the router never calls it.

Abstention asks: when the corpus genuinely does not cover a question, does the
assistant decline rather than answer from the nearest unrelated document?

Failing cases are returned alongside the scores. A number tells you something
regressed; the cases tell you what to do about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from zepto.assistant.graph import GENERAL_INTENT, POLICY_INTENT, classify
from zepto.assistant.retrieval import RetrievedChunk
from zepto.core.errors import DatasetError
from zepto.core.logging import get_logger

logger = get_logger(__name__)


class Searcher(Protocol):
    """The retrieval operation an evaluation needs."""

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class EvalCase:
    """One labelled question.

    An empty expected list marks a question the corpus does not cover, where the
    correct behaviour is to decline rather than retrieve. Several documents may
    be listed where the corpus genuinely overlaps -- damaged items are described
    in both the returns policy and the damaged-goods policy, and treating either
    as wrong would measure the labelling rather than the system.
    """

    query: str
    expected: tuple[str, ...]

    @property
    def is_in_scope(self) -> bool:
        return bool(self.expected)


@dataclass(frozen=True)
class CaseOutcome:
    """What actually happened for one case."""

    query: str
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]
    rank: int | None
    routed_intent: str


@dataclass(frozen=True)
class RetrievalReport:
    """Retrieval quality over the in-scope cases."""

    cases: int
    k: int
    hit_rate_at_1: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    failures: list[CaseOutcome] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingReport:
    """How reliably the classifier separates in-scope from out-of-scope."""

    cases: int
    accuracy: float
    in_scope_recall: float
    out_of_scope_recall: float
    misrouted: list[CaseOutcome] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    """Read labelled cases from a JSON Lines file."""
    if not path.exists():
        raise DatasetError("evaluation set not found", path=str(path))

    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            cases.append(EvalCase(query=payload["query"], expected=tuple(payload["expected"])))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DatasetError("malformed evaluation case", path=str(path), line=number) from exc

    if not cases:
        raise DatasetError("evaluation set is empty", path=str(path))

    return cases


def rank_of_first_expected(
    retrieved: list[RetrievedChunk], expected: tuple[str, ...]
) -> int | None:
    """One-based rank of the first acceptable document, or None if absent."""
    for position, chunk in enumerate(retrieved, start=1):
        if chunk.document_id in expected:
            return position
    return None


def evaluate_retrieval(
    store: Searcher,
    cases: list[EvalCase],
    k: int = 3,
) -> RetrievalReport:
    """Score retrieval over the in-scope cases.

    Out-of-scope cases are excluded here: there is no correct document to find,
    so including them would drag the score down for behaviour that is handled
    correctly elsewhere. They are measured by the routing and abstention checks.
    """
    in_scope = [case for case in cases if case.is_in_scope]
    if not in_scope:
        raise DatasetError("evaluation set contains no in-scope cases")

    outcomes: list[CaseOutcome] = []
    reciprocal_ranks: list[float] = []
    hits_at_1 = 0
    hits_at_k = 0

    for case in in_scope:
        retrieved = store.search(case.query, top_k=k)
        rank = rank_of_first_expected(retrieved, case.expected)

        outcome = CaseOutcome(
            query=case.query,
            expected=case.expected,
            retrieved=tuple(chunk.document_id for chunk in retrieved),
            rank=rank,
            routed_intent="",
        )
        outcomes.append(outcome)

        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(in_scope)
    return RetrievalReport(
        cases=total,
        k=k,
        hit_rate_at_1=hits_at_1 / total,
        hit_rate_at_k=hits_at_k / total,
        mean_reciprocal_rank=sum(reciprocal_ranks) / total,
        failures=[outcome for outcome in outcomes if outcome.rank != 1],
    )


def evaluate_routing(cases: list[EvalCase], keywords: tuple[str, ...]) -> RoutingReport:
    """Score the intent classifier against the labelled scope of each case.

    Reported as two recalls rather than accuracy alone, because the two error
    types cost different things: sending an in-scope question away means a
    customer gets no answer at all, while sending an out-of-scope question to
    retrieval is caught downstream by the relevance floor.
    """
    misrouted: list[CaseOutcome] = []
    in_scope_correct = 0
    out_of_scope_correct = 0
    in_scope_total = 0
    out_of_scope_total = 0

    for case in cases:
        intent = classify(case.query, keywords)
        expected_intent = POLICY_INTENT if case.is_in_scope else GENERAL_INTENT

        if case.is_in_scope:
            in_scope_total += 1
        else:
            out_of_scope_total += 1

        if intent == expected_intent:
            if case.is_in_scope:
                in_scope_correct += 1
            else:
                out_of_scope_correct += 1
        else:
            misrouted.append(
                CaseOutcome(
                    query=case.query,
                    expected=case.expected,
                    retrieved=(),
                    rank=None,
                    routed_intent=intent,
                )
            )

    total = len(cases)
    return RoutingReport(
        cases=total,
        accuracy=(in_scope_correct + out_of_scope_correct) / total if total else 0.0,
        in_scope_recall=in_scope_correct / in_scope_total if in_scope_total else 0.0,
        out_of_scope_recall=(
            out_of_scope_correct / out_of_scope_total if out_of_scope_total else 0.0
        ),
        misrouted=misrouted,
    )


def evaluate_abstention(
    store: Searcher,
    cases: list[EvalCase],
    min_relevance: float,
    k: int = 3,
) -> tuple[float, list[CaseOutcome]]:
    """Fraction of out-of-scope questions whose best match falls below the floor.

    This measures whether the relevance floor is set somewhere useful. A floor
    so low that nothing is ever declined provides no protection; one so high
    that real questions are declined makes the assistant useless.
    """
    out_of_scope = [case for case in cases if not case.is_in_scope]
    if not out_of_scope:
        return 1.0, []

    leaked: list[CaseOutcome] = []
    for case in out_of_scope:
        retrieved = store.search(case.query, top_k=k)
        best = retrieved[0].relevance if retrieved else 0.0

        if best >= min_relevance:
            leaked.append(
                CaseOutcome(
                    query=case.query,
                    expected=case.expected,
                    retrieved=tuple(chunk.document_id for chunk in retrieved),
                    rank=None,
                    routed_intent="",
                )
            )

    return (len(out_of_scope) - len(leaked)) / len(out_of_scope), leaked
