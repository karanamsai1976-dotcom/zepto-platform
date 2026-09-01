"""Compare retrieval designs against the dev split before shipping one.

Every remaining retrieval failure is the right document arriving at rank 2 or 3
rather than not arriving at all -- hit@3 is 96.6% on dev and 100% on test. That
is a ranking problem, and two standard fixes target it:

  sentence   Index each sentence separately instead of each document. The policy
             documents are four-sentence paragraphs covering several topics, so a
             whole-document vector is an average of things the question does not
             ask about. "Phone support is not offered." as its own vector should
             sit much closer to "Is there a number I can ring?" than the
             paragraph containing it does.

  bm25       Fuse dense results with lexical BM25 by reciprocal rank. Dense
             retrieval matches meaning and misses exact rare terms; BM25 does the
             opposite.

There is a reason to doubt the second before running it. The evaluation set was
written deliberately to avoid corpus wording -- customers say "voucher" where the
corpus says "gift card", "ring" where it says "phone". A lexical retriever is
structurally disadvantaged on a set built that way, and it would be easy to ship
hybrid retrieval on the general reputation of the technique rather than on
evidence from this corpus. Hence measuring first.

Run:  python experiments/retrieval_ablation.py

Tuning happens against dev only. The test split is not read here.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zepto.assistant.evaluation import DEV_SPLIT, EvalCase, load_cases
from zepto.assistant.retrieval import Document, RetrievedChunk, VectorStore, load_corpus
from zepto.assistant.settings import AssistantSettings
from zepto.core.logging import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"
CORPUS = REPO_ROOT / "data" / "corpus"

#: Rank-fusion constant from the original RRF paper. It damps the influence of
#: the very top ranks so one confident-but-wrong list cannot dominate.
RRF_K = 60

#: BM25 term-frequency saturation and length normalisation. Standard defaults;
#: this corpus is far too small to fit them without fitting noise.
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "i",
        "you",
        "my",
        "me",
        "we",
        "our",
        "it",
        "its",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "and",
        "or",
        "if",
        "can",
        "could",
        "will",
        "would",
        "should",
        "my",
        "what",
        "when",
        "where",
        "how",
        "why",
        "who",
        "which",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "not",
        "no",
        "have",
        "has",
        "had",
        "get",
        "got",
        "any",
        "some",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped, crudely singularised.

    The singularisation is deliberately crude -- it exists so that "deals"
    matches "deals" and "items" matches "item", which is most of the benefit a
    real stemmer would give on eight short documents.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


class Bm25Index:
    """Lexical scoring over a small document set."""

    def __init__(self, documents: Sequence[tuple[str, str]]) -> None:
        self.ids = [identifier for identifier, _ in documents]
        self.tokens = [tokenize(text) for _, text in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.frequencies = [Counter(t) for t in self.tokens]

        appearances: Counter[str] = Counter()
        for token_set in (set(t) for t in self.tokens):
            appearances.update(token_set)
        total = len(self.tokens)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in appearances.items()
        }

    def ranked(self, query: str) -> list[str]:
        """Document ids, best lexical match first."""
        terms = tokenize(query)
        scored: list[tuple[float, str]] = []
        for index, identifier in enumerate(self.ids):
            frequency = self.frequencies[index]
            length = self.lengths[index]
            score = 0.0
            for term in terms:
                occurrences = frequency.get(term, 0)
                if not occurrences:
                    continue
                denominator = occurrences + BM25_K1 * (
                    1 - BM25_B + BM25_B * length / max(self.average_length, 1e-9)
                )
                score += self.idf.get(term, 0.0) * occurrences * (BM25_K1 + 1) / denominator
            if score > 0:
                scored.append((score, identifier))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [identifier for _, identifier in scored]


def split_sentences(text: str) -> list[str]:
    """Split a policy paragraph into sentences, keeping semicolon clauses whole."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def sentence_documents(documents: list[Document]) -> list[Document]:
    """One indexable unit per sentence, tagged with its parent document id."""
    expanded = []
    for document in documents:
        for number, sentence in enumerate(split_sentences(document.text)):
            expanded.append(
                Document(document_id=f"{document.document_id}::{number}", text=sentence)
            )
    return expanded


def parent_of(chunk_id: str) -> str:
    """The source document id behind an indexed unit."""
    return chunk_id.split("::")[0]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Merge several ranked id lists into one by reciprocal rank."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (RRF_K + position)
    return sorted(scores, key=lambda identifier: -scores[identifier])


def dedupe(ordered: Sequence[str]) -> list[str]:
    """First occurrence wins, order preserved."""
    seen: set[str] = set()
    result = []
    for item in ordered:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass
class Variant:
    """One retrieval design under test."""

    name: str
    sentence_level: bool
    use_bm25: bool


VARIANTS = [
    Variant("baseline (whole doc, dense)", sentence_level=False, use_bm25=False),
    Variant("sentence chunks, dense", sentence_level=True, use_bm25=False),
    Variant("whole doc, dense + bm25", sentence_level=False, use_bm25=True),
    Variant("sentence chunks, dense + bm25", sentence_level=True, use_bm25=True),
]


def build_store(documents: list[Document], directory: Path, name: str) -> VectorStore:
    settings = AssistantSettings(
        corpus_dir=CORPUS, chroma_dir=directory, collection_name=name, top_k=8
    )
    store = VectorStore(settings=settings)
    store.ingest(documents)
    return store


def evaluate(
    variant: Variant, cases: list[EvalCase], workspace: Path, k: int = 3
) -> dict[str, float]:
    """Hit rate and MRR for one design, and the cases it still gets wrong."""
    base = load_corpus(CORPUS)
    indexed = sentence_documents(base) if variant.sentence_level else base
    store = build_store(indexed, workspace / variant.name.replace(" ", "_"), "ablation")
    lexical = Bm25Index([(document.document_id, document.text) for document in indexed])

    in_scope = [case for case in cases if case.is_in_scope]
    first_hits = 0
    k_hits = 0
    reciprocal = 0.0
    failures: list[str] = []

    for case in in_scope:
        dense: list[RetrievedChunk] = store.search(case.query, top_k=8)
        dense_ids = dedupe([parent_of(chunk.document_id) for chunk in dense])

        if variant.use_bm25:
            lexical_ids = dedupe([parent_of(i) for i in lexical.ranked(case.query)])
            ordered = reciprocal_rank_fusion([dense_ids, lexical_ids])
        else:
            ordered = dense_ids

        top = ordered[:k]
        rank = next(
            (position for position, i in enumerate(top, start=1) if i in case.expected), None
        )
        if rank == 1:
            first_hits += 1
        if rank is not None:
            k_hits += 1
            reciprocal += 1.0 / rank
        else:
            failures.append(f"{case.query[:54]:<54} -> {'/'.join(top[:2])}")

    total = len(in_scope)
    return {
        "hit@1": first_hits / total,
        f"hit@{k}": k_hits / total,
        "mrr": reciprocal / total,
        "missed": failures,  # type: ignore[dict-item]
    }


def main() -> None:
    configure_logging(level="ERROR")
    cases = load_cases(CASES, split=DEV_SPLIT)
    workspace = Path(tempfile.mkdtemp(prefix="ablation-"))

    print(f"\nDev split: {len(cases)} cases, {sum(c.is_in_scope for c in cases)} in scope\n")
    print(f"{'variant':<32} {'hit@1':>7} {'hit@3':>7} {'MRR':>7}")
    print("-" * 56)

    results = {}
    try:
        for variant in VARIANTS:
            scores = evaluate(variant, cases, workspace)
            results[variant.name] = scores
            print(
                f"{variant.name:<32} {scores['hit@1']:>6.1%} "
                f"{scores['hit@3']:>6.1%} {scores['mrr']:>7.3f}"
            )

        # Not the same as "failures": a case ranked 2nd is wrong for hit@1 and
        # still appears nowhere below. These are the cases a design loses
        # entirely, which is the more serious kind and the one worth listing.
        print("\nCases where the expected document never reached the top 3:")
        for name, scores in results.items():
            print(f"\n  {name}")
            for line in scores["missed"]:  # type: ignore[index]
                print(f"    {line}")
            if not scores["missed"]:  # type: ignore[index]
                print("    none")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print()


if __name__ == "__main__":
    main()
