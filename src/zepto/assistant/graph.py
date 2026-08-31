"""Question answering as an explicit state graph.

Two nodes and one decision: retrieve, then either answer from what came back or
decline because the corpus does not cover the question.

Routing is by retrieval relevance, not keywords. v1 classified intent by testing
the question against eight substrings, and this rebuild carried that over
unchanged until an evaluation set was built. Measured against 29 real customer
questions, keyword routing had 3.4% recall: it sent 28 of them to "I can only
answer questions about Zepto policies", because customers ask "How much does
shipping cost?" rather than "What is the delivery fee?". The four queries used
for manual checking had all happened to contain keywords, which is precisely how
the defect survived.

Relevance separates the same questions cleanly. Across the evaluation set,
in-scope questions score 0.186 to 0.567 and out-of-scope questions 0.000 to
0.089, with no overlap. The floor sits between those bands.

The cost is that every question now embeds before it can be declined, so an
off-topic question takes roughly 150ms rather than returning instantly. That is
a good trade for answering 29 questions instead of one.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings, get_assistant_settings
from zepto.core.logging import get_logger

logger = get_logger(__name__)


class Retriever(Protocol):
    """The single operation the graph needs from a vector store.

    Declared structurally so tests can supply a stub with controlled relevance
    scores, which is the only practical way to exercise the decline path
    deterministically.
    """

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


ANSWERED_INTENT = "policy_question"
DECLINED_INTENT = "out_of_scope"

DECLINE_ANSWER = (
    "I could not find a Zepto policy that covers that. Please rephrase, or "
    "contact support directly if it is urgent."
)


class GraphState(TypedDict, total=False):
    """State passed between nodes."""

    query: str
    intent: str
    chunks: list[RetrievedChunk]
    answer: str
    confidence: float


def compose_policy_answer(chunks: list[RetrievedChunk]) -> str:
    """Assemble a deterministic answer from the best matching policy text.

    Quotes the matched policy in full. v1 truncated at 200 characters, which
    routinely cut sentences mid-word.
    """
    best = chunks[0]
    return f"According to Zepto's policy: {best.text}"


def build_graph(
    store: Retriever,
    settings: AssistantSettings | None = None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Build the compiled question-answering graph.

    The vector store is injected rather than constructed here, so tests can
    supply one backed by a temporary index and the API can share a single warm
    store across requests.
    """
    resolved = settings or get_assistant_settings()

    def retrieve(state: GraphState) -> GraphState:
        chunks = store.search(state["query"])
        best = chunks[0].relevance if chunks else 0.0
        in_scope = bool(chunks) and best >= resolved.min_relevance

        logger.info(
            "scope_decided",
            in_scope=in_scope,
            best_relevance=round(best, 4),
            floor=resolved.min_relevance,
            query_length=len(state["query"]),
        )
        return {
            "chunks": chunks if in_scope else [],
            "confidence": best,
            "intent": ANSWERED_INTENT if in_scope else DECLINED_INTENT,
        }

    def answer(state: GraphState) -> GraphState:
        chunks = state["chunks"]

        if resolved.mock_llm:
            return {"answer": compose_policy_answer(chunks)}

        from zepto.assistant.llm import generate_grounded_answer

        return {"answer": generate_grounded_answer(state["query"], chunks, settings=resolved)}

    def decline(state: GraphState) -> GraphState:
        return {"answer": DECLINE_ANSWER}

    def route(state: GraphState) -> Literal["answer", "decline"]:
        return "answer" if state["intent"] == ANSWERED_INTENT else "decline"

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("decline", decline)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route, {"answer": "answer", "decline": "decline"})
    graph.add_edge("answer", END)
    graph.add_edge("decline", END)

    return graph.compile()
