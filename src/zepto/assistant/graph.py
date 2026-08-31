"""Question routing and answer generation, as an explicit state graph.

Three nodes: classify a question, answer it from retrieved policy text, or
decline because it is not about policy. Routing is a keyword heuristic and never
depends on the generation mode -- only what happens inside the answering nodes
differs between mock and real-LLM operation.

The addition over v1 is the abstain path. v1 answered from whatever came back,
however poor the match, and reported full confidence while doing it. Retrieval
always returns something: asking a vector index about football still yields the
closest policy document. Answering from that is worse than saying the corpus
does not cover the question, so a result below the relevance floor is declined.

Mock answers quote the matched policy in full rather than truncating it at a
fixed character count, which in v1 routinely cut sentences mid-word.
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
    scores, which is the only practical way to exercise the abstain path
    deterministically.
    """

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


POLICY_INTENT = "policy_question"
GENERAL_INTENT = "general_question"

GENERAL_ANSWER = "I can only answer questions about Zepto policies."
ABSTAIN_ANSWER = (
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


def classify(query: str, keywords: tuple[str, ...]) -> str:
    """Route by plain substring match, so 'cancellation' matches 'cancel'.

    Deliberately not an LLM call: routing must behave identically whether or not
    a language model is configured, and a keyword test is auditable in a way a
    model's judgement is not.
    """
    lowered = query.lower()
    return POLICY_INTENT if any(word in lowered for word in keywords) else GENERAL_INTENT


def compose_policy_answer(chunks: list[RetrievedChunk]) -> str:
    """Assemble a deterministic answer from the best matching policy text."""
    best = chunks[0]
    return f"According to Zepto's policy: {best.text}"


def build_graph(
    store: Retriever,
    settings: AssistantSettings | None = None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Build the compiled question-answering graph.

    The vector store is injected rather than constructed here, so tests can
    supply one backed by a temporary index and the API can share a single
    warm store across requests.
    """
    resolved = settings or get_assistant_settings()

    def classify_intent(state: GraphState) -> GraphState:
        intent = classify(state["query"], resolved.policy_keywords)
        logger.info("intent_classified", intent=intent, query_length=len(state["query"]))
        return {"intent": intent}

    def retrieve_and_answer(state: GraphState) -> GraphState:
        chunks = store.search(state["query"])

        if not chunks or chunks[0].relevance < resolved.min_relevance:
            best = chunks[0].relevance if chunks else 0.0
            logger.info(
                "abstained",
                best_relevance=round(best, 4),
                floor=resolved.min_relevance,
            )
            return {"chunks": [], "answer": ABSTAIN_ANSWER, "confidence": best}

        if resolved.mock_llm:
            answer = compose_policy_answer(chunks)
        else:
            from zepto.assistant.llm import generate_grounded_answer

            answer = generate_grounded_answer(state["query"], chunks, settings=resolved)

        return {"chunks": chunks, "answer": answer, "confidence": chunks[0].relevance}

    def direct_answer(state: GraphState) -> GraphState:
        return {"chunks": [], "answer": GENERAL_ANSWER, "confidence": 0.0}

    def route(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
        return "retrieve_and_answer" if state["intent"] == POLICY_INTENT else "direct_answer"

    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()
