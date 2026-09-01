"""Streamlit front end for the support assistant.

Lives at the repository root because Streamlit Community Cloud expects a main
file it can point at, and it is the only file in the project that has to sit
there.

It contains no retrieval, routing, or scoring logic. Every decision is made by
`zepto.assistant`, the same package the FastAPI service and the 433 tests use.
This is the same rule the notebooks follow, for the same reason: a UI that
computes anything is a second implementation that can quietly disagree with the
tested one, and the disagreement surfaces in front of a user rather than in CI.

The store and graph are built once per process. Rebuilding them per interaction
would reload the embedding model every time -- the exact defect this rebuild
removed from v1, reintroduced at a different layer.
"""

from __future__ import annotations

import streamlit as st

from zepto.assistant.graph import ANSWERED_INTENT, build_graph
from zepto.assistant.retrieval import VectorStore, load_corpus
from zepto.assistant.settings import get_assistant_settings
from zepto.core.logging import configure_logging

#: (button label, question). The label is written rather than derived from the
#: question: taking the first word gave "How", "Can", "How", "Do", "Who" -- two
#: of them identical and none of them naming a topic. A button has to say what
#: pressing it does.
EXAMPLES = (
    ("Delivery", "How much is the delivery fee?"),
    ("Returns", "Can I return an opened bottle of shampoo?"),
    ("Tracking", "How do I track where my rider is?"),
    ("Support", "Do you have a helpline I can dial?"),
    ("Off-topic", "Who won the world cup?"),
)


@st.cache_resource(show_spinner="Loading the policy index…")
def load_assistant() -> object:
    """Build the retrieval stack once and reuse it for every question.

    Cached at resource scope, not data scope: this is a live object holding an
    embedding model and a database handle, and it is shared across sessions
    rather than recomputed per user.
    """
    configure_logging(level="WARNING")
    settings = get_assistant_settings()

    store = VectorStore(settings=settings)
    if store.count() == 0:
        store.ingest(load_corpus(settings.corpus_dir))

    return build_graph(store, settings=settings)


def render_answer(state: dict[str, object]) -> None:
    """Show the answer together with the evidence behind it."""
    answered = state["intent"] == ANSWERED_INTENT

    if answered:
        st.success(str(state["answer"]))
    else:
        st.warning(str(state["answer"]))

    columns = st.columns(2)
    columns[0].metric("Intent", str(state["intent"]))
    columns[1].metric("Confidence", f"{float(state['confidence']):.3f}")

    chunks = state.get("chunks") or []
    if chunks:
        st.caption("Sources, best match first")
        st.dataframe(
            [
                {"document": chunk.document_id, "relevance": round(chunk.relevance, 4)}
                for chunk in chunks  # type: ignore[union-attr]
            ],
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "Confidence is retrieval similarity on a 0-1 scale, meaning how closely the "
        "question resembled the matched policy. It is not a calibrated "
        "probability that the answer is correct."
    )


def main() -> None:
    st.set_page_config(page_title="Zepto Support Assistant", page_icon="🛒")

    st.title("Zepto Support Assistant")
    st.write(
        "Retrieval-augmented answers over a Zepto policy corpus. Ask about "
        "delivery, returns, membership, tracking, cancellation, gift cards, or "
        "support — it quotes the matching policy, or declines when the corpus "
        "does not cover the question."
    )

    graph = load_assistant()

    if "query" not in st.session_state:
        st.session_state.query = ""

    st.caption("Try one:")
    for column, (label, example) in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        if column.button(label, help=example, width="stretch"):
            st.session_state.query = example

    query = st.text_input(
        "Your question",
        value=st.session_state.query,
        placeholder="How much is the delivery fee?",
    )

    if query.strip():
        state = graph.invoke({"query": query.strip()})  # type: ignore[attr-defined]
        render_answer(state)

    with st.expander("How this works, and how well"):
        st.markdown(
            """
Questions are embedded with `all-MiniLM-L6-v2` (ONNX, via ChromaDB's bundled
runtime — no PyTorch) and matched against eight policy documents by cosine
similarity. Whether a question is in scope is decided by that similarity rather
than by keyword matching, which is what lets it answer questions phrased the way
people actually phrase them.

Answers are assembled deterministically from the retrieved text. There is no
language model in this path, so it cannot invent a policy that does not exist.

Measured against a labelled 77-case set, split into cases used for tuning
(`dev`) and cases held back from every tuning decision (`test`):

| Metric | dev (34) | test (43, held out) |
| --- | --- | --- |
| Retrieval hit rate @1 | 86.2% | 88.9% |
| Retrieval hit rate @3 | 96.6% | 100% |
| Mean reciprocal rank | 0.914 | 0.940 |
| Out-of-scope declined | 100% | 85.7% |

The held-out column is the honest one. Retrieval generalises; the scope decision
does not, quite — *"Convert 500 dollars to rupees"* slips past the relevance
floor because currency wording sits close to the delivery-fee document. It is
left standing rather than tuned away, since retuning against the split that
caught it is what the split exists to prevent.

[Source on GitHub](https://github.com/karanamsai1976-dotcom/zepto-platform)
"""
        )


if __name__ == "__main__":
    main()
