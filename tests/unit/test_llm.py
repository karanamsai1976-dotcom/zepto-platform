"""Tests for the real-LLM path, in particular the retry loop.

v1 shipped a retry path that could never succeed -- its prompt requested prose
while its parser required JSON -- and no test ever ran it. These drive it
directly with a stub client, so the loop's behaviour is verified rather than
assumed.
"""

from __future__ import annotations

import pytest

from zepto.assistant.llm import (
    GroqClient,
    LLMAnswer,
    TransportError,
    generate_grounded_answer,
    parse_answer,
)
from zepto.assistant.prompts import build_answer_prompt, build_retry_prompt
from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings
from zepto.core.errors import GenerationError

CHUNKS = [
    RetrievedChunk(
        document_id="doc_01",
        chunk_id="doc_01#0",
        text="Standard delivery is free on orders over INR 149.",
        relevance=0.6,
    )
]


class ScriptedClient:
    """Returns queued replies in order, recording the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


class FailingClient:
    """Raises a transport error a fixed number of times, then succeeds."""

    def __init__(self, failures: int, reply: str) -> None:
        self.failures = failures
        self.reply = reply
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise TransportError("connection reset")
        return self.reply


# --- reply parsing ---


def test_valid_reply_is_accepted() -> None:
    answer = parse_answer('{"answer": "Delivery is free over INR 149."}')

    assert isinstance(answer, LLMAnswer)
    assert answer.answer == "Delivery is free over INR 149."


def test_non_json_reply_is_rejected_with_a_usable_message() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_answer("Delivery is free over INR 149.")


def test_json_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an object"):
        parse_answer('["an", "array"]')


def test_object_missing_the_answer_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="required shape"):
        parse_answer('{"reply": "wrong key"}')


def test_empty_answer_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_answer('{"answer": ""}')


# --- prompt construction ---


def test_answer_prompt_includes_question_and_context() -> None:
    prompt = build_answer_prompt("What is the delivery fee?", "[doc_01] Free over INR 149.")

    assert "What is the delivery fee?" in prompt
    assert "Free over INR 149." in prompt
    assert "JSON" in prompt


def test_retry_prompt_keeps_the_original_and_adds_the_correction() -> None:
    original = build_answer_prompt("Question?", "Context.")

    retry_prompt = build_retry_prompt(original, "reply was not valid JSON")

    assert original in retry_prompt
    assert "reply was not valid JSON" in retry_prompt


# --- the retry loop ---


def test_a_good_reply_needs_only_one_call() -> None:
    client = ScriptedClient(['{"answer": "Free over INR 149."}'])

    answer = generate_grounded_answer(
        "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
    )

    assert answer == "Free over INR 149."
    assert len(client.prompts) == 1


def test_a_malformed_reply_is_retried_and_can_succeed() -> None:
    """The case v1's implementation made impossible."""
    client = ScriptedClient(["not json at all", '{"answer": "Free over INR 149."}'])

    answer = generate_grounded_answer(
        "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
    )

    assert answer == "Free over INR 149."
    assert len(client.prompts) == 2


def test_the_retry_prompt_explains_what_was_wrong() -> None:
    """Reissuing the identical prompt would invite the identical bad reply."""
    client = ScriptedClient(["not json at all", '{"answer": "ok"}'])

    generate_grounded_answer(
        "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
    )

    assert "not valid JSON" in client.prompts[1]
    assert len(client.prompts[1]) > len(client.prompts[0])


def test_attempts_are_capped_and_failure_is_explicit() -> None:
    """After the limit, raise rather than return a fabrication."""
    client = ScriptedClient(["bad", "worse", "worst"])

    with pytest.raises(GenerationError) as exc_info:
        generate_grounded_answer(
            "What is the delivery fee?",
            CHUNKS,
            settings=AssistantSettings(llm_max_attempts=3),
            client=client,
        )

    assert exc_info.value.context["attempts"] == 3
    assert len(client.prompts) == 3


def test_attempt_limit_is_configurable() -> None:
    client = ScriptedClient(["bad", "bad", "bad", "bad", "bad"])

    with pytest.raises(GenerationError):
        generate_grounded_answer(
            "What is the delivery fee?",
            CHUNKS,
            settings=AssistantSettings(llm_max_attempts=2),
            client=client,
        )

    assert len(client.prompts) == 2


# --- transport failures ---


def test_transport_failures_are_retried_separately_from_bad_replies() -> None:
    """A timeout is worth retrying unchanged; a malformed reply is not."""
    client = FailingClient(failures=2, reply='{"answer": "Free over INR 149."}')

    answer = generate_grounded_answer(
        "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
    )

    assert answer == "Free over INR 149."
    assert client.calls == 3


def test_persistent_transport_failure_propagates() -> None:
    client = FailingClient(failures=99, reply="never reached")

    with pytest.raises(TransportError):
        generate_grounded_answer(
            "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
        )


# --- context assembly ---


def test_retrieved_documents_are_labelled_in_the_prompt() -> None:
    """Labelling lets the model attribute, and lets a reviewer trace the answer."""
    client = ScriptedClient(['{"answer": "ok"}'])

    generate_grounded_answer(
        "What is the delivery fee?", CHUNKS, settings=AssistantSettings(), client=client
    )

    assert "[doc_01]" in client.prompts[0]


# --- optional dependency handling ---


def test_missing_sdk_is_reported_with_a_usable_hint() -> None:
    """groq is deliberately not a declared dependency: the default path must
    work without it. When the real path is requested and the package is absent,
    the error must say what to do rather than surfacing a bare ImportError.
    """
    client = GroqClient(AssistantSettings())

    with pytest.raises(GenerationError) as exc_info:
        client.complete("any prompt")

    assert "groq" in str(exc_info.value)
    assert exc_info.value.context["hint"]
