"""The optional real-LLM answering path.

This is the only module that talks to a language model. The SDK import lives
inside the client factory, never at module scope, so the package imports and the
default path runs on a machine with no LLM library installed and no API key set.

Two failure modes are handled separately, because they need different responses.

A transport failure -- timeout, connection reset, rate limit -- is worth
retrying with backoff, since the same request may well succeed shortly. That is
handled by tenacity around the network call.

A malformed reply is not a transport problem: retrying the identical prompt
invites the identical bad output. So the retry appends a correction explaining
what was wrong, and only then asks again.

v1 had retry code for the second case that could never succeed: its prompt
requested prose while its parser required JSON, so every attempt failed by
construction. It went unnoticed because the path only ran in a non-default mode
that was never exercised. The tests here drive it directly.

Confidence is deliberately not requested from the model. A language model's
self-reported certainty is not calibrated against anything, and presenting it as
a probability would be inventing precision. Confidence comes from retrieval
relevance instead.
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from zepto.assistant.prompts import build_answer_prompt, build_retry_prompt
from zepto.assistant.retrieval import RetrievedChunk
from zepto.assistant.settings import AssistantSettings, get_assistant_settings
from zepto.core.errors import GenerationError
from zepto.core.logging import get_logger

logger = get_logger(__name__)


class LLMAnswer(BaseModel):
    """The contract a model reply must satisfy."""

    answer: str = Field(min_length=1, max_length=2000)


class LLMClient(Protocol):
    """The single operation this module needs from a language model."""

    def complete(self, prompt: str) -> str: ...


class TransportError(GenerationError):
    """A call failed in a way that may succeed on retry."""


class GroqClient:
    """Calls Groq's API. Constructed only when the real path is used."""

    def __init__(self, settings: AssistantSettings) -> None:
        self._settings = settings

    def complete(self, prompt: str) -> str:
        # Imported here, not at module scope, so this file stays importable and
        # the default path keeps working without the SDK installed.
        try:
            from groq import Groq
        except ImportError as exc:
            raise GenerationError(
                "the groq package is required for the real-LLM path",
                hint="pip install groq, or leave ZEPTO_ASSISTANT_MOCK_LLM unset",
            ) from exc

        return self._call_api(Groq, prompt)

    def _call_api(self, groq_class: type, prompt: str) -> str:  # pragma: no cover
        """Issue the actual request.

        Excluded from coverage deliberately and narrowly: reaching this code
        requires both the optional groq package and a live API key, neither of
        which exists in CI. Everything above it -- the missing-package path, the
        retry loop, reply validation, and prompt construction -- is covered by
        tests with a stub client.
        """
        import os

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise GenerationError(
                "GROQ_API_KEY is not set",
                hint="required only when mock_llm is disabled",
            )

        client = groq_class(api_key=api_key, timeout=self._settings.llm_timeout_seconds)
        try:
            response = client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._settings.llm_max_output_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise TransportError("language model call failed", cause=str(exc)) from exc

        content = response.choices[0].message.content
        return str(content or "")


@retry(
    retry=retry_if_exception_type(TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=4),
    reraise=True,
)
def _complete_with_backoff(client: LLMClient, prompt: str) -> str:
    """Call the model, retrying transport failures with exponential backoff."""
    return client.complete(prompt)


def parse_answer(raw: str) -> LLMAnswer:
    """Validate a raw reply against the expected contract.

    Raises ValueError with a description a model can act on, since the message
    is fed back to it on the retry.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("reply was valid JSON but not an object")

    try:
        return LLMAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"reply did not match the required shape: {exc.error_count()} problem(s)"
        ) from exc


def generate_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
    settings: AssistantSettings | None = None,
    client: LLMClient | None = None,
) -> str:
    """Answer a question from retrieved policy text, using a language model.

    On a malformed reply the prompt is reissued with a correction describing the
    problem, up to the configured attempt limit. If every attempt fails, a
    GenerationError is raised rather than a plausible-looking fabrication being
    returned.
    """
    resolved = settings or get_assistant_settings()
    llm = client if client is not None else GroqClient(resolved)

    context = "\n\n".join(f"[{chunk.document_id}] {chunk.text}" for chunk in chunks)
    prompt = build_answer_prompt(question, context)
    last_error = ""

    for attempt in range(1, resolved.llm_max_attempts + 1):
        raw = _complete_with_backoff(llm, prompt)

        try:
            return parse_answer(raw).answer
        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                "llm_reply_rejected",
                attempt=attempt,
                max_attempts=resolved.llm_max_attempts,
                reason=last_error,
            )
            prompt = build_retry_prompt(prompt, last_error)

    raise GenerationError(
        "language model did not return a valid answer",
        attempts=resolved.llm_max_attempts,
        last_error=last_error,
    )
