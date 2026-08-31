"""Prompt templates for the optional real-LLM answering path.

Kept as data in their own module rather than inline in the calling code, so the
wording can be reviewed and changed without touching control flow, and so the
correction appended on a retry sits next to the prompt it corrects.

The template asks for JSON and describes the exact shape expected. v1's template
asked for prose and its caller then parsed the reply as JSON, so every response
failed validation, burned all its retries, and returned an error -- a path that
was never exercised because it only ran in the non-default mode.
"""

from __future__ import annotations

ANSWER_PROMPT = """\
You are a customer support assistant for Zepto, a quick-commerce grocery \
delivery service.

Answer the customer's question using ONLY the policy extracts provided below. \
Do not use outside knowledge, and do not guess at details the extracts do not \
state. If the extracts do not contain the answer, say so plainly.

Policy extracts:
{context}

Customer question: {question}

Respond with a single JSON object and nothing else, in exactly this shape:
{{"answer": "<your answer in 1-3 plain sentences, under 80 words>"}}

Example:
{{"answer": "You have 24 hours from delivery to report a damaged or missing \
item, using the Report an Issue button on the order page."}}
"""

RETRY_CORRECTION = """\

Your previous reply could not be parsed: {error}

Reply with a single valid JSON object and nothing else. No markdown, no code \
fences, no commentary before or after. The shape must be exactly:
{{"answer": "<your answer>"}}
"""


def build_answer_prompt(question: str, context: str) -> str:
    """Fill the answering template."""
    return ANSWER_PROMPT.format(context=context, question=question)


def build_retry_prompt(previous_prompt: str, error: str) -> str:
    """Append a correction describing why the last reply was rejected.

    The original prompt is retained so the model keeps the context and question;
    only the instruction about output format is reinforced.
    """
    return previous_prompt + RETRY_CORRECTION.format(error=error)
