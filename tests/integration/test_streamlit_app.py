"""Tests for the Streamlit front end.

Run through Streamlit's own headless harness, which executes the real script
rather than importing pieces of it. That matters because the failure mode this
guards against is the script raising on startup -- a broken deployment shows a
stack trace to every visitor, and nothing else in the suite touches this file.

Marked integration because each run builds a real index, embedding model
included.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "streamlit_app.py"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

pytestmark = pytest.mark.integration


def run_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str | None = None) -> AppTest:
    """Execute the app against a throwaway index, optionally asking a question."""
    monkeypatch.setenv("ZEPTO_ASSISTANT_CORPUS_DIR", str(CORPUS_DIR))
    monkeypatch.setenv("ZEPTO_ASSISTANT_CHROMA_DIR", str(tmp_path / "chroma"))

    from zepto.assistant.settings import get_assistant_settings

    get_assistant_settings.cache_clear()

    # The cached loader is shared across sessions by design, so it has to be
    # cleared between tests or the second test reuses the first one's index.
    import streamlit as st

    st.cache_resource.clear()

    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()

    if query is not None:
        app.text_input[0].set_value(query).run()

    return app


def test_the_app_starts_without_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The one failure that would greet every visitor with a stack trace."""
    app = run_app(tmp_path, monkeypatch)

    assert not app.exception
    assert "Zepto Support Assistant" in app.title[0].value


def test_an_in_scope_question_is_answered_from_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = run_app(tmp_path, monkeypatch, query="How much is the delivery fee?")

    assert not app.exception
    assert "INR 25" in app.success[0].value
    assert app.metric[0].value == "policy_question"


def test_an_out_of_scope_question_is_declined_rather_than_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The behaviour worth demonstrating: it says no rather than answering from
    the nearest unrelated document."""
    app = run_app(tmp_path, monkeypatch, query="Who won the world cup?")

    assert not app.exception
    assert not app.success
    assert "could not find" in app.warning[0].value
    assert app.metric[0].value == "out_of_scope"


def test_sources_are_shown_for_an_answered_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hiding the evidence would misrepresent the system, since the retrieval is
    the part doing the work."""
    app = run_app(tmp_path, monkeypatch, query="Can I return an opened bottle of shampoo?")

    assert not app.exception
    assert app.dataframe


def test_the_ui_defines_no_logic_of_its_own() -> None:
    """Same rule the notebooks follow. A UI that computes anything is a second
    implementation free to disagree with the tested one, and the disagreement
    would surface in front of a user rather than in CI.

    `main` and the two render helpers are presentation; anything beyond that
    means a decision is being made here instead of in the package.
    """
    import ast

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)}

    assert defined == {"load_assistant", "render_answer", "main"}, defined


def test_the_ui_imports_its_behaviour_from_the_package() -> None:
    import ast

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert any(module.startswith("zepto.") for module in imported)


def test_an_example_button_fills_and_answers_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example buttons are the first thing a visitor clicks, and they take a
    different path from typing: a button writes to session state, and the text
    input reads its value back on the rerun."""
    app = run_app(tmp_path, monkeypatch)

    app.button[0].click().run()

    assert not app.exception
    assert app.text_input[0].value == "How much is the delivery fee?"
    assert "INR 25" in app.success[0].value


def test_every_example_button_is_labelled_distinctly_by_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deriving labels from the questions produced "How", "Can", "How", "Do",
    "Who" -- two identical and none naming a topic. A button has to say what
    pressing it does, and that is worth a test because the defect is invisible
    in the source and obvious on screen."""
    app = run_app(tmp_path, monkeypatch)

    labels = [button.label for button in app.button]

    assert labels == ["Delivery", "Returns", "Tracking", "Support", "Off-topic"]
    assert len(set(labels)) == len(labels)


def test_a_blank_question_renders_nothing_rather_than_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state every visitor sees first."""
    app = run_app(tmp_path, monkeypatch, query="   ")

    assert not app.exception
    assert not app.success
    assert not app.warning
