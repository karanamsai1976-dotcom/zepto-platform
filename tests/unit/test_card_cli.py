"""Tests for model card generation end to end."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from zepto.analytics.card_cli import CARD_FILENAME, main, run
from zepto.analytics.datasets import load_titanic
from zepto.analytics.features import build_features
from zepto.analytics.registry import ModelRegistry
from zepto.analytics.settings import AnalyticsSettings
from zepto.analytics.training import train_and_evaluate
from zepto.core.errors import ModelArtifactError

REPO_ROOT = Path(__file__).resolve().parents[2]
TITANIC_CSV = REPO_ROOT / "data" / "samples" / "titanic.csv"


@pytest.fixture
def settings(tmp_path: Path) -> AnalyticsSettings:
    return AnalyticsSettings(dataset_path=TITANIC_CSV, model_dir=tmp_path / "models")


@pytest.fixture
def trained(settings: AnalyticsSettings) -> AnalyticsSettings:
    """Train and store one model so a card has something to describe."""
    frame = load_titanic(settings=settings)
    features, target = build_features(frame, settings=settings)
    pipeline, result = train_and_evaluate(
        LogisticRegression(max_iter=1000, random_state=42),
        features,
        target,
        model_name="logistic_regression",
        settings=settings,
    )
    ModelRegistry(settings.model_dir).save(pipeline, result, tuple(features.columns))
    return settings


def test_card_is_written_beside_the_artifact(trained: AnalyticsSettings) -> None:
    """Storing the card with the model keeps the description attached to the
    thing it describes, rather than drifting in a separate document."""
    destination = run(settings=trained)

    assert destination.name == CARD_FILENAME
    assert destination.exists()
    assert destination.parent.parent.name == "logistic_regression"


def test_card_reports_real_measured_numbers(trained: AnalyticsSettings) -> None:
    card = run(settings=trained).read_text(encoding="utf-8")

    assert "## Disaggregated performance" in card
    assert "sex=female" in card
    assert "sex=male" in card
    assert "Subgroup majority baseline" in card


def test_card_surfaces_intersectional_groups(trained: AnalyticsSettings) -> None:
    """The finding a single-attribute breakdown misses: the model predicts death
    for every man in second and third class."""
    card = run(settings=trained).read_text(encoding="utf-8")

    assert "sex=male & pclass=3" in card
    assert "no discriminating ability" in card


def test_card_states_the_model_is_not_for_deployment(trained: AnalyticsSettings) -> None:
    card = run(settings=trained).read_text(encoding="utf-8")

    assert "## Out of scope" in card
    assert "should not be" in card or "not intended" in card


def test_generating_a_card_for_an_untrained_model_is_refused(
    settings: AnalyticsSettings,
) -> None:
    with pytest.raises(ModelArtifactError):
        run(settings=settings, model_name="never_trained")


def test_console_entry_point_configures_logging_and_writes(
    trained: AnalyticsSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from zepto.analytics import card_cli

    events: list[str] = []

    def fake_run() -> Path:
        events.append("ran")
        return Path("card.md")

    monkeypatch.setattr(card_cli, "configure_logging", lambda: events.append("configured"))
    monkeypatch.setattr(card_cli, "run", fake_run)

    main()

    assert events == ["configured", "ran"]
    assert "card.md" in capsys.readouterr().out
