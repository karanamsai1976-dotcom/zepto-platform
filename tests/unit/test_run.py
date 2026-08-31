"""Tests for the training entry point and model selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from zepto.analytics.registry import ModelRegistry
from zepto.analytics.run import MODEL_FACTORIES, main, select_best, train_all
from zepto.analytics.settings import AnalyticsSettings
from zepto.analytics.training import EvaluationMetrics, TrainingResult
from zepto.core.errors import AnalyticsError

REPO_ROOT = Path(__file__).resolve().parents[2]
TITANIC_CSV = REPO_ROOT / "data" / "samples" / "titanic.csv"


def _result(name: str, f1: float, roc_auc: float = 0.5) -> TrainingResult:
    return TrainingResult(
        model_name=name,
        metrics=EvaluationMetrics(accuracy=0.8, precision=0.8, recall=0.8, f1=f1, roc_auc=roc_auc),
        train_rows=100,
        test_rows=25,
        train_positive_rate=0.4,
        test_positive_rate=0.4,
    )


# --- selection ---


def test_best_model_is_chosen_by_the_configured_metric() -> None:
    results = [_result("a", f1=0.70), _result("b", f1=0.85), _result("c", f1=0.60)]

    assert select_best(results, "f1").model_name == "b"


def test_selection_metric_actually_changes_the_winner() -> None:
    """Different metrics can disagree; the configured one decides."""
    results = [
        _result("high_f1", f1=0.90, roc_auc=0.70),
        _result("high_auc", f1=0.60, roc_auc=0.95),
    ]

    assert select_best(results, "f1").model_name == "high_f1"
    assert select_best(results, "roc_auc").model_name == "high_auc"


def test_unknown_metric_lists_the_valid_ones() -> None:
    with pytest.raises(AnalyticsError) as exc_info:
        select_best([_result("a", f1=0.7)], "not_a_metric")

    assert "f1" in exc_info.value.context["available"]


def test_selecting_from_nothing_is_refused() -> None:
    with pytest.raises(AnalyticsError):
        select_best([], "f1")


# --- end to end ---


def test_train_all_trains_and_persists_every_model(tmp_path: Path) -> None:
    settings = AnalyticsSettings(
        dataset_path=TITANIC_CSV,
        model_dir=tmp_path / "models",
    )

    results = train_all(settings=settings)

    assert len(results) == len(MODEL_FACTORIES)
    assert {result.model_name for result in results} == set(MODEL_FACTORIES)

    registry = ModelRegistry(tmp_path / "models")
    for name in MODEL_FACTORIES:
        assert registry.list_versions(name), f"{name} was not persisted"


def test_persisted_models_are_loadable_and_predict(tmp_path: Path) -> None:
    """The artifact chain end to end: train, save, reload, predict."""
    settings = AnalyticsSettings(dataset_path=TITANIC_CSV, model_dir=tmp_path / "models")
    train_all(settings=settings)

    registry = ModelRegistry(tmp_path / "models")
    pipeline, metadata = registry.load("logistic_regression")

    import pandas as pd

    raw = pd.DataFrame(
        [
            {
                "pclass": 1,
                "sex": "female",
                "age": 29.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
                "embarked": "S",
            }
        ]
    )

    assert pipeline.predict(raw).shape == (1,)
    assert metadata.feature_names == (
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    )


def test_all_models_clear_a_sanity_floor(tmp_path: Path) -> None:
    settings = AnalyticsSettings(dataset_path=TITANIC_CSV, model_dir=tmp_path / "models")

    results = train_all(settings=settings)

    for result in results:
        assert result.metrics.roc_auc > 0.7, result


def test_console_entry_point_configures_logging_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        "zepto.analytics.run.configure_logging", lambda: events.append("configured")
    )
    monkeypatch.setattr("zepto.analytics.run.train_all", lambda: events.append("trained"))

    main()

    assert events == ["configured", "trained"]
