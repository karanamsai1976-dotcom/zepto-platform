"""Tests for training and evaluation.

The important properties are that the split preserves class balance, that the
leakage guard still fires at training time, and that results come back as data
rather than printed output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from zepto.analytics.features import build_features
from zepto.analytics.settings import AnalyticsSettings
from zepto.analytics.training import (
    DataSplit,
    EvaluationMetrics,
    TrainingResult,
    build_pipeline,
    evaluate,
    split_data,
    train_and_evaluate,
)
from zepto.core.errors import LeakageError

REPO_ROOT = Path(__file__).resolve().parents[2]
TITANIC_CSV = REPO_ROOT / "data" / "samples" / "titanic.csv"


@pytest.fixture
def titanic() -> pd.DataFrame:
    return pd.read_csv(TITANIC_CSV)


@pytest.fixture
def features_and_target(titanic: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_features(titanic, settings=AnalyticsSettings())


# --- splitting ---


def test_split_preserves_class_balance(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Without stratification the test split's positive rate can drift, making
    metrics reflect the split rather than the model."""
    features, target = features_and_target

    data = split_data(features, target, settings=AnalyticsSettings())

    overall = target.mean()
    assert data.target_train.mean() == pytest.approx(overall, abs=0.01)
    assert data.target_test.mean() == pytest.approx(overall, abs=0.01)


def test_split_sizes_match_configured_fraction(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = features_and_target

    data = split_data(features, target, settings=AnalyticsSettings(test_size=0.25))

    assert len(data.features_test) == pytest.approx(len(features) * 0.25, abs=1)
    assert len(data.features_train) + len(data.features_test) == len(features)


def test_split_is_reproducible(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """A fixed random_state is what makes two runs comparable."""
    features, target = features_and_target

    first = split_data(features, target, settings=AnalyticsSettings())
    second = split_data(features, target, settings=AnalyticsSettings())

    assert list(first.features_test.index) == list(second.features_test.index)


def test_split_returns_a_grouped_object(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Four loose values invite argument-order mistakes; one object does not."""
    features, target = features_and_target

    data = split_data(features, target, settings=AnalyticsSettings())

    assert isinstance(data, DataSplit)
    assert len(data.features_train) == len(data.target_train)
    assert len(data.features_test) == len(data.target_test)


# --- training ---


def test_training_returns_structured_results(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = features_and_target

    pipeline, result = train_and_evaluate(
        LogisticRegression(max_iter=1000),
        features,
        target,
        model_name="logistic_regression",
        settings=AnalyticsSettings(),
    )

    assert isinstance(result, TrainingResult)
    assert isinstance(result.metrics, EvaluationMetrics)
    assert result.model_name == "logistic_regression"
    assert result.train_rows + result.test_rows == len(features)
    assert pipeline.predict(features.head(3)).shape == (3,)


def test_metrics_are_in_valid_ranges(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = features_and_target

    _, result = train_and_evaluate(
        LogisticRegression(max_iter=1000),
        features,
        target,
        model_name="logistic_regression",
        settings=AnalyticsSettings(),
    )

    for name, value in vars(result.metrics).items():
        assert 0.0 <= value <= 1.0, f"{name}={value}"


def test_model_beats_chance_on_real_data(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """A sanity floor: if this fails, something upstream is broken, not just
    suboptimal."""
    features, target = features_and_target

    _, result = train_and_evaluate(
        LogisticRegression(max_iter=1000),
        features,
        target,
        model_name="logistic_regression",
        settings=AnalyticsSettings(),
    )

    assert result.metrics.roc_auc > 0.7
    assert result.metrics.accuracy > 0.7


def test_training_refuses_leaking_features(titanic: pd.DataFrame) -> None:
    """Defence in depth: the guard ran when features were built, and runs again
    here, because this function can be called with any frame."""
    leaking = titanic.drop(columns=["survived"])

    with pytest.raises(LeakageError):
        train_and_evaluate(
            LogisticRegression(max_iter=1000),
            leaking,
            titanic["survived"],
            model_name="leaky",
            settings=AnalyticsSettings(),
        )


def test_pipeline_bundles_preprocessing_with_the_estimator(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The artifact must carry its own preprocessing; a bare estimator cannot
    consume raw input."""
    features, _ = features_and_target

    pipeline = build_pipeline(LogisticRegression(), features)

    assert list(pipeline.named_steps) == ["preprocess", "model"]


def test_pipeline_predicts_from_raw_unprocessed_input(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Raw text categories and missing values go in; a prediction comes out."""
    features, target = features_and_target
    pipeline, _ = train_and_evaluate(
        LogisticRegression(max_iter=1000),
        features,
        target,
        model_name="logistic_regression",
        settings=AnalyticsSettings(),
    )

    raw = pd.DataFrame(
        [
            {
                "pclass": 1,
                "sex": "female",
                "age": None,
                "sibsp": 0,
                "parch": 0,
                "fare": 100.0,
                "embarked": "S",
            }
        ]
    )

    prediction = pipeline.predict(raw)

    assert prediction.shape == (1,)
    assert prediction[0] in {0, 1}


def test_evaluate_scores_a_fitted_pipeline(
    features_and_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = features_and_target
    data = split_data(features, target, settings=AnalyticsSettings())
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), data.features_train)
    pipeline.fit(data.features_train, data.target_train)

    metrics = evaluate(pipeline, data.features_test, data.target_test)

    assert isinstance(metrics, EvaluationMetrics)
    assert 0.0 <= metrics.f1 <= 1.0
