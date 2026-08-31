"""Model training and evaluation.

v1 trained models in notebook cells and printed metrics. That works once, for
the person running it. It cannot be tested, re-run programmatically, compared
across commits, or called from a scheduled job.

Here training is a function that takes an estimator and returns structured
results. Metrics are data, so they can be asserted on in tests, logged as
queryable fields, and recorded alongside a model artifact.

The leakage guard runs again immediately before fitting. It already ran when
features were built, but this module can be called with any frame, and the
whole point of the guard is that it holds regardless of how the caller got
there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from zepto.analytics.features import assert_no_leakage
from zepto.analytics.preprocessing import build_preprocessor_for
from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings
from zepto.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Classification metrics on a held-out split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


@dataclass(frozen=True)
class TrainingResult:
    """Everything worth recording about one training run."""

    model_name: str
    metrics: EvaluationMetrics
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float


@dataclass(frozen=True)
class DataSplit:
    """A stratified train/test split, kept together so the four parts cannot
    be passed around in the wrong order."""

    features_train: pd.DataFrame
    features_test: pd.DataFrame
    target_train: pd.Series
    target_test: pd.Series


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    settings: AnalyticsSettings | None = None,
) -> DataSplit:
    """Split into train and test, preserving the target's class balance.

    Stratification matters when classes are unbalanced: a plain random split
    can hand the test set a materially different positive rate, which makes
    metrics reflect the split rather than the model.
    """
    resolved = settings or get_analytics_settings()

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=resolved.test_size,
        stratify=target,
        random_state=resolved.random_state,
    )

    logger.info(
        "data_split",
        train_rows=len(features_train),
        test_rows=len(features_test),
        train_positive_rate=round(float(target_train.mean()), 4),
        test_positive_rate=round(float(target_test.mean()), 4),
    )

    return DataSplit(features_train, features_test, target_train, target_test)


def evaluate(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> EvaluationMetrics:
    """Score a fitted pipeline on held-out data."""
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)[:, 1]

    return EvaluationMetrics(
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(target, probabilities)),
    )


def build_pipeline(estimator: BaseEstimator, features: pd.DataFrame) -> Pipeline:
    """Compose preprocessing and an estimator into one fittable object."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(features)),
            ("model", estimator),
        ]
    )


def train_and_evaluate(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    settings: AnalyticsSettings | None = None,
) -> tuple[Pipeline, TrainingResult]:
    """Split, fit on train only, and score on the held-out test split.

    Returns the fitted pipeline together with structured results, so callers
    can persist both without re-deriving either.
    """
    resolved = settings or get_analytics_settings()

    assert_no_leakage(features, target, settings=resolved)

    data = split_data(features, target, settings=resolved)
    pipeline = build_pipeline(estimator, data.features_train)
    pipeline.fit(data.features_train, data.target_train)

    metrics = evaluate(pipeline, data.features_test, data.target_test)
    result = TrainingResult(
        model_name=model_name,
        metrics=metrics,
        train_rows=len(data.features_train),
        test_rows=len(data.features_test),
        train_positive_rate=round(float(data.target_train.mean()), 4),
        test_positive_rate=round(float(data.target_test.mean()), 4),
    )

    logger.info("model_trained", model=model_name, **asdict(metrics))
    return pipeline, result
