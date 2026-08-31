"""Training entry point: load, build features, train every model, persist.

v1's equivalent was a sequence of notebook cells. That works once, for whoever
runs it. This is a function, so it can be scheduled, tested, and re-run to
produce a comparable result rather than a screenshot.

Which model wins is decided by an explicit, configurable metric rather than by
whoever reads the comparison table -- so the choice is recorded rather than
remembered.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from zepto.analytics.datasets import load_titanic
from zepto.analytics.features import build_features
from zepto.analytics.registry import ModelRegistry
from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings
from zepto.analytics.training import TrainingResult, train_and_evaluate
from zepto.core.errors import AnalyticsError
from zepto.core.logging import configure_logging, get_logger

logger = get_logger(__name__)

#: Estimators are constructed per run so that the random seed is applied
#: consistently and no state leaks between models.
MODEL_FACTORIES: dict[str, Callable[[int], BaseEstimator]] = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
}


def select_best(results: list[TrainingResult], metric: str) -> TrainingResult:
    """Pick the winning run by a named metric.

    Made explicit because "which model is best" is a decision, not an
    observation, and it should be recorded rather than left to whoever reads
    the comparison table.
    """
    if not results:
        raise AnalyticsError("no training results to choose from")

    try:
        return max(results, key=lambda result: getattr(result.metrics, metric))
    except AttributeError as exc:
        raise AnalyticsError(
            "unknown selection metric",
            metric=metric,
            available=sorted(vars(results[0].metrics)),
        ) from exc


def train_all(settings: AnalyticsSettings | None = None) -> list[TrainingResult]:
    """Train every configured model on the same split and persist each one."""
    resolved = settings or get_analytics_settings()

    frame = load_titanic(settings=resolved)
    features, target = build_features(frame, settings=resolved)
    registry = ModelRegistry(resolved.model_dir)

    results: list[TrainingResult] = []
    for name, factory in MODEL_FACTORIES.items():
        estimator = factory(resolved.random_state)
        pipeline, result = train_and_evaluate(
            estimator, features, target, model_name=name, settings=resolved
        )
        registry.save(pipeline, result, tuple(features.columns))
        results.append(result)

    best = select_best(results, resolved.selection_metric)
    logger.info(
        "training_complete",
        models=len(results),
        best_model=best.model_name,
        selection_metric=resolved.selection_metric,
        best_score=round(getattr(best.metrics, resolved.selection_metric), 4),
    )
    return results


def main() -> None:
    """Console entry point: zepto-train."""
    configure_logging()
    train_all()
