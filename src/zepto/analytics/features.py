"""Feature selection with an enforced anti-leakage guard.

Target leakage is the most dangerous defect in this project, because it is
silent: a leaking feature does not raise, it produces excellent metrics from a
model that has learned nothing. v1's defence was a comment and a drop() call in
a notebook cell -- correct, but unenforced and easy to undo.

Two guards run here instead.

Name-based: columns known to encode the target are refused outright.

Behaviour-based: any single feature that predicts the target at or above a
configured accuracy is refused, whether or not anyone anticipated it. This is
what catches leakage nobody thought to name. Low-cardinality columns are judged
by group purity (a depth-one decision tree), high-cardinality numeric columns
by correlation, because a near-unique column trivially separates any target and
would otherwise produce false alarms.
"""

from __future__ import annotations

import pandas as pd

from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings
from zepto.core.errors import LeakageError
from zepto.core.logging import get_logger

logger = get_logger(__name__)


def single_feature_accuracy(feature: pd.Series, target: pd.Series) -> float:
    """Accuracy of predicting the target from this feature alone.

    Equivalent to a depth-one decision tree: predict each group's majority
    class. A legitimate feature scores meaningfully above chance but well below
    one; a leaking feature scores at or near one.
    """
    frame = pd.DataFrame({"value": feature, "target": target}).dropna()
    if frame.empty:
        return 0.0

    majority = frame.groupby("value", observed=True)["target"].transform(
        lambda group: group.mode().iloc[0]
    )
    return float((majority == frame["target"]).mean())


def leakage_score(
    feature: pd.Series,
    target: pd.Series,
    cardinality_limit: int,
    cardinality_ratio: float = 0.5,
) -> float:
    """Score how completely a single feature determines the target, in [0, 1].

    Group purity is only meaningful when values repeat. A column approaching one
    distinct value per row separates any target perfectly while leaking nothing,
    so such columns are scored by correlation instead (or not at all, if they are
    not numeric).

    That degeneracy is measured two ways: against an absolute limit, and against
    the sample size. The ratio matters because three distinct values across four
    rows is exactly as uninformative as a thousand across two thousand, and an
    absolute limit alone would miss the first case.
    """
    rows = int(feature.notna().sum())
    if rows == 0:
        return 0.0

    distinct = int(feature.nunique(dropna=True))
    degenerate = distinct > cardinality_limit or (distinct / rows) > cardinality_ratio

    if degenerate:
        if not pd.api.types.is_numeric_dtype(feature):
            return 0.0
        correlation = feature.corr(target)
        return 0.0 if pd.isna(correlation) else float(abs(correlation))

    return single_feature_accuracy(feature, target)


def assert_no_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    settings: AnalyticsSettings | None = None,
) -> None:
    """Raise LeakageError if any feature encodes the target.

    Called automatically when features are built, and again before training, so
    that a leaking column cannot reach a model even if it is introduced later
    by a different code path.
    """
    resolved = settings or get_analytics_settings()

    named = [column for column in resolved.leakage_columns if column in features.columns]
    if named:
        raise LeakageError(
            "known leaking column present in feature matrix",
            columns=named,
        )

    offenders: dict[str, float] = {}
    for column in features.columns:
        score = leakage_score(
            features[column],
            target,
            cardinality_limit=resolved.leakage_cardinality_limit,
            cardinality_ratio=resolved.leakage_cardinality_ratio,
        )
        if score >= resolved.max_single_feature_accuracy:
            offenders[column] = round(score, 4)

    if offenders:
        raise LeakageError(
            "feature predicts the target almost perfectly and is treated as leakage",
            columns=sorted(offenders),
            scores=offenders,
            threshold=resolved.max_single_feature_accuracy,
        )


def build_features(
    frame: pd.DataFrame,
    settings: AnalyticsSettings | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw frame into a validated feature matrix and target vector.

    Drops the target, columns known to leak it, and columns that duplicate or
    derive from others already present, then asserts that nothing leaking
    survived.
    """
    resolved = settings or get_analytics_settings()

    if resolved.target_column not in frame.columns:
        raise LeakageError(
            "target column absent from frame",
            target=resolved.target_column,
        )

    target = frame[resolved.target_column]

    to_drop = [
        column
        for column in (
            resolved.target_column,
            *resolved.leakage_columns,
            *resolved.redundant_columns,
        )
        if column in frame.columns
    ]
    features = frame.drop(columns=to_drop)

    assert_no_leakage(features, target, settings=resolved)

    logger.info(
        "features_built",
        features=list(features.columns),
        dropped=to_drop,
        rows=len(features),
    )
    return features, target
