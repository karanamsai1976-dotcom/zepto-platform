"""Tests for feature building and the anti-leakage guard.

The guard is tested two ways: that it catches leakage, and -- equally important
-- that it does not fire on the real dataset's legitimate features. A leakage
detector that cries wolf is one that gets disabled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from zepto.analytics.features import (
    assert_no_leakage,
    build_features,
    leakage_score,
    single_feature_accuracy,
)
from zepto.analytics.settings import AnalyticsSettings
from zepto.core.errors import LeakageError

REPO_ROOT = Path(__file__).resolve().parents[2]
TITANIC_CSV = REPO_ROOT / "data" / "samples" / "titanic.csv"


@pytest.fixture
def titanic() -> pd.DataFrame:
    return pd.read_csv(TITANIC_CSV)


def _settings(**overrides: object) -> AnalyticsSettings:
    return AnalyticsSettings(**overrides)  # type: ignore[arg-type]


# --- scoring primitives ---


def test_perfect_predictor_scores_one() -> None:
    target = pd.Series([0, 1, 0, 1, 1, 0])
    mirror = pd.Series(["no", "yes", "no", "yes", "yes", "no"])

    assert single_feature_accuracy(mirror, target) == 1.0


def test_uninformative_feature_scores_at_chance() -> None:
    target = pd.Series([0, 1, 0, 1])
    constant = pd.Series(["a", "a", "a", "a"])

    assert single_feature_accuracy(constant, target) == 0.5


def test_high_cardinality_numeric_is_scored_by_correlation() -> None:
    """A near-unique column separates any target perfectly by group purity,
    so numeric columns above the cardinality limit use correlation instead."""
    target = pd.Series(range(50)) % 2
    unrelated = pd.Series(range(50)) * 3.7

    score = leakage_score(unrelated, target, cardinality_limit=10)

    assert score < 0.5


def test_high_cardinality_non_numeric_is_not_flagged() -> None:
    """A free-text or identifier column cannot be correlated and must not be
    treated as leakage purely for being unique."""
    target = pd.Series([0, 1] * 25)
    identifiers = pd.Series([f"id-{index}" for index in range(50)])

    assert leakage_score(identifiers, target, cardinality_limit=10) == 0.0


def test_all_missing_feature_scores_zero() -> None:
    """An entirely empty column carries no information and must not be flagged."""
    target = pd.Series([0, 1, 0, 1])
    empty = pd.Series([None, None, None, None], dtype="object")

    assert single_feature_accuracy(empty, target) == 0.0
    assert leakage_score(empty, target, cardinality_limit=20) == 0.0


def test_cardinality_guard_is_relative_to_sample_size() -> None:
    """Three distinct values across four rows is as degenerate as a thousand
    across two thousand: group purity is meaningless when values barely repeat.

    An absolute cardinality limit alone misses this, which produced a false
    positive on a four-row frame during development.
    """
    tiny_target = pd.Series([0, 1, 0, 1])
    tiny_feature = pd.Series([3, 1, 3, 2])

    score = leakage_score(tiny_feature, tiny_target, cardinality_limit=20, cardinality_ratio=0.5)

    assert score < 0.99


def test_small_but_genuine_leak_is_still_caught() -> None:
    """The relative guard must not become an escape hatch for real leakage."""
    target = pd.Series([0, 1, 0, 1, 1, 0])
    mirror = pd.Series(["no", "yes", "no", "yes", "yes", "no"])

    score = leakage_score(mirror, target, cardinality_limit=20, cardinality_ratio=0.5)

    assert score == 1.0


# --- the guard ---


def test_known_leaking_column_is_refused_by_name(titanic: pd.DataFrame) -> None:
    features = titanic.drop(columns=["survived"])

    with pytest.raises(LeakageError) as exc_info:
        assert_no_leakage(features, titanic["survived"], settings=_settings())

    assert exc_info.value.context["columns"] == ["alive"]


def test_leakage_is_caught_behaviourally_even_when_not_named(
    titanic: pd.DataFrame,
) -> None:
    """The critical test: with 'alive' removed from the known list, the
    behavioural guard must still catch it. This is what protects against
    leakage nobody anticipated."""
    settings = _settings(leakage_columns=())
    features = titanic.drop(columns=["survived"])

    with pytest.raises(LeakageError) as exc_info:
        assert_no_leakage(features, titanic["survived"], settings=settings)

    assert "alive" in exc_info.value.context["columns"]
    assert exc_info.value.context["scores"]["alive"] == 1.0


def test_guard_does_not_fire_on_legitimate_features(titanic: pd.DataFrame) -> None:
    """A detector that false-alarms is a detector that gets switched off.

    Measured scores on this dataset: the strongest legitimate feature reaches
    0.789, well clear of the 0.99 threshold.
    """
    features, target = build_features(titanic, settings=_settings())

    assert_no_leakage(features, target, settings=_settings())
    assert "alive" not in features.columns


def test_every_legitimate_feature_scores_well_below_threshold(
    titanic: pd.DataFrame,
) -> None:
    features, target = build_features(titanic, settings=_settings())

    scores = {
        column: leakage_score(features[column], target, cardinality_limit=20)
        for column in features.columns
    }

    assert max(scores.values()) < 0.9, scores


# --- feature construction ---


def test_build_features_drops_target_leakage_and_redundant_columns(
    titanic: pd.DataFrame,
) -> None:
    features, target = build_features(titanic, settings=_settings())

    assert sorted(features.columns) == [
        "age",
        "embarked",
        "fare",
        "parch",
        "pclass",
        "sex",
        "sibsp",
    ]
    assert target.name == "survived"
    assert len(features) == len(titanic)


def test_build_features_is_tolerant_of_already_absent_columns() -> None:
    """Dropping should not fail because an expected column was already removed."""
    frame = pd.DataFrame({"survived": [0, 1, 0, 1], "pclass": [3, 1, 3, 2]})

    features, target = build_features(frame, settings=_settings())

    assert list(features.columns) == ["pclass"]
    assert list(target) == [0, 1, 0, 1]


def test_missing_target_is_refused() -> None:
    frame = pd.DataFrame({"pclass": [1, 2, 3]})

    with pytest.raises(LeakageError):
        build_features(frame, settings=_settings())


def test_leaking_frame_cannot_produce_features() -> None:
    """build_features must refuse rather than return a poisoned matrix."""
    frame = pd.DataFrame(
        {
            "survived": [0, 1, 0, 1, 1, 0],
            "outcome_text": ["no", "yes", "no", "yes", "yes", "no"],
            "pclass": [3, 1, 3, 2, 1, 3],
        }
    )

    with pytest.raises(LeakageError):
        build_features(frame, settings=_settings(leakage_columns=()))
