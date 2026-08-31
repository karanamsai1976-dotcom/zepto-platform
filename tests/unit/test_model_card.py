"""Tests for model card computation.

A card that overstates a model is worse than no card, so the parts that could
flatter it -- disaggregation and the baseline comparison -- are tested against
cases with known answers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from zepto.analytics.model_card import (
    MIN_GROUP_SIZE,
    BaselineComparison,
    GroupMetrics,
    compare_to_baseline,
    evaluate_by_group,
    generate_card,
    majority_class_baseline,
)
from zepto.analytics.registry import ModelMetadata


class ConstantModel:
    """Always predicts the same class, whatever it is shown."""

    def __init__(self, value: int) -> None:
        self.value = value

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series([self.value] * len(features), index=features.index)


class RuleModel:
    """Predicts survival for women, death for men -- the rule the real model
    approximates, used here so expected metrics can be computed by hand."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return (features["sex"] == "female").astype(int)


def _frame(per_cell: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Four sex-by-class cells with enough rows each to clear MIN_GROUP_SIZE,
    so intersections are actually reported."""
    features = pd.DataFrame(
        {
            "sex": ["female"] * (per_cell * 2) + ["male"] * (per_cell * 2),
            "pclass": ([1] * per_cell + [2] * per_cell) * 2,
        }
    )
    target = pd.Series([1] * (per_cell * 2) + [0] * (per_cell * 2))
    return features, target


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="logistic_regression",
        version="20260101T000000Z",
        created_at="2026-01-01T00:00:00+00:00",
        sklearn_version="1.9.0",
        python_version="3.12.10",
        platform="Windows-AMD64",
        feature_names=("sex", "pclass"),
        train_rows=712,
        test_rows=179,
        metrics={
            "accuracy": 0.8045,
            "precision": 0.7931,
            "recall": 0.6667,
            "f1": 0.7244,
            "roc_auc": 0.8437,
        },
    )


# --- degenerate group detection ---


def test_a_group_never_predicted_positive_is_flagged() -> None:
    group = GroupMetrics("sex=male", 100, 0.2, 0.8, 0.0, 0.0, 0.0)

    assert group.predicts_one_class_only


def test_a_group_with_real_discrimination_is_not_flagged() -> None:
    group = GroupMetrics("sex=female", 100, 0.7, 0.8, 0.8, 0.9, 0.85)

    assert not group.predicts_one_class_only


# --- disaggregation ---


def test_overall_and_per_attribute_groups_are_reported() -> None:
    features, target = _frame()

    groups = evaluate_by_group(RuleModel(), features, target, ("sex",))
    labels = [group.label for group in groups]

    assert labels[0] == "overall"
    assert "sex=female" in labels
    assert "sex=male" in labels


def test_intersections_are_reported_when_multiple_attributes_are_given() -> None:
    """Single-attribute slicing hides harm that appears only where attributes
    combine -- the reason the first version of this card reported no problem."""
    features, target = _frame()

    groups = evaluate_by_group(RuleModel(), features, target, ("sex", "pclass"))
    labels = [group.label for group in groups]

    assert any(" & " in label for label in labels)
    assert "sex=male & pclass=1" in labels


def test_a_small_intersection_is_omitted() -> None:
    """A cell can clear the threshold on each attribute alone and still be too
    small once they are crossed."""
    features = pd.DataFrame(
        {
            "sex": ["female"] * 10 + ["male"] * 10,
            "pclass": [1] * 10 + [1] * 8 + [2] * 2,
        }
    )
    target = pd.Series([1] * 10 + [0] * 10)

    labels = [
        group.label for group in evaluate_by_group(RuleModel(), features, target, ("sex", "pclass"))
    ]

    assert "sex=male & pclass=1" in labels
    assert "sex=male & pclass=2" not in labels


def test_small_groups_are_omitted() -> None:
    features = pd.DataFrame({"sex": ["female"] * 10 + ["male"] * (MIN_GROUP_SIZE - 1)})
    target = pd.Series([1] * 10 + [0] * (MIN_GROUP_SIZE - 1))

    groups = evaluate_by_group(ConstantModel(1), features, target, ("sex",))

    assert "sex=male" not in [group.label for group in groups]


def test_unknown_grouping_columns_are_ignored() -> None:
    features, target = _frame()

    groups = evaluate_by_group(RuleModel(), features, target, ("sex", "not_a_column"))

    assert [group.label for group in groups if "not_a_column" in group.label] == []


def test_group_metrics_are_computed_correctly() -> None:
    """A model that predicts survival for exactly the women, in data where
    exactly the women survived, is perfect within both groups."""
    features, target = _frame()

    groups = {
        group.label: group for group in evaluate_by_group(RuleModel(), features, target, ("sex",))
    }

    assert groups["sex=female"].recall == 1.0
    assert groups["sex=female"].precision == 1.0
    assert groups["sex=male"].accuracy == 1.0
    assert groups["sex=male"].predicts_one_class_only


# --- baseline ---


def test_baseline_predicts_each_training_subgroup_majority() -> None:
    features_train = pd.DataFrame({"sex": ["female"] * 4 + ["male"] * 4})
    target_train = pd.Series([1, 1, 1, 0, 0, 0, 0, 1])
    features_test = pd.DataFrame({"sex": ["female", "male"]})

    baseline = majority_class_baseline(features_train, target_train, features_test, ("sex",))

    assert list(baseline) == [1, 0]


def test_baseline_falls_back_for_an_unseen_subgroup() -> None:
    """A combination absent from training still needs a prediction."""
    features_train = pd.DataFrame({"sex": ["female"] * 4})
    target_train = pd.Series([1, 1, 1, 0])
    features_test = pd.DataFrame({"sex": ["other"]})

    baseline = majority_class_baseline(features_train, target_train, features_test, ("sex",))

    assert list(baseline) == [1]


def test_baseline_uses_the_intersection_when_given_two_columns() -> None:
    features_train = pd.DataFrame({"sex": ["f", "f", "m", "m"], "pclass": [1, 1, 1, 1]})
    target_train = pd.Series([1, 1, 0, 0])
    features_test = pd.DataFrame({"sex": ["f", "m"], "pclass": [1, 1]})

    baseline = majority_class_baseline(
        features_train, target_train, features_test, ("sex", "pclass")
    )

    assert list(baseline) == [1, 0]


def test_a_model_matching_the_baseline_shows_no_gain() -> None:
    """The disclosure that matters: a learned model that adds nothing."""
    features, target = _frame()

    comparison = compare_to_baseline(RuleModel(), features, target, features, target, ("sex",))

    assert comparison.agreement == 1.0
    assert comparison.accuracy_gain == pytest.approx(0.0)
    assert comparison.f1_gain == pytest.approx(0.0)


def test_a_model_worse_than_the_baseline_shows_a_negative_gain() -> None:
    features, target = _frame()

    comparison = compare_to_baseline(ConstantModel(0), features, target, features, target, ("sex",))

    assert comparison.accuracy_gain < 0
    assert comparison.agreement < 1.0


# --- rendering ---


def _card(groups: list[GroupMetrics], baseline: BaselineComparison) -> str:
    return generate_card(
        metadata=_metadata(),
        groups=groups,
        baseline=baseline,
        dataset_description="A dataset.",
        intended_use="Teaching.",
        out_of_scope_uses="Anything real.",
        ethical_considerations="Considerable.",
    )


def test_card_includes_the_required_sections() -> None:
    baseline = BaselineComparison(0.77, 0.69, 0.80, 0.72, 0.927, ("sex", "pclass"))
    card = _card([GroupMetrics("overall", 179, 0.385, 0.804, 0.793, 0.667, 0.724)], baseline)

    for heading in (
        "## Model details",
        "## Intended use",
        "## Out of scope",
        "## Training data",
        "## Disaggregated performance",
        "## Comparison against a trivial baseline",
        "## Limitations",
        "## Ethical considerations",
    ):
        assert heading in card


def test_card_reports_the_baseline_gain_and_agreement() -> None:
    baseline = BaselineComparison(0.7765, 0.6923, 0.8045, 0.7244, 0.927, ("sex", "pclass"))
    card = _card([GroupMetrics("overall", 179, 0.385, 0.804, 0.793, 0.667, 0.724)], baseline)

    assert "0.7765" in card
    assert "+0.0280" in card
    assert "92.7%" in card


def test_card_names_groups_that_receive_a_single_class_prediction() -> None:
    """The disclosure must appear in the document, not only in the data."""
    baseline = BaselineComparison(0.77, 0.69, 0.80, 0.72, 0.9, ("sex", "pclass"))
    groups = [
        GroupMetrics("overall", 179, 0.385, 0.804, 0.793, 0.667, 0.724),
        GroupMetrics("sex=male & pclass=3", 72, 0.139, 0.861, 0.0, 0.0, 0.0),
    ]

    card = _card(groups, baseline)

    assert "sex=male & pclass=3" in card
    assert "no discriminating ability" in card


def test_card_says_so_when_no_group_is_degenerate() -> None:
    baseline = BaselineComparison(0.77, 0.69, 0.80, 0.72, 0.9, ("sex",))
    card = _card([GroupMetrics("overall", 179, 0.385, 0.804, 0.793, 0.667, 0.724)], baseline)

    assert "No subgroup received a single-class prediction" in card


def test_card_records_provenance_from_the_artifact() -> None:
    baseline = BaselineComparison(0.77, 0.69, 0.80, 0.72, 0.9, ("sex",))
    card = _card([GroupMetrics("overall", 179, 0.385, 0.804, 0.793, 0.667, 0.724)], baseline)

    assert "20260101T000000Z" in card
    assert "scikit-learn 1.9.0" in card
    assert "3.12.10" in card
