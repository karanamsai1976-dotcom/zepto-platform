"""Model card generation: what a model does, and what it does badly.

A headline metric is the least informative thing you can say about a model. This
module computes the disclosures that actually matter and renders them as a
document a reviewer can read without running anything.

Two of those disclosures are computed rather than written, because writing them
by hand is how they end up flattering.

Disaggregated performance breaks the metrics down by subgroup. An aggregate
score can look healthy while the model is useless for a segment of the
population -- and on this dataset it is: overall accuracy is 0.80 while recall
for men is 0.125.

The trivial-baseline comparison asks how much the model adds over predicting the
majority class within each subgroup. If a learned model barely beats a lookup
table, that is the single most important thing to know about it, and it is
invisible from accuracy alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from zepto.analytics.registry import ModelMetadata
from zepto.core.logging import get_logger

logger = get_logger(__name__)

#: Minimum rows before a subgroup's metrics are reported. Below this the numbers
#: are too noisy to mean anything, and publishing them invites over-reading.
MIN_GROUP_SIZE = 5


@dataclass(frozen=True)
class GroupMetrics:
    """Performance within one slice of the evaluation data."""

    label: str
    size: int
    base_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    @property
    def predicts_one_class_only(self) -> bool:
        """True when the model never predicts the positive class for this group.

        Worth flagging explicitly: it means the model has no discriminating
        power here at all, however good its accuracy looks.
        """
        return self.precision == 0.0 and self.recall == 0.0


@dataclass(frozen=True)
class BaselineComparison:
    """The model measured against predicting each subgroup's majority class."""

    baseline_accuracy: float
    baseline_f1: float
    model_accuracy: float
    model_f1: float
    agreement: float
    grouped_by: tuple[str, ...]

    @property
    def accuracy_gain(self) -> float:
        return self.model_accuracy - self.baseline_accuracy

    @property
    def f1_gain(self) -> float:
        return self.model_f1 - self.baseline_f1


def _metrics_for(label: str, actual: pd.Series, predicted: pd.Series) -> GroupMetrics:
    return GroupMetrics(
        label=label,
        size=len(actual),
        base_rate=float(actual.mean()),
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1=float(f1_score(actual, predicted, zero_division=0)),
    )


def evaluate_by_group(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    group_columns: tuple[str, ...],
) -> list[GroupMetrics]:
    """Score the model overall, within each attribute, and at their intersections.

    Intersections are reported because single-attribute slicing hides harm that
    only appears where attributes combine. On this dataset every single-attribute
    slice looks unremarkable, while the model predicts death for every man in
    second and third class -- visible only when sex and class are crossed.

    Groups smaller than MIN_GROUP_SIZE are omitted rather than reported with wide
    error bars that readers will ignore.
    """
    predictions = pd.Series(pipeline.predict(features), index=features.index)
    present = [column for column in group_columns if column in features.columns]
    results = [_metrics_for("overall", target, predictions)]

    for column in present:
        for value in sorted(features[column].dropna().unique()):
            mask = features[column] == value
            if int(mask.sum()) < MIN_GROUP_SIZE:
                continue
            results.append(_metrics_for(f"{column}={value}", target[mask], predictions[mask]))

    if len(present) > 1:
        for combination, subset in features.groupby(present, observed=True):
            values = combination if isinstance(combination, tuple) else (combination,)
            if len(subset) < MIN_GROUP_SIZE:
                continue
            label = " & ".join(
                f"{column}={value}" for column, value in zip(present, values, strict=True)
            )
            results.append(_metrics_for(label, target[subset.index], predictions[subset.index]))

    return results


def majority_class_baseline(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> pd.Series:
    """Predict each row using the majority class of its training subgroup.

    Fit on the training split only, exactly like the model it is compared
    against -- a baseline that peeked at test data would not be a fair contest.
    """
    frame = features_train[list(group_columns)].copy()
    frame["__target"] = target_train.to_numpy()

    lookup = (
        frame.groupby(list(group_columns))["__target"]
        .mean()
        .apply(lambda mean: int(mean >= 0.5))
        .to_dict()
    )
    overall_majority = int(target_train.mean() >= 0.5)

    keys = features_test[list(group_columns)].itertuples(index=False, name=None)
    return pd.Series(
        [lookup.get(key if len(group_columns) > 1 else key[0], overall_majority) for key in keys],
        index=features_test.index,
    )


def compare_to_baseline(
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    group_columns: tuple[str, ...],
) -> BaselineComparison:
    """Measure how much the model adds over a subgroup lookup table."""
    baseline = majority_class_baseline(features_train, target_train, features_test, group_columns)
    predictions = pd.Series(pipeline.predict(features_test), index=features_test.index)

    return BaselineComparison(
        baseline_accuracy=float(accuracy_score(target_test, baseline)),
        baseline_f1=float(f1_score(target_test, baseline, zero_division=0)),
        model_accuracy=float(accuracy_score(target_test, predictions)),
        model_f1=float(f1_score(target_test, predictions, zero_division=0)),
        agreement=float((predictions == baseline).mean()),
        grouped_by=group_columns,
    )


def _group_table(groups: list[GroupMetrics]) -> str:
    header = (
        "| Group | n | Base rate | Accuracy | Precision | Recall | F1 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [
        f"| `{group.label}` | {group.size} | {group.base_rate:.3f} | {group.accuracy:.3f} | "
        f"{group.precision:.3f} | {group.recall:.3f} | {group.f1:.3f} |"
        for group in groups
    ]
    return "\n".join([header, *rows])


def _degenerate_groups_note(groups: list[GroupMetrics]) -> str:
    degenerate = [group for group in groups if group.predicts_one_class_only and group.size > 0]
    if not degenerate:
        return "No subgroup received a single-class prediction for every member."

    labels = ", ".join(f"`{group.label}`" for group in degenerate)
    return (
        f"The model predicts the negative class for every member of: {labels}. "
        "Accuracy within these groups therefore equals their base rate and reflects "
        "no discriminating ability whatsoever."
    )


def generate_card(
    metadata: ModelMetadata,
    groups: list[GroupMetrics],
    baseline: BaselineComparison,
    dataset_description: str,
    intended_use: str,
    out_of_scope_uses: str,
    ethical_considerations: str,
    additional_limitations: str = "",
) -> str:
    """Render a model card in Markdown.

    Quantitative sections are computed from the model. Qualitative sections are
    supplied by the caller, because judgements about intended use cannot be
    derived from metrics and should be attributable to a person.
    """
    grouped_by = ", ".join(f"`{column}`" for column in baseline.grouped_by)
    generated = datetime.now(UTC).strftime("%Y-%m-%d")

    return f"""# Model card: {metadata.model_name}

Version `{metadata.version}` · generated {generated}

## Model details

| Field | Value |
| --- | --- |
| Name | `{metadata.model_name}` |
| Version | `{metadata.version}` |
| Trained | {metadata.created_at} |
| Framework | scikit-learn {metadata.sklearn_version} |
| Python | {metadata.python_version} |
| Platform | {metadata.platform} |
| Features | {", ".join(f"`{name}`" for name in metadata.feature_names)} |
| Training rows | {metadata.train_rows} |
| Evaluation rows | {metadata.test_rows} |

The artifact is a complete pipeline: preprocessing and estimator together. It
accepts raw, unprocessed input.

## Intended use

{intended_use}

## Out of scope

{out_of_scope_uses}

## Training data

{dataset_description}

## Aggregate performance

| Metric | Value |
| --- | --- |
| Accuracy | {metadata.metrics.get("accuracy", float("nan")):.4f} |
| Precision | {metadata.metrics.get("precision", float("nan")):.4f} |
| Recall | {metadata.metrics.get("recall", float("nan")):.4f} |
| F1 | {metadata.metrics.get("f1", float("nan")):.4f} |
| ROC AUC | {metadata.metrics.get("roc_auc", float("nan")):.4f} |

## Disaggregated performance

An aggregate score can look healthy while the model is unusable for part of the
population. Groups smaller than {MIN_GROUP_SIZE} rows are omitted as too noisy to report.

{_group_table(groups)}

{_degenerate_groups_note(groups)}

## Comparison against a trivial baseline

The baseline predicts the majority class within each subgroup of {grouped_by},
learned from the training split only. It requires no model.

| | Accuracy | F1 |
| --- | ---: | ---: |
| Subgroup majority baseline | {baseline.baseline_accuracy:.4f} | {baseline.baseline_f1:.4f} |
| This model | {baseline.model_accuracy:.4f} | {baseline.model_f1:.4f} |
| Gain | {baseline.accuracy_gain:+.4f} | {baseline.f1_gain:+.4f} |

The model agrees with the baseline on **{baseline.agreement:.1%}** of evaluation rows.

Read the headline accuracy in that light: most of it comes from the base rates of
{grouped_by}, not from anything the model learned beyond them.

## Limitations

- The evaluation set is a single random split of {metadata.test_rows} rows. Differences
  smaller than a few points are not meaningful at this size, and no confidence
  intervals are reported.
- Performance is reported on historical data from one event. There is no held-out
  temporal validation, because the data has no time dimension to hold out.
- The model is calibrated for neither probability nor cost. Predicted probabilities
  should not be read as calibrated likelihoods.
{additional_limitations}

## Ethical considerations

{ethical_considerations}

## Reproducing this card

```bash
zepto-train      # trains and versions the model
zepto-card       # regenerates this document from the stored artifact
```
"""
