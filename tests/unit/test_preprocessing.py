"""Tests for preprocessing construction and the train-only fitting guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from zepto.analytics.preprocessing import (
    build_preprocessor,
    build_preprocessor_for,
    infer_feature_types,
)


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0, np.nan],
            "fare": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sex": ["male", "female", "male", "female", "male"],
            "embarked": ["S", "C", "S", "Q", "S"],
        }
    )


def test_feature_types_are_inferred_from_dtypes(features: pd.DataFrame) -> None:
    numeric, categorical = infer_feature_types(features)

    assert numeric == ["age", "fare"]
    assert categorical == ["sex", "embarked"]


def test_preprocessor_transforms_to_a_numeric_matrix(features: pd.DataFrame) -> None:
    preprocessor = build_preprocessor_for(features)

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_missing_values_are_imputed(features: pd.DataFrame) -> None:
    """The nan in age must not survive into the model input."""
    preprocessor = build_preprocessor_for(features)

    transformed = np.asarray(preprocessor.fit_transform(features), dtype=float)

    assert not np.isnan(transformed).any()


def test_unseen_category_at_scoring_time_does_not_raise(features: pd.DataFrame) -> None:
    """A category absent from training would otherwise take the service down."""
    preprocessor = build_preprocessor_for(features)
    preprocessor.fit(features)

    unseen = pd.DataFrame({"age": [25.0], "fare": [9.0], "sex": ["male"], "embarked": ["X"]})

    transformed = preprocessor.transform(unseen)

    assert transformed.shape[0] == 1


def test_imputer_learns_training_statistics_only() -> None:
    """The guarantee that matters: fitting inside a Pipeline uses only the rows
    the estimator trains on.

    The training median and the full-dataset median are deliberately different,
    so a preprocessor that had seen the test rows would learn the wrong value.
    """
    train = pd.DataFrame({"age": [10.0, 10.0, 10.0, np.nan], "sex": ["male"] * 4})
    test = pd.DataFrame({"age": [100.0, 200.0, 300.0], "sex": ["male"] * 3})
    combined_median = pd.concat([train, test])["age"].median()

    preprocessor = build_preprocessor(["age"], ["sex"])
    preprocessor.fit(train)

    learned = preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_[0]

    assert learned == 10.0
    assert learned != combined_median


def test_scaler_learns_training_statistics_only() -> None:
    train = pd.DataFrame({"fare": [1.0, 2.0, 3.0], "sex": ["male"] * 3})
    test = pd.DataFrame({"fare": [1000.0, 2000.0], "sex": ["male"] * 2})

    preprocessor = build_preprocessor(["fare"], ["sex"])
    preprocessor.fit(train)
    preprocessor.transform(test)

    learned_mean = preprocessor.named_transformers_["numeric"].named_steps["scale"].mean_[0]

    assert learned_mean == pytest.approx(2.0)


def test_transform_does_not_refit(features: pd.DataFrame) -> None:
    """Calling transform on new data must not update learned statistics."""
    preprocessor = build_preprocessor_for(features)
    preprocessor.fit(features)
    before = preprocessor.named_transformers_["numeric"].named_steps["scale"].mean_.copy()

    preprocessor.transform(
        pd.DataFrame({"age": [999.0], "fare": [999.0], "sex": ["male"], "embarked": ["S"]})
    )
    after = preprocessor.named_transformers_["numeric"].named_steps["scale"].mean_

    assert np.array_equal(before, after)


def test_preprocessor_composes_into_a_full_pipeline(features: pd.DataFrame) -> None:
    """End to end: a single fit call trains preprocessing and estimator together,
    which is what removes the opportunity to fit on test data by mistake."""
    target = pd.Series([0, 1, 0, 1, 0])
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor_for(features)),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    pipeline.fit(features, target)
    predictions = pipeline.predict(features)

    assert len(predictions) == len(features)
    assert set(predictions) <= {0, 1}
