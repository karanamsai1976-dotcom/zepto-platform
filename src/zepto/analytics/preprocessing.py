"""Preprocessing pipeline construction.

The property that matters here is that preprocessing is fit on training data
only. v1 achieved this by remembering to call fit_transform on the training
split and transform on the test split -- correct, but a convention someone
could break without anything complaining.

Here it is structural. The preprocessor is a step inside a Pipeline, so fitting
the pipeline fits preprocessing on exactly the rows the estimator trains on,
and scoring transforms without refitting. There is no call site where the wrong
thing can be done by accident.

Feature types are inferred from dtypes rather than hardcoded, so adding a
column does not require editing this module -- but the inferred split is logged
on every build, so the decision stays visible.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from zepto.core.logging import get_logger

logger = get_logger(__name__)


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical groups by dtype."""
    numeric = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Build the preprocessing transformer.

    Numeric columns are median-imputed then standardised; median rather than
    mean because it is unaffected by the long fare tail. Categorical columns are
    imputed with the most frequent value then one-hot encoded.

    handle_unknown="ignore" matters in production: a category present at scoring
    time but absent during training would otherwise raise, taking down the
    service for an input it could have handled.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logger.info(
        "preprocessor_built",
        numeric=list(numeric_features),
        categorical=list(categorical_features),
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def build_preprocessor_for(features: pd.DataFrame) -> ColumnTransformer:
    """Infer feature types from a frame and build the matching preprocessor."""
    numeric_features, categorical_features = infer_feature_types(features)
    return build_preprocessor(numeric_features, categorical_features)
