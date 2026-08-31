"""Dataset loading with an enforced schema contract.

v1 called pd.read_csv and assumed the result was correct. If a column had been
renamed or the file truncated, the failure surfaced much later as a confusing
KeyError inside feature engineering, or worse, as a quietly wrong model. Here
the contract is checked at load time and violations name the specific problem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zepto.analytics.settings import AnalyticsSettings, get_analytics_settings
from zepto.core.errors import DatasetError
from zepto.core.logging import get_logger

logger = get_logger(__name__)

#: Columns the raw extract is expected to provide.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "class",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alive",
    "alone",
)


def validate_schema(frame: pd.DataFrame, target_column: str = "survived") -> None:
    """Raise DatasetError if the frame violates the expected contract."""
    if frame.empty:
        raise DatasetError("dataset is empty")

    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError("dataset is missing expected columns", missing=missing)

    if target_column not in frame.columns:
        raise DatasetError("target column is absent", target=target_column)

    observed = set(frame[target_column].dropna().unique())
    if not observed <= {0, 1}:
        raise DatasetError(
            "target column is not binary",
            target=target_column,
            observed=sorted(str(value) for value in observed),
        )


def load_titanic(
    path: Path | None = None,
    settings: AnalyticsSettings | None = None,
) -> pd.DataFrame:
    """Load the dataset from a versioned local file and validate its schema.

    Reading from disk rather than fetching over the network at runtime is what
    makes training reproducible: the same commit always trains on the same
    bytes, and training works offline.
    """
    resolved = settings or get_analytics_settings()
    dataset_path = path if path is not None else resolved.dataset_path

    if not dataset_path.exists():
        raise DatasetError("dataset file not found", path=str(dataset_path))

    frame = pd.read_csv(dataset_path)
    validate_schema(frame, target_column=resolved.target_column)

    logger.info(
        "dataset_loaded",
        path=str(dataset_path),
        rows=len(frame),
        columns=len(frame.columns),
    )
    return frame
