"""Tests for dataset loading and its schema contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from zepto.analytics.datasets import EXPECTED_COLUMNS, load_titanic, validate_schema
from zepto.analytics.settings import AnalyticsSettings
from zepto.core.errors import DatasetError

REPO_ROOT = Path(__file__).resolve().parents[2]
TITANIC_CSV = REPO_ROOT / "data" / "samples" / "titanic.csv"


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame({column: [0, 1] for column in EXPECTED_COLUMNS})


def test_committed_dataset_loads_and_satisfies_its_contract() -> None:
    frame = load_titanic(path=TITANIC_CSV, settings=AnalyticsSettings())

    assert len(frame) == 891
    assert set(EXPECTED_COLUMNS) <= set(frame.columns)


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc_info:
        load_titanic(path=tmp_path / "absent.csv", settings=AnalyticsSettings())

    assert "absent.csv" in exc_info.value.context["path"]


def test_empty_dataset_is_refused() -> None:
    with pytest.raises(DatasetError):
        validate_schema(pd.DataFrame())


def test_missing_columns_are_named_in_the_error() -> None:
    frame = _valid_frame().drop(columns=["fare", "embarked"])

    with pytest.raises(DatasetError) as exc_info:
        validate_schema(frame)

    assert set(exc_info.value.context["missing"]) == {"fare", "embarked"}


def test_absent_target_is_refused() -> None:
    frame = _valid_frame()

    with pytest.raises(DatasetError):
        validate_schema(frame, target_column="not_a_column")


def test_non_binary_target_is_refused() -> None:
    """A target with unexpected values means the wrong column or wrong file."""
    frame = _valid_frame()
    frame["survived"] = [0, 7]

    with pytest.raises(DatasetError) as exc_info:
        validate_schema(frame)

    assert "7" in exc_info.value.context["observed"]


def test_truncated_file_is_caught_by_the_contract(tmp_path: Path) -> None:
    """The failure mode this guards against: a partial file that reads fine but
    is missing columns, which v1 would only discover much later."""
    truncated = tmp_path / "partial.csv"
    truncated.write_text("survived,pclass\n0,3\n1,1\n", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_titanic(path=truncated, settings=AnalyticsSettings())
