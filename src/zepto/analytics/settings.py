"""Configuration for the analytics and modelling pipeline.

Environment variables are prefixed ZEPTO_ANALYTICS_, e.g.
ZEPTO_ANALYTICS_TEST_SIZE=0.25.

The column lists are configuration rather than constants buried in code
because they encode modelling decisions that a reviewer needs to see and a
future maintainer may need to revisit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalyticsSettings(BaseSettings):
    """Settings for loading data, building features, and training models."""

    model_config = SettingsConfigDict(
        env_prefix="ZEPTO_ANALYTICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Data ---
    dataset_path: Path = Field(
        default=Path("data/samples/titanic.csv"),
        description="Versioned local copy of the dataset. Loading from disk rather "
        "than fetching at runtime keeps training reproducible and offline.",
    )
    target_column: str = Field(default="survived")

    # --- Feature policy ---
    leakage_columns: tuple[str, ...] = Field(
        default=("alive",),
        description="Columns that encode the target and must never reach the feature "
        "matrix. 'alive' is 'survived' as yes/no text: leaving it in yields a model "
        "that scores near-perfectly and has learned nothing.",
    )
    redundant_columns: tuple[str, ...] = Field(
        default=("class", "who", "adult_male", "embark_town", "alone", "deck"),
        description="Dropped as duplicates or derivations of columns already present: "
        "'class' duplicates pclass, 'who'/'adult_male' derive from sex and age, "
        "'embark_town' duplicates embarked, 'alone' derives from sibsp and parch, "
        "and 'deck' is missing for roughly three quarters of rows.",
    )

    # --- Leakage detection ---
    max_single_feature_accuracy: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="If any single feature predicts the target at or above this "
        "accuracy, it is treated as leakage. Catches leaking columns that are not "
        "on the known list.",
    )
    leakage_cardinality_limit: int = Field(
        default=20,
        ge=2,
        description="Columns with more distinct values than this are checked by "
        "correlation instead of group purity, since a near-unique column trivially "
        "predicts anything.",
    )
    leakage_cardinality_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Same guard expressed relative to sample size. A column with "
        "three distinct values across four rows is as degenerate as one with a "
        "thousand across two thousand, and an absolute limit alone misses it.",
    )

    # --- Splitting and training ---
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(
        default=42,
        description="Fixed so that runs are reproducible and comparable.",
    )
    cv_folds: int = Field(default=5, ge=2)

    # --- Artifacts ---
    model_dir: Path = Field(default=Path("models"))


@lru_cache(maxsize=1)
def get_analytics_settings() -> AnalyticsSettings:
    """Return the process-wide analytics settings, parsed from the environment once."""
    return AnalyticsSettings()
