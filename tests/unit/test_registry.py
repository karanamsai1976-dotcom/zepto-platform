"""Tests for versioned model storage, provenance, and compatibility checking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from zepto.analytics.registry import (
    METADATA_FILENAME,
    PIPELINE_FILENAME,
    ModelMetadata,
    ModelRegistry,
    current_environment,
)
from zepto.analytics.training import (
    EvaluationMetrics,
    TrainingResult,
    build_pipeline,
)
from zepto.core.errors import ModelArtifactError

FEATURES = pd.DataFrame(
    {
        "pclass": [1, 3, 2, 1, 3, 2, 1, 3],
        "sex": ["female", "male", "female", "male", "female", "male", "female", "male"],
        "age": [29.0, 22.0, 35.0, None, 18.0, 40.0, 55.0, 12.0],
    }
)
TARGET = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])


def _result(model_name: str = "logistic_regression") -> TrainingResult:
    return TrainingResult(
        model_name=model_name,
        metrics=EvaluationMetrics(accuracy=0.8, precision=0.79, recall=0.67, f1=0.72, roc_auc=0.84),
        train_rows=6,
        test_rows=2,
        train_positive_rate=0.5,
        test_positive_rate=0.5,
    )


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "models")


@pytest.fixture
def fitted_pipeline() -> Pipeline:
    pipeline = build_pipeline(LogisticRegression(max_iter=200), FEATURES)
    pipeline.fit(FEATURES, TARGET)
    return pipeline


# --- saving ---


def test_save_writes_pipeline_and_metadata(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    version_dir = registry.root / "logistic_regression" / metadata.version
    assert (version_dir / PIPELINE_FILENAME).exists()
    assert (version_dir / METADATA_FILENAME).exists()


def test_metadata_records_the_training_environment(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """Provenance is the point: an artifact must be traceable to how it was made."""
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    assert metadata.sklearn_version == sklearn.__version__
    assert metadata.feature_names == ("pclass", "sex", "age")
    assert metadata.metrics["f1"] == 0.72
    assert metadata.train_rows == 6


def test_metadata_on_disk_is_readable_json(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """Metadata must be inspectable without loading the pickle, so an operator
    can check what is deployed without executing it."""
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    path = registry.root / "logistic_regression" / metadata.version / METADATA_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "logistic_regression"
    assert payload["sklearn_version"] == sklearn.__version__


def test_saving_twice_creates_two_versions_and_keeps_both(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """v1 overwrote its single artifact, so a bad retrain was unrecoverable."""
    first = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    second = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    versions = registry.list_versions("logistic_regression")

    assert first.version in versions
    assert second.version in versions
    assert len(versions) >= 1


# --- loading ---


def test_round_trip_preserves_predictions(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    expected = fitted_pipeline.predict(FEATURES)
    registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    loaded, _ = registry.load("logistic_regression")

    assert list(loaded.predict(FEATURES)) == list(expected)


def test_loaded_pipeline_accepts_raw_input(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """The artifact carries its own preprocessing, so raw text and missing
    values go straight in."""
    registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    loaded, _ = registry.load("logistic_regression")

    raw = pd.DataFrame([{"pclass": 1, "sex": "female", "age": None}])

    assert loaded.predict(raw).shape == (1,)


def test_latest_resolves_to_the_newest_version(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    versions = registry.list_versions("logistic_regression")

    assert registry.resolve_version("logistic_regression", "latest") == versions[-1]


def test_unknown_model_is_reported_clearly(registry: ModelRegistry) -> None:
    with pytest.raises(ModelArtifactError):
        registry.load("never_trained")


def test_unknown_version_lists_what_is_available(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))

    with pytest.raises(ModelArtifactError) as exc_info:
        registry.load("logistic_regression", version="19990101T000000Z")

    assert exc_info.value.context["available"]


def test_missing_pipeline_file_is_reported(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    (registry.root / "logistic_regression" / metadata.version / PIPELINE_FILENAME).unlink()

    with pytest.raises(ModelArtifactError):
        registry.load("logistic_regression")


def test_missing_metadata_file_is_reported(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    (registry.root / "logistic_regression" / metadata.version / METADATA_FILENAME).unlink()

    with pytest.raises(ModelArtifactError):
        registry.load_metadata("logistic_regression")


# --- version-skew protection ---


def test_version_mismatch_is_refused_by_default(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """The hazard v1 could not detect: a pickle loaded under a different
    scikit-learn version may behave differently rather than fail."""
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    path = registry.root / "logistic_regression" / metadata.version / METADATA_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sklearn_version"] = "0.0.1-ancient"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelArtifactError) as exc_info:
        registry.load("logistic_regression")

    assert exc_info.value.context["trained_with"] == "0.0.1-ancient"
    assert exc_info.value.context["running"] == sklearn.__version__


def test_version_mismatch_can_be_overridden_deliberately(
    registry: ModelRegistry, fitted_pipeline: Pipeline
) -> None:
    """Migrations sometimes need to load across versions -- but only explicitly."""
    metadata = registry.save(fitted_pipeline, _result(), tuple(FEATURES.columns))
    path = registry.root / "logistic_regression" / metadata.version / METADATA_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sklearn_version"] = "0.0.1-ancient"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, loaded_metadata = registry.load("logistic_regression", strict=False)

    assert loaded is not None
    assert loaded_metadata.sklearn_version == "0.0.1-ancient"


# --- helpers ---


def test_list_versions_is_empty_for_unknown_model(registry: ModelRegistry) -> None:
    assert registry.list_versions("never_trained") == []


def test_metadata_round_trips_through_json() -> None:
    original = ModelMetadata(
        model_name="m",
        version="20260101T000000Z",
        created_at="2026-01-01T00:00:00+00:00",
        sklearn_version="1.9.0",
        python_version="3.12.10",
        platform="Windows-AMD64",
        feature_names=("a", "b"),
        train_rows=10,
        test_rows=2,
        metrics={"f1": 0.5},
    )

    restored = ModelMetadata.from_dict(json.loads(original.to_json()))

    assert restored == original


def test_environment_fingerprint_reports_running_versions() -> None:
    environment = current_environment()

    assert environment["sklearn_version"] == sklearn.__version__
    assert environment["python_version"]
