"""Versioned model storage with provenance and compatibility checking.

v1 wrote a single best_pipeline.joblib and overwrote it on every run. That
loses three things a production system needs.

Provenance: nothing recorded which data, code, or library versions produced the
artifact, so a model in production could not be traced back to how it was made.

Safety: joblib artifacts embed pickled scikit-learn objects. Loading one under a
different scikit-learn version can fail loudly, or worse, succeed and behave
subtly differently. v1 stored no version, so nothing could check.

History: overwriting means no rollback. If a retrain degrades the model, the
previous one is gone.

Each save here creates a timestamped version directory holding the pipeline and
a metadata document. Loading verifies the environment matches what trained it,
and refuses by default when it does not.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from zepto.analytics.training import TrainingResult
from zepto.core.errors import ModelArtifactError
from zepto.core.logging import get_logger

logger = get_logger(__name__)

PIPELINE_FILENAME = "pipeline.joblib"
METADATA_FILENAME = "metadata.json"
LATEST = "latest"


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to trace an artifact back to how it was produced."""

    model_name: str
    version: str
    created_at: str
    sklearn_version: str
    python_version: str
    platform: str
    feature_names: tuple[str, ...]
    train_rows: int
    test_rows: int
    metrics: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelMetadata:
        return cls(
            model_name=payload["model_name"],
            version=payload["version"],
            created_at=payload["created_at"],
            sklearn_version=payload["sklearn_version"],
            python_version=payload["python_version"],
            platform=payload["platform"],
            feature_names=tuple(payload["feature_names"]),
            train_rows=payload["train_rows"],
            test_rows=payload["test_rows"],
            metrics=payload["metrics"],
        )


def _new_version() -> str:
    """A UTC timestamp that sorts lexicographically, so 'latest' is just max().

    Chosen over a symlink because symlinks require elevated privileges on
    Windows by default.
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class ModelRegistry:
    """Stores and retrieves versioned model artifacts."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _model_dir(self, model_name: str) -> Path:
        return self._root / model_name

    def list_versions(self, model_name: str) -> list[str]:
        """Return known versions for a model, oldest first."""
        model_dir = self._model_dir(model_name)
        if not model_dir.exists():
            return []
        return sorted(entry.name for entry in model_dir.iterdir() if entry.is_dir())

    def save(
        self,
        pipeline: Pipeline,
        result: TrainingResult,
        feature_names: tuple[str, ...],
    ) -> ModelMetadata:
        """Persist a fitted pipeline as a new version, with its provenance.

        The whole pipeline is stored, not the bare estimator: an estimator
        without its preprocessing cannot consume raw input, which is the only
        kind of input production has.
        """
        version = _new_version()
        version_dir = self._model_dir(result.model_name) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        metadata = ModelMetadata(
            model_name=result.model_name,
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            sklearn_version=sklearn.__version__,
            python_version=platform.python_version(),
            platform=f"{platform.system()}-{platform.machine()}",
            feature_names=feature_names,
            train_rows=result.train_rows,
            test_rows=result.test_rows,
            metrics=asdict(result.metrics),
        )

        joblib.dump(pipeline, version_dir / PIPELINE_FILENAME)
        (version_dir / METADATA_FILENAME).write_text(metadata.to_json(), encoding="utf-8")

        logger.info(
            "model_saved",
            model=result.model_name,
            version=version,
            path=str(version_dir),
            **metadata.metrics,
        )
        return metadata

    def resolve_version(self, model_name: str, version: str = LATEST) -> str:
        """Turn 'latest' into a concrete version, or validate an explicit one."""
        versions = self.list_versions(model_name)
        if not versions:
            raise ModelArtifactError("no versions stored for model", model=model_name)

        if version == LATEST:
            return versions[-1]

        if version not in versions:
            raise ModelArtifactError(
                "requested version does not exist",
                model=model_name,
                requested=version,
                available=versions,
            )
        return version

    def load_metadata(self, model_name: str, version: str = LATEST) -> ModelMetadata:
        resolved = self.resolve_version(model_name, version)
        path = self._model_dir(model_name) / resolved / METADATA_FILENAME

        if not path.exists():
            raise ModelArtifactError(
                "metadata missing for stored version", model=model_name, version=resolved
            )

        return ModelMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load(
        self,
        model_name: str,
        version: str = LATEST,
        strict: bool = True,
    ) -> tuple[Pipeline, ModelMetadata]:
        """Load a stored pipeline, verifying the environment matches.

        A joblib artifact embeds pickled scikit-learn objects. Loading one under
        a different scikit-learn version can raise, or silently behave
        differently. With strict=True the mismatch is refused; with strict=False
        it is logged and allowed, which is occasionally what you want when
        deliberately migrating.
        """
        resolved = self.resolve_version(model_name, version)
        metadata = self.load_metadata(model_name, resolved)

        running = sklearn.__version__
        if metadata.sklearn_version != running:
            if strict:
                raise ModelArtifactError(
                    "model was trained with a different scikit-learn version",
                    model=model_name,
                    version=resolved,
                    trained_with=metadata.sklearn_version,
                    running=running,
                )
            logger.warning(
                "sklearn_version_mismatch",
                model=model_name,
                trained_with=metadata.sklearn_version,
                running=running,
            )

        pipeline_path = self._model_dir(model_name) / resolved / PIPELINE_FILENAME
        if not pipeline_path.exists():
            raise ModelArtifactError(
                "pipeline artifact missing", model=model_name, version=resolved
            )

        pipeline: Pipeline = joblib.load(pipeline_path)
        logger.info("model_loaded", model=model_name, version=resolved)
        return pipeline, metadata


def current_environment() -> dict[str, str]:
    """The environment fingerprint recorded with every artifact."""
    return {
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "python_executable": sys.executable,
    }
