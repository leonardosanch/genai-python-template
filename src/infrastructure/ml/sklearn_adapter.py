import asyncio
from pathlib import Path
from typing import Any

import joblib  # type: ignore

from src.domain.ports.ml_model_port import MLModelPort


class SklearnClassifierAdapter(MLModelPort[dict[str, Any], str]):
    """Adapter for Scikit-Learn classifiers."""

    def __init__(self, model_path: str):
        self._model_path = Path(model_path)
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found at {self._model_path}")
        self._model = joblib.load(self._model_path)

    async def predict(self, input_data: dict[str, Any]) -> str:
        """
        Make a prediction.
        Assumes input_data dictionary can be converted to what the model expects
        (e.g., list of values).
        """
        # Simulate async I/O wrapper since sklearn is sync
        return await asyncio.to_thread(self._predict_sync, input_data)

    async def predict_batch(self, inputs: list[dict[str, Any]]) -> list[str]:
        """Make batch predictions."""
        return await asyncio.to_thread(self._predict_batch_sync, inputs)

    def _predict_sync(self, input_data: dict[str, Any]) -> str:
        """Synchronous prediction."""
        # This implementation assumes the dict values are the features in order
        features = [list(input_data.values())]
        prediction = self._model.predict(features)
        return str(prediction[0])

    def _predict_batch_sync(self, inputs: list[dict[str, Any]]) -> list[str]:
        """Synchronous batch prediction."""
        if not inputs:
            return []
        features = [list(d.values()) for d in inputs]
        predictions = self._model.predict(features)
        return [str(p) for p in predictions]

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "type": "scikit-learn",
            "path": str(self._model_path),
            "model_class": self._model.__class__.__name__,
        }
