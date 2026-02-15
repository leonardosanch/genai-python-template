# src/domain/ports/ml_model_port.py
from abc import ABC, abstractmethod
from typing import TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class MLModelPort[TInput, TOutput](ABC):
    """Generic port for Machine Learning models."""

    @abstractmethod
    async def predict(self, input_data: TInput) -> TOutput:
        """Make a single prediction."""
        ...

    @abstractmethod
    async def predict_batch(self, inputs: list[TInput]) -> list[TOutput]:
        """Make batch predictions for efficiency."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, str]:
        """Return model metadata (name, version, etc.)."""
        ...
