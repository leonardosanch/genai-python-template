# tests/unit/test_ml_capabilities.py
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import joblib
    from sklearn.dummy import DummyClassifier

    from src.application.use_cases.classify_and_generate import ClassifyAndGenerateUseCase
    from src.infrastructure.ml.sklearn_adapter import SklearnClassifierAdapter

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.fixture
def mock_model_path():
    """Create a dummy sklearn model and save it to a temp file."""
    if not SKLEARN_AVAILABLE:
        yield Path("dummy")
        return

    model = DummyClassifier(strategy="constant", constant="technical_support")
    X = [[0, 0], [1, 1]]
    y = ["technical_support", "technical_support"]
    model.fit(X, y)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        joblib.dump(model, tmp.name)
        yield Path(tmp.name)

    Path(tmp.name).unlink(missing_ok=True)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
@pytest.mark.asyncio
async def test_sklearn_adapter(mock_model_path):
    """Test the sklearn adapter loads and predicts."""
    adapter = SklearnClassifierAdapter(str(mock_model_path))

    # Input format doesn't matter much for DummyClassifier, but adapter expects dict
    prediction = await adapter.predict({"f1": 0.5, "f2": 0.2})

    assert prediction == "technical_support"
    assert adapter.metadata["type"] == "scikit-learn"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
@pytest.mark.asyncio
async def test_classify_and_generate_use_case():
    """Test the hybrid flow."""
    # Mock ML Port
    mock_classifier = MagicMock()
    # Assume predict is async in the port interface
    mock_classifier.predict = AsyncMock(return_value="refund")
    mock_classifier.metadata = {"type": "mock"}

    # Mock LLM Port
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Sure, I can help with your refund.")

    use_case = ClassifyAndGenerateUseCase(mock_classifier, mock_llm)

    result = await use_case.execute(features={"amount": 100.0}, user_query="I want my money back")

    # Verify intent classification happened
    mock_classifier.predict.assert_called_once()

    # Verify LLM was called with the correct system prompt for "refund"
    call_args = mock_llm.generate.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "billing support agent" in kwargs["system_instruction"]

    # Verify result
    assert result.intent == "refund"
    assert result.response == "Sure, I can help with your refund."
