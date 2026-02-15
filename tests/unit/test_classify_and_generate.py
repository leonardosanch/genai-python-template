"""Tests for ClassifyAndGenerateUseCase."""

from unittest.mock import AsyncMock, PropertyMock

import pytest

from src.application.use_cases.classify_and_generate import (
    ClassifyAndGenerateUseCase,
    HybridResponse,
)
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.ml_model_port import MLModelPort


@pytest.fixture
def mock_classifier() -> AsyncMock:
    classifier = AsyncMock(spec=MLModelPort)
    classifier.predict.return_value = "technical_support"
    type(classifier).metadata = PropertyMock(return_value={"name": "intent-clf", "version": "1.0"})
    return classifier


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock(spec=LLMPort)
    llm.generate.return_value = "Please restart the service."
    return llm


@pytest.mark.asyncio
async def test_classifier_called_before_llm(
    mock_classifier: AsyncMock, mock_llm: AsyncMock
) -> None:
    uc = ClassifyAndGenerateUseCase(classifier=mock_classifier, llm=mock_llm)
    await uc.execute({"feature_a": 1.0}, "my app crashed")
    mock_classifier.predict.assert_awaited_once_with({"feature_a": 1.0})
    mock_llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_intent_routes_to_correct_prompt(
    mock_classifier: AsyncMock, mock_llm: AsyncMock
) -> None:
    uc = ClassifyAndGenerateUseCase(classifier=mock_classifier, llm=mock_llm)
    await uc.execute({"feature_a": 1.0}, "my app crashed")
    call_kwargs = mock_llm.generate.call_args[1]
    assert "troubleshooting" in call_kwargs["system_instruction"].lower()


@pytest.mark.asyncio
async def test_returns_hybrid_response(mock_classifier: AsyncMock, mock_llm: AsyncMock) -> None:
    uc = ClassifyAndGenerateUseCase(classifier=mock_classifier, llm=mock_llm)
    result = await uc.execute({"feature_a": 1.0}, "my app crashed")
    assert isinstance(result, HybridResponse)
    assert result.intent == "technical_support"
    assert result.response == "Please restart the service."
    assert result.model_metadata["name"] == "intent-clf"


@pytest.mark.asyncio
async def test_unknown_intent_uses_default_prompt(
    mock_classifier: AsyncMock, mock_llm: AsyncMock
) -> None:
    mock_classifier.predict.return_value = "unknown_intent"
    uc = ClassifyAndGenerateUseCase(classifier=mock_classifier, llm=mock_llm)
    await uc.execute({"feature_a": 1.0}, "hello")
    call_kwargs = mock_llm.generate.call_args[1]
    assert "helpful assistant" in call_kwargs["system_instruction"].lower()
