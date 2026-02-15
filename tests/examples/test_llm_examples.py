"""
Tests for LLM examples.

These tests verify that the example code is functional and demonstrates
best practices correctly.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.examples.llm.basic_client import BasicLLMClient, LLMResponse
from src.examples.llm.structured_output import (
    EntityExtraction,
    Person,
    Sentiment,
    SentimentAnalysis,
    StructuredOutputClient,
)


class TestBasicLLMClient:
    """Tests for BasicLLMClient example."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch("src.examples.llm.basic_client.AsyncOpenAI") as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_openai_client):
        """Create test client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            return BasicLLMClient(provider="openai")

    async def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client.provider == "openai"
        assert client.model == "gpt-4o-mini"
        assert client.input_cost_per_1m > 0
        assert client.output_cost_per_1m > 0

    async def test_client_requires_api_key(self):
        """Test client raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                BasicLLMClient(provider="openai")

    async def test_generate_returns_response(self, client):
        """Test generate returns properly structured response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Test
        response = await client.generate(prompt="Test prompt")

        # Verify
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.tokens_used == 30
        assert response.cost_usd > 0

    async def test_stream_generate_yields_chunks(self, client):
        """Test streaming generation yields content chunks."""
        # Mock streaming response
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        client.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        # Test
        chunks = []
        async for chunk in client.stream_generate(prompt="Test"):
            chunks.append(chunk)

        # Verify
        assert chunks == ["Hello", " world", "!"]


class TestStructuredOutputClient:
    """Tests for StructuredOutputClient example."""

    @pytest.fixture
    def mock_instructor_client(self):
        """Mock Instructor-patched client."""
        with patch("src.examples.llm.structured_output.instructor.from_openai") as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_instructor_client):
        """Create test client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            return StructuredOutputClient()

    async def test_client_initialization(self, client):
        """Test client initializes correctly."""
        assert client.model == "gpt-4o-mini"

    async def test_analyze_sentiment_returns_structured_output(self, client):
        """Test sentiment analysis returns proper Pydantic model."""
        # Mock response
        mock_result = SentimentAnalysis(
            text="Great product!",
            sentiment=Sentiment.POSITIVE,
            confidence=0.95,
            reasoning="Positive language and exclamation",
        )

        client.client.chat.completions.create = AsyncMock(return_value=mock_result)

        # Test
        result = await client.analyze_sentiment("Great product!")

        # Verify
        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment == Sentiment.POSITIVE
        assert 0 <= result.confidence <= 1

    async def test_extract_entities_returns_structured_output(self, client):
        """Test entity extraction returns proper Pydantic model."""
        # Mock response
        mock_result = EntityExtraction(
            people=[Person(name="John Doe", age=30, occupation="Engineer")],
            organizations=["TechCorp"],
            locations=["San Francisco"],
            dates=["2024-01-15"],
        )

        client.client.chat.completions.create = AsyncMock(return_value=mock_result)

        # Test
        result = await client.extract_entities("John Doe works at TechCorp")

        # Verify
        assert isinstance(result, EntityExtraction)
        assert len(result.people) > 0
        assert isinstance(result.people[0], Person)

    async def test_pydantic_validation_works(self):
        """Test Pydantic models validate correctly."""
        # Valid sentiment
        sentiment = SentimentAnalysis(
            text="Test",
            sentiment=Sentiment.POSITIVE,
            confidence=0.8,
            reasoning="Test",
        )
        assert sentiment.confidence == 0.8

        # Invalid confidence (out of range)
        with pytest.raises(ValueError):
            SentimentAnalysis(
                text="Test",
                sentiment=Sentiment.POSITIVE,
                confidence=1.5,  # Invalid: > 1.0
                reasoning="Test",
            )


# Integration tests (require actual API keys)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestLLMExamplesIntegration:
    """Integration tests with real API calls."""

    async def test_basic_client_real_api(self):
        """Test basic client with real OpenAI API."""
        client = BasicLLMClient(provider="openai")

        response = await client.generate(
            prompt="Say 'test successful' and nothing else.",
            temperature=0,
            max_tokens=10,
        )

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.tokens_used > 0
        assert response.cost_usd > 0

    async def test_structured_output_real_api(self):
        """Test structured output with real OpenAI API."""
        client = StructuredOutputClient()

        result = await client.analyze_sentiment("This is amazing!")

        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment in [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL]
        assert 0 <= result.confidence <= 1
