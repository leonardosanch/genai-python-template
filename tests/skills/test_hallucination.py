import pytest

from docs.skills.examples.hallucination_detection.detection import (
    MockAsyncOpenAI,
    calculate_semantic_entropy,
    chain_of_verification,
)


@pytest.mark.asyncio
async def test_semantic_entropy_calculation():
    """Test entropy calculation (mocked)."""
    client = MockAsyncOpenAI()
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("Skipping entropy test (numpy/sklearn not installed)")

    entropy = await calculate_semantic_entropy(client, "Test prompt", n_samples=3)
    assert isinstance(entropy, float)
    assert entropy >= 0.0


@pytest.mark.asyncio
async def test_chain_of_verification_run():
    """Test CoVe pipeline stub."""
    client = MockAsyncOpenAI()
    result = await chain_of_verification(client, "Capital of France?")
    assert result.confidence in ["high", "medium", "low"]
    assert "Paris" in result.original_answer
