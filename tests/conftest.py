"""Shared test fixtures.

Reference conftest showing:
- Mock fixtures for ports
- Sample data fixtures
"""

import os
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.config.settings import get_settings

# Force testing environment and disable JWT auth for integration tests
os.environ["ENVIRONMENT"] = "testing"
os.environ["JWT__ENABLED"] = "false"
get_settings.cache_clear()


from src.domain.entities.document import Document  # noqa: E402
from src.domain.ports.llm_port import LLMPort  # noqa: E402
from src.domain.ports.vector_store_port import VectorStorePort  # noqa: E402


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(content="Python is a programming language", metadata={"source": "wiki"}),
        Document(content="FastAPI is a web framework for Python", metadata={"source": "docs"}),
        Document(content="RAG improves LLM accuracy", metadata={"source": "paper"}),
    ]


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock(spec=LLMPort)


@pytest.fixture
def mock_vector_store(sample_documents: list[Document]) -> AsyncMock:
    store = AsyncMock(spec=VectorStorePort)
    store.search.return_value = sample_documents
    return store
