"""Unit test fixtures.

Provides mocked ports and sample data for isolated unit testing.
"""

from unittest.mock import AsyncMock

import pytest

from src.domain.entities.document import Document
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.vector_store_port import VectorStorePort


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock(spec=LLMPort)


@pytest.fixture
def mock_vector_store(sample_documents: list[Document]) -> AsyncMock:
    store = AsyncMock(spec=VectorStorePort)
    store.search.return_value = sample_documents
    return store


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(content="Python is a programming language", metadata={"source": "wiki"}),
        Document(content="FastAPI is a web framework for Python", metadata={"source": "docs"}),
        Document(content="RAG improves LLM accuracy", metadata={"source": "paper"}),
    ]
