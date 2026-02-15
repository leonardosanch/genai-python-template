# src/application/dtos/stream.py
from pydantic import BaseModel, Field


class StreamChatRequest(BaseModel):
    """Request model for streaming chat."""

    message: str = Field(..., description="The user message to send to the LLM")
    model: str | None = Field(None, description="Optional model override")
