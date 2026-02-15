# src/interfaces/api/routes/stream_routes.py
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.application.dtos.stream import StreamChatRequest
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.config import Settings, get_settings
from src.infrastructure.container import Container

router = APIRouter(prefix="/api/v1/stream", tags=["Streaming"])
logger = logging.getLogger(__name__)


async def stream_generator(llm: LLMPort, prompt: str) -> AsyncIterator[str]:
    """Yields SSE-formatted events from the LLM stream."""
    try:
        async for token in llm.stream(prompt):
            # Format as SSE data event
            # Ensure newlines in token are escaped or handled if necessary for JSON
            data = json.dumps({"token": token})
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        # In SSE, if we already sent headers, we can't change status code.
        # Best we can do is send an error event.
        error_data = json.dumps({"error": str(e)})
        yield f"event: error\ndata: {error_data}\n\n"


def get_container(settings: Settings = Depends(get_settings)) -> Container:
    return Container(settings)


@router.post("/chat")
async def stream_chat(
    request: StreamChatRequest, container: Container = Depends(get_container)
) -> StreamingResponse:
    """
    Stream LLM response using Server-Sent Events (SSE).

    Returns:
        StreamingResponse with media_type="text/event-stream"
    """
    # Assuming the container has an 'llm' property.
    # If not, we might need to resolve it differently or add it to Container.
    # Based on previous context, Container is a DI container.
    llm = container.llm_adapter

    return StreamingResponse(stream_generator(llm, request.message), media_type="text/event-stream")
