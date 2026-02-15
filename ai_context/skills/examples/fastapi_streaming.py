"""
FastAPI SSE Streaming Example

This example demonstrates:
- Server-Sent Events for LLM streaming
- Error handling
- OpenTelemetry instrumentation
"""

import asyncio

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel

app = FastAPI(title="LLM Streaming API")

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)


class QueryRequest(BaseModel):
    prompt: str
    model: str = "gpt-4"
    max_tokens: int = 500


async def stream_llm_response(prompt: str, model: str):
    """Stream LLM response using SSE."""
    with tracer.start_as_current_span("llm.stream") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_length", len(prompt))

        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            async for chunk in response:
                if chunk.choices[0].delta.get("content"):
                    content = chunk.choices[0].delta.content
                    # SSE format: data: {content}\n\n
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for better UX

            # Send completion signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            span.record_exception(e)
            yield f"data: [ERROR] {str(e)}\n\n"


@app.post("/stream")
async def stream_completion(request: QueryRequest):
    """Stream LLM completion via SSE."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    return StreamingResponse(
        stream_llm_response(request.prompt, request.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
