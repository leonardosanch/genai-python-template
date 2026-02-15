"""
GCP Cloud Run Agent Service.

Demonstrates:
- Cloud Run container deployment
- Firestore for state
- Vertex AI for LLM
- Cloud Tasks for async processing
- Identity-aware proxy (IAP)

Deploy: gcloud run deploy agent-service --source . --region us-central1

Cost optimization:
- Use minimum instances = 0 (scale to zero)
- Set max instances based on expected load
- Use Vertex AI text-bison for cost efficiency
- Enable request-based autoscaling
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# GCP SDK
try:
    from google.cloud import aiplatform, firestore, tasks_v2  # type: ignore

    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("⚠️  google-cloud libraries not installed")
    print(
        "Install with: uv pip install google-cloud-firestore google-cloud-aiplatform "
        "google-cloud-tasks"
    )


app = FastAPI(title="GCP Cloud Run Agent")


# Initialize GCP clients
if GCP_AVAILABLE:
    db = firestore.Client()
    aiplatform.init(project=os.environ.get("GCP_PROJECT"), location="us-central1")
    tasks_client = tasks_v2.CloudTasksClient()


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(description="User query")
    conversation_id: str = Field(default="default", description="Conversation ID")
    async_processing: bool = Field(default=False, description="Process asynchronously")


class QueryResponse(BaseModel):
    """Query response model."""

    answer: str
    conversation_id: str
    processing_time_ms: float
    task_id: str | None = None


class FirestoreConversationStore:
    """Firestore conversation management."""

    def __init__(self) -> None:
        if not GCP_AVAILABLE:
            raise ImportError("google-cloud-firestore required")

        self.db = db
        self.collection = "conversations"

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history."""
        doc_ref = self.db.collection(self.collection).document(conversation_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict().get("messages", [])
            return [dict(d) for d in data]
        return []

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Save message to Firestore."""
        doc_ref = self.db.collection(self.collection).document(conversation_id)

        doc_ref.set(
            {
                "messages": firestore.ArrayUnion(
                    [{"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}]
                ),
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )


def call_vertex_ai(prompt: str, context: list[str]) -> str:
    """
    Call Vertex AI PaLM 2.

    Cost optimization:
    - Use text-bison for simple queries ($0.50/1M chars)
    - Use chat-bison for conversational ($0.50/1M chars)
    - Cache predictions for common queries
    """
    if not GCP_AVAILABLE:
        return "GCP SDK not available"

    from vertexai.language_models import TextGenerationModel  # type: ignore

    model = TextGenerationModel.from_pretrained("text-bison@002")

    # Build prompt
    context_text = "\n\n".join(context) if context else "No context available."
    full_prompt = f"""Use the following context to answer the question.

Context:
{context_text}

Question: {prompt}

Answer:"""

    # Generate
    response = model.predict(
        full_prompt,
        max_output_tokens=500,
        temperature=0.7,
    )

    return str(response.text)


def create_async_task(
    query: str,
    conversation_id: str,
    queue_name: str = "agent-tasks",
) -> str:
    """
    Create Cloud Task for async processing.

    Returns task ID.
    """
    if not GCP_AVAILABLE:
        return "task-mock"

    project = os.environ.get("GCP_PROJECT")
    location = os.environ.get("GCP_LOCATION", "us-central1")

    parent = tasks_client.queue_path(project, location, queue_name)

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{os.environ.get('SERVICE_URL')}/process",
            "headers": {"Content-Type": "application/json"},
            "body": f'{{"query": "{query}", "conversation_id": "{conversation_id}"}}'.encode(),
        }
    }

    response = tasks_client.create_task(request={"parent": parent, "task": task})
    return str(response.name)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Query endpoint with optional async processing.

    Sync: Returns answer immediately
    Async: Returns task ID, processes in background
    """
    import time

    start_time = time.time()

    try:
        if request.async_processing:
            # Create async task
            task_id = create_async_task(request.query, request.conversation_id)

            return QueryResponse(
                answer="Processing asynchronously...",
                conversation_id=request.conversation_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                task_id=task_id,
            )

        # Sync processing
        store = FirestoreConversationStore()

        # Get history
        # Get history (verify connection)
        _ = store.get_history(request.conversation_id)

        # Retrieve context (simplified - use vector search in production)
        context = ["Machine learning is a subset of AI.", "Python is a programming language."]

        # Generate answer
        answer = call_vertex_ai(request.query, context)

        # Save to Firestore
        store.save_message(request.conversation_id, "user", request.query)
        store.save_message(request.conversation_id, "assistant", answer)

        return QueryResponse(
            answer=answer,
            conversation_id=request.conversation_id,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "gcp-cloud-run-agent"}


@app.post("/process")
async def process_async(request: Request) -> dict[str, str]:
    """
    Async processing endpoint (called by Cloud Tasks).

    Protected by Cloud Tasks service account.
    """
    body = await request.json()
    query = body.get("query")
    conversation_id = body.get("conversation_id")

    # Process query
    store = FirestoreConversationStore()
    context: list[str] = []
    answer = call_vertex_ai(query, context)

    # Save result
    store.save_message(conversation_id, "user", query)
    store.save_message(conversation_id, "assistant", answer)

    return {"status": "processed", "conversation_id": conversation_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
