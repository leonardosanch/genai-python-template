"""
Azure Container Apps Agent.

Demonstrates:
- Container Apps deployment
- Cosmos DB for state
- Azure OpenAI Service
- Service Bus for events
- Managed Identity

Deploy: az containerapp up --name agent-app --resource-group rg-agents --source .

Cost optimization:
- Use consumption plan (pay per request)
- Scale to zero when idle
- Use Azure OpenAI gpt-3.5-turbo for cost efficiency
- Enable Cosmos DB autoscale
"""

import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure SDK
try:
    from azure.cosmos import CosmosClient  # type: ignore
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.servicebus import ServiceBusClient, ServiceBusMessage  # type: ignore
    from openai import AzureOpenAI

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️  Azure libraries not installed")
    print("Install with: uv pip install azure-cosmos azure-identity azure-servicebus openai")


app = FastAPI(title="Azure Container Apps Agent")


# Initialize Azure clients with Managed Identity
if AZURE_AVAILABLE:
    credential = DefaultAzureCredential()

    # Cosmos DB
    cosmos_client = CosmosClient(
        url=os.environ.get("COSMOS_ENDPOINT"),
        credential=credential,
    )
    database = cosmos_client.get_database_client(os.environ.get("COSMOS_DATABASE", "agents"))
    container = database.get_container_client("conversations")

    # Azure OpenAI
    azure_openai = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    )

    # Service Bus
    servicebus_client = ServiceBusClient(
        fully_qualified_namespace=os.environ.get("SERVICEBUS_NAMESPACE"),
        credential=credential,
    )


class QueryRequest(BaseModel):
    """Query request."""

    query: str
    conversation_id: str = "default"
    publish_event: bool = False


class QueryResponse(BaseModel):
    """Query response."""

    answer: str
    conversation_id: str
    event_published: bool = False


class CosmosConversationStore:
    """Cosmos DB conversation management."""

    def __init__(self) -> None:
        if not AZURE_AVAILABLE:
            raise ImportError("azure-cosmos required")

        self.container = container

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history."""
        try:
            items = list(
                self.container.query_items(
                    query="SELECT * FROM c WHERE c.conversation_id=@id",
                    parameters=[{"name": "@id", "value": conversation_id}],
                    enable_cross_partition_query=True,
                )
            )
            if items:
                msgs = items[0].get("messages", [])
                return [dict(m) for m in msgs]
            return []
        except Exception:
            return []

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Save message to Cosmos DB."""
        try:
            # Try to read existing
            try:
                item = self.container.read_item(
                    item=conversation_id,
                    partition_key=conversation_id,
                )
            except Exception:
                item = {
                    "id": conversation_id,
                    "conversation_id": conversation_id,
                    "messages": [],
                }

            # Append message
            item["messages"].append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            item["updated_at"] = datetime.utcnow().isoformat()

            # Upsert
            self.container.upsert_item(item)

        except Exception as e:
            print(f"Error saving message: {e}")


def call_azure_openai(prompt: str, context: list[str]) -> str:
    """
    Call Azure OpenAI Service.

    Cost optimization:
    - Use gpt-3.5-turbo for simple queries ($0.50/1M tokens)
    - Use gpt-4 only for complex reasoning ($30/1M tokens)
    - Enable prompt caching (50% cost reduction)
    """
    if not AZURE_AVAILABLE:
        return "Azure SDK not available"

    # Build messages
    context_text = "\n\n".join(context) if context else "No context available."

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": f"""Use the following context to answer the question.

Context:
{context_text}

Question: {prompt}

Answer:""",
        },
    ]

    # Call Azure OpenAI
    response = azure_openai.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-35-turbo"),
        messages=messages,  # type: ignore
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content or ""


def publish_to_service_bus(
    conversation_id: str,
    query: str,
    answer: str,
) -> None:
    """
    Publish event to Service Bus.

    Use for:
    - Analytics pipeline
    - Audit logging
    - Downstream processing
    """
    if not AZURE_AVAILABLE:
        return

    sender = servicebus_client.get_queue_sender(queue_name="agent-events")

    message = ServiceBusMessage(
        body={
            "conversation_id": conversation_id,
            "query": query,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    sender.send_messages(message)
    sender.close()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Query endpoint with optional event publishing."""
    try:
        store = CosmosConversationStore()

        # Get history
        # Get history (verify connection)
        _ = store.get_history(request.conversation_id)

        # Retrieve context (simplified)
        context = ["Python is a programming language.", "Azure is a cloud platform."]

        # Generate answer
        answer = call_azure_openai(request.query, context)

        # Save to Cosmos DB
        store.save_message(request.conversation_id, "user", request.query)
        store.save_message(request.conversation_id, "assistant", answer)

        # Publish event if requested
        event_published = False
        if request.publish_event:
            publish_to_service_bus(request.conversation_id, request.query, answer)
            event_published = True

        return QueryResponse(
            answer=answer,
            conversation_id=request.conversation_id,
            event_published=event_published,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check for Container Apps."""
    return {"status": "healthy", "service": "azure-container-apps-agent"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
