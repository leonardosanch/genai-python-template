"""
AWS Lambda RAG Function.

Demonstrates:
- Lambda handler for RAG queries
- DynamoDB for conversation state
- S3 for document storage
- Bedrock for LLM calls
- API Gateway integration

Deploy: sam build && sam deploy

Cost optimization:
- Use Lambda ARM64 (20% cheaper)
- Enable S3 Intelligent-Tiering
- Use DynamoDB on-demand pricing for low traffic
- Cache embeddings in S3 to avoid recomputation
"""

import json
import os
from datetime import datetime
from typing import Any

# AWS SDK
try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("⚠️  boto3 not installed. Install with: uv pip install boto3")


# Initialize AWS clients (lazy initialization in production)
def get_clients() -> dict[str, Any]:
    """Get AWS clients (singleton pattern)."""
    if not AWS_AVAILABLE:
        raise ImportError("boto3 required")

    return {
        "s3": boto3.client("s3"),
        "dynamodb": boto3.resource("dynamodb"),
        "bedrock": boto3.client("bedrock-runtime", region_name="us-east-1"),
    }


class ConversationStore:
    """DynamoDB conversation state management."""

    def __init__(self, table_name: str):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 required")

        clients = get_clients()
        self.table = clients["dynamodb"].Table(table_name)

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Get conversation history."""
        try:
            response = self.table.get_item(Key={"conversation_id": conversation_id})
            # Ensure list of dicts
            items = response.get("Item", {}).get("messages", [])
            return [dict(item) for item in items]
        except ClientError as e:
            print(f"Error getting history: {e}")
            return []

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Save message to conversation."""
        try:
            self.table.update_item(
                Key={"conversation_id": conversation_id},
                UpdateExpression=(
                    "SET messages = list_append(if_not_exists(messages, :empty_list), "
                    ":new_message), updated_at = :timestamp"
                ),
                ExpressionAttributeValues={
                    ":new_message": [{"role": role, "content": content}],
                    ":empty_list": [],
                    ":timestamp": datetime.utcnow().isoformat(),
                },
            )
        except ClientError as e:
            print(f"Error saving message: {e}")


class S3DocumentStore:
    """S3 document storage."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = get_clients()["s3"]

    def retrieve_documents(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieve relevant documents from S3.

        In production, use:
        - Pre-computed embeddings stored in S3
        - Vector index (OpenSearch, Pinecone)
        - Metadata filtering
        """
        try:
            # List objects with relevant prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="documents/",
                MaxKeys=top_k,
            )

            documents = []
            for obj in response.get("Contents", []):
                # Get object content
                file_obj = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj["Key"],
                )
                content = file_obj["Body"].read().decode("utf-8")
                documents.append(content[:500])  # Truncate for demo

            return documents

        except ClientError as e:
            print(f"Error retrieving documents: {e}")
            return []


def call_bedrock_claude(
    prompt: str,
    context: list[str],
    max_tokens: int = 1000,
) -> str:
    """
    Call Claude via AWS Bedrock.

    Cost optimization:
    - Use Claude Haiku for simple queries ($0.25/1M tokens)
    - Use Claude Sonnet for complex queries ($3/1M tokens)
    - Cache system prompts (50% cost reduction)
    """
    if not AWS_AVAILABLE:
        return "AWS SDK not available"

    bedrock = get_clients()["bedrock"]

    # Prepare context
    context_text = "\n\n".join(context) if context else "No context available."

    # Build messages for Claude 3
    messages = [
        {
            "role": "user",
            "content": f"""Use the following context to answer the question.

Context:
{context_text}

Question: {prompt}

Answer:""",
        }
    ]

    # Call Bedrock
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",  # Cheapest model
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
            ),
        )

        response_body = json.loads(response["body"].read())
        content = response_body["content"][0]["text"]
        return str(content)

    except ClientError as e:
        print(f"Error calling Bedrock: {e}")
        return "Error generating response"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda handler for RAG queries.

    Event structure (API Gateway):
    {
        "body": "{\"query\": \"What is Python?\", \"conversation_id\": \"abc123\"}",
        "requestContext": {...}
    }

    Environment variables:
    - DYNAMODB_TABLE: DynamoDB table name
    - S3_BUCKET: S3 bucket for documents
    """
    # Parse request
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query", "")
        conversation_id = body.get("conversation_id", "default")

        if not query:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing query parameter"}),
            }

        # Get environment variables
        table_name = os.environ.get("DYNAMODB_TABLE", "conversations")
        bucket_name = os.environ.get("S3_BUCKET", "rag-documents")

        # Initialize stores
        conversation_store = ConversationStore(table_name)
        document_store = S3DocumentStore(bucket_name)

        # Get conversation history
        # Get conversation history (for completeness, though not used in this simple handler)
        _ = conversation_store.get_history(conversation_id)

        # Retrieve relevant documents
        documents = document_store.retrieve_documents(query, top_k=3)

        # Generate response
        answer = call_bedrock_claude(query, documents, max_tokens=500)

        # Save to conversation history
        conversation_store.save_message(conversation_id, "user", query)
        conversation_store.save_message(conversation_id, "assistant", answer)

        # Return response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",  # CORS
            },
            "body": json.dumps(
                {
                    "answer": answer,
                    "conversation_id": conversation_id,
                    "sources": len(documents),
                }
            ),
        }

    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


# For local testing
if __name__ == "__main__":
    # Mock event
    test_event = {
        "body": json.dumps(
            {
                "query": "What is machine learning?",
                "conversation_id": "test-123",
            }
        )
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
