---
name: Event-Driven & Distributed Systems
description: Architecture patterns for async processing, queues, and inter-service communication.
---

# Event-Driven & Distributed Systems

## Core Philosophy
Modern GenAI applications are inherently asynchronous. Long-running LLM tasks, RAG indexing pipelines, and multi-agent coordination require robust event-driven architectures.

## Key Technologies

### Message Brokers & Queues
- **Celery**: Robust distributed task queue. Standard for background jobs.
- **RabbitMQ**: Reliable message broker for service-to-service communication.
- **Kafka**: High-throughput event streaming. Ideal for data pipelines and analytics.
- **RQ (Redis Queue)**: Lightweight alternative to Celery for simpler use cases.

### Communication Protocols
- **gRPC**: High-performance, strongly-typed internal communication (Protobuf).
- **Webhooks**: Standard pattern for receiving events from external systems (GitHub, Stripe, Slack).

## Implementation Patterns

### 1. Background Tasks (Fire-and-Forget)
Offload heavy or slow operations (e.g., generating a long report, processing a file upload) to background workers.

```python
# tasks.py
@celery.task
def process_upload(file_id: str):
    # Long running logic
    ...
```

### 2. Outbox Pattern
Ensure data consistency when writing to DB and publishing events using the Outbox pattern (write event to DB table in same transaction, then publish).

### 3. Saga Pattern
Manage distributed transactions across services by defining a sequence of local transactions and compensating actions (rollbacks).

## Best Practices
1.  **Idempotency**: Consumers must handle duplicate messages gracefully.
2.  **Dead Letter Queues (DLQ)**: Capture failed messages for manual inspection.
3.  **Circuit Breakers**: Protect services from cascading failures during outages.
4.  **Observability**: Trace events across service boundaries using correlation IDs.

## External Resources
- [Celery User Guide](https://docs.celeryq.dev/en/stable/)
- [Microservices Patterns (Chris Richardson)](https://microservices.io/)
- [CloudEvents Standard](https://cloudevents.io/)
