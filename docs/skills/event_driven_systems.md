---
name: Event-Driven & Distributed Systems
description: Architecture patterns for async processing, queues, and inter-service communication.
---

# Skill: Event-Driven & Distributed Systems

## Description

Modern GenAI applications are inherently asynchronous: long-running LLM tasks, RAG indexing pipelines, and multi-agent coordination all require robust event-driven architectures. This skill covers message brokers, task queues, inter-service communication protocols, and reliability patterns essential for distributed GenAI systems.

## Executive Summary

**Critical event-driven rules:**
- ALL consumers MUST be idempotent — use deduplication tables or idempotency keys (duplicate delivery is inevitable)
- Dead Letter Queues (DLQ) required for ALL queues — never silently drop failed messages after max retries
- Outbox pattern for DB + event consistency — write event to outbox table in same transaction, publish separately
- Bounded retries with exponential backoff — max 3-5 retries, base 2-5s, jitter enabled, DLQ as terminal destination
- Correlation IDs in ALL events — propagate across service boundaries for end-to-end tracing

**Read full skill when:** Implementing task queues, setting up message brokers, designing saga patterns, implementing circuit breakers, or orchestrating multi-service workflows.

---

## Versiones y Reliability

| Dependencia | Versión Mínima | Estabilidad |
|-------------|----------------|-------------|
| celery | >= 5.3.0 | ✅ Estable |
| aiokafka | >= 0.10.0 | ⚠️ API async |
| faststream | >= 0.4.0 | ✅ Recomendado |
| temporalio | >= 1.4.0 | ✅ Estable |

### Idempotencia Requerida

```python
async def process_event(event: dict, session: AsyncSession):
    # SIEMPRE verificar si el mensaje ya fue procesado
    stmt = select(ProcessedMessage).where(ProcessedMessage.id == event["id"])
    if (await session.execute(stmt)).scalar():
        return  # Ignorar duplicado
    
    # Procesar y marcar como hecho en la MISMA transacción
    await do_work(event)
    session.add(ProcessedMessage(id=event["id"]))
    await session.commit()
```

---

## Deep Dive

## Core Concepts

1. **Idempotent Consumers** — Every message consumer must produce the same result when processing the same message multiple times. Duplicate delivery is inevitable in distributed systems. Use unique message IDs and deduplication tables to enforce exactly-once semantics at the application level.

2. **Dead Letter Queues (DLQ)** — Messages that fail processing after bounded retries must be routed to a DLQ for inspection and manual reprocessing. Never silently drop failed messages. DLQs are the safety net that prevents data loss.

3. **Outbox Pattern** — When a service needs to update its database and publish an event, both operations must succeed or fail together. Write the event to an outbox table in the same database transaction, then a separate process publishes it to the broker. This guarantees consistency without distributed transactions.

4. **Circuit Breaker** — When a downstream service is failing, a circuit breaker stops sending requests after a threshold of failures, allowing the downstream to recover. After a cooldown period, it allows a probe request. This prevents cascading failures across the system.

5. **Correlation IDs** — Every event and request carries a unique correlation ID that propagates across service boundaries. This enables end-to-end tracing of a business operation through multiple services, queues, and workers.

6. **Saga Pattern** — Long-running business transactions that span multiple services are orchestrated as a sequence of local transactions. Each step has a compensating action (rollback). Sagas can be orchestrated (central coordinator) or choreographed (event-driven).

## External Resources

### :zap: Task Queues & Message Brokers

- [Celery User Guide](https://docs.celeryq.dev/en/stable/userguide/)
  *Best for:* Configuring distributed task queues with retry policies, rate limiting, and result backends.

- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
  *Best for:* Understanding exchange types, routing, and queue durability patterns.

- [aiokafka — Async Kafka Client](https://aiokafka.readthedocs.io/en/stable/)
  *Best for:* High-throughput async Kafka producers and consumers in Python.

- [Redis Streams](https://redis.io/docs/data-types/streams/)
  *Best for:* Lightweight event streaming when Kafka is overkill.

### :shield: Reliability Patterns

- [Microservices Patterns — Chris Richardson](https://microservices.io/patterns/)
  *Best for:* Saga, Outbox, CQRS, and other distributed system patterns with concrete examples.

- [Martin Fowler — Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
  *Best for:* Understanding event sourcing as an alternative to CRUD for audit-heavy domains.

- [Temporal.io Documentation](https://docs.temporal.io/)
  *Best for:* Durable workflow execution with built-in retry, timeout, and compensation support.

- [CloudEvents Specification](https://cloudevents.io/)
  *Best for:* Standardized event envelope format for interoperability across services.

### :wrench: Communication Protocols

- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
  *Best for:* High-performance internal service communication with Protobuf contracts.

- [FastStream — Async Messaging Framework](https://faststream.airt.ai/latest/)
  *Best for:* Unified async interface for Kafka, RabbitMQ, NATS, and Redis brokers.

### :book: Architecture References

- [Designing Data-Intensive Applications — Martin Kleppmann](https://dataintensive.net/)
  *Best for:* Deep understanding of distributed systems, replication, partitioning, and stream processing.

- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
  *Best for:* Canonical messaging patterns (routing, transformation, splitting).

## Decision Trees

### Decision Tree 1: Qué sistema de mensajería usar

```
¿Cuál es tu escenario principal?
├── Background tasks (enviar email, procesar doc, llamar LLM)
│   └── Celery + Redis broker
│       ├── Simple, bien documentado, retry nativo
│       └── No para event streaming ni pub/sub complejo
├── Comunicación entre microservicios
│   └── ¿Necesitas ordering y replay de eventos?
│       ├── SÍ → Kafka (log-based, partitioned, replay)
│       │   └── Alto throughput, consumer groups, retention
│       └── NO → RabbitMQ (routing flexible, exchanges, DLQ nativo)
├── Event streaming ligero (no justifica Kafka)
│   └── Redis Streams
│       ├── Si ya tienes Redis, zero infra adicional
│       └── Consumer groups, acknowledgment, pero sin replay largo
├── Workflows durables (multi-step, compensación, timeouts)
│   └── Temporal.io
│       ├── Built-in retry, timeout, compensation
│       └── Ideal para orquestación de agentes LLM
└── Comunicación interna alta performance
    └── gRPC (protobuf, streaming bidireccional)
```

| Sistema | Throughput | Ordering | Replay | Complejidad | Caso ideal |
|---------|-----------|----------|--------|------------|------------|
| Celery + Redis | Medio | No | No | Baja | Background tasks |
| RabbitMQ | Medio-Alto | Por queue | No | Media | Service messaging |
| Kafka | Muy alto | Por partition | Sí | Alta | Event streaming |
| Redis Streams | Medio | Por stream | Limitado | Baja | Lightweight events |
| Temporal | N/A | Sí | Sí | Media | Durable workflows |

### Decision Tree 2: Saga Orchestration vs Choreography

```
¿Cuántos servicios participan en la transacción?
├── 2-3 servicios
│   └── Choreography (event-driven, sin coordinador central)
│       ├── Cada servicio escucha eventos y reacciona
│       ├── Compensación por eventos inversos
│       └── Simple, pero difícil de debuggear con muchos pasos
├── 4+ servicios
│   └── Orchestration (coordinador central)
│       ├── Un servicio orquesta la secuencia
│       ├── Más fácil de entender y debuggear
│       └── El orquestador es un single point of failure
└── Workflow complejo con timeouts, human-in-the-loop, branching
    └── Temporal.io / durable workflows
        └── No implementar saga manualmente — usar Temporal
```

### Decision Tree 3: Sync vs Async entre servicios

```
¿El caller necesita la respuesta inmediatamente?
├── SÍ (< 500ms) → Sync (HTTP REST o gRPC)
│   └── ¿Alto throughput o binary data?
│       ├── SÍ → gRPC (protobuf, streaming, multiplexing)
│       └── NO → HTTP REST (simple, universal)
├── NO (puede esperar) → Async (message queue)
│   └── Publicar evento → consumer procesa → notifica resultado
│       └── Retornar 202 Accepted con task ID para polling
└── Mixto (respuesta parcial inmediata + procesamiento async)
    └── Sync para acknowledgment + Async para el trabajo real
        └── POST /tasks → 202 + task_id → worker procesa → GET /tasks/{id}
```

---

## Instructions for the Agent

1. **Consultar decision tree antes de elegir sistema de mensajería.** Celery para
   background tasks, RabbitMQ para service communication, Kafka para event streaming,
   Temporal para workflows durables. No usar Kafka para simple background tasks.

2. **All consumers must be idempotent** — Deduplication table o idempotency keys.
   Nunca asumir entrega exactly-once.

3. **Dead Letter Queues para todas las queues** — Alertas en DLQ depth.

4. **Correlation IDs en todas las fronteras** — Generar en entry point, propagar
   por headers HTTP, metadata de mensajes, y log context.

5. **Circuit breakers obligatorios para calls externos** — LLM providers, APIs
   externas, databases remotas. `tenacity` o `pybreaker`.

6. **Async > Sync entre servicios** — Message queues para inter-service. Sync solo
   cuando necesitas respuesta inmediata.

7. **Bounded retries con exponential backoff** — Max 3-5 retries, base 2-5s, jitter.
   DLQ como destino terminal.

8. **Events self-contained** — El evento lleva toda la data necesaria. No callbacks
   al producer.

9. **Outbox pattern para DB + event** — Nunca publicar evento y escribir a DB como
   operaciones separadas.

10. **Para pipelines GenAI (RAG indexing, agent coordination), usar cadena de eventos.**
    Cada step es un consumer idempotente. Si falla, retry desde ese step.

## Code Examples

### GenAI-Specific: LLM Task Queue with Rate Limiting

```python
"""Queue para llamadas LLM con rate limiting y prioridad."""
from celery import Celery
from kombu import Queue

app = Celery("llm_worker", broker="redis://localhost:6379/0")

# Queues con prioridad
app.conf.task_queues = [
    Queue("llm_high", routing_key="llm.high"),
    Queue("llm_low", routing_key="llm.low"),
    Queue("llm_dlq", routing_key="llm.dlq"),
]

# Rate limiting: máximo 50 llamadas/minuto al LLM provider
app.conf.task_annotations = {
    "tasks.call_llm": {"rate_limit": "50/m"},
}


@app.task(
    bind=True,
    max_retries=3,
    retry_backoff=True,
    retry_backoff_max=120,
    acks_late=True,
)
def call_llm(self, prompt: str, model: str, correlation_id: str):
    """LLM call as async task with rate limiting and retry."""
    try:
        result = llm_client.generate(prompt=prompt, model=model)
        return {"output": result.content, "tokens": result.usage.total_tokens}
    except RateLimitError as exc:
        self.retry(exc=exc, countdown=60)
    except ProviderError as exc:
        self.retry(exc=exc)
```

### GenAI-Specific: RAG Indexing Pipeline as Event Chain

```python
"""Pipeline de indexación RAG como cadena de eventos idempotentes."""
from enum import Enum


class IndexingEvent(str, Enum):
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PARSED = "document.parsed"
    CHUNKS_CREATED = "chunks.created"
    EMBEDDINGS_GENERATED = "embeddings.generated"
    INDEX_UPDATED = "index.updated"
    INDEXING_FAILED = "indexing.failed"


# Consumer chain — cada step es idempotente:
# document.uploaded → parse_document → document.parsed
# document.parsed → create_chunks → chunks.created
# chunks.created → generate_embeddings → embeddings.generated
# embeddings.generated → update_index → index.updated
# Cualquier fallo → retry desde ese step → DLQ si max retries

@app.task(bind=True, max_retries=3, retry_backoff=True)
def parse_document(self, event: dict):
    """Step 1: Parse document into text."""
    doc = fetch_document(event["document_id"])
    text = parser.parse(doc.content, doc.content_type)
    publish_event(IndexingEvent.DOCUMENT_PARSED, {
        "document_id": event["document_id"],
        "text": text,
        "correlation_id": event["correlation_id"],
    })


@app.task(bind=True, max_retries=3, retry_backoff=True)
def create_chunks(self, event: dict):
    """Step 2: Split text into chunks."""
    chunks = chunker.split(event["text"])
    publish_event(IndexingEvent.CHUNKS_CREATED, {
        "document_id": event["document_id"],
        "chunks": [{"id": c.id, "text": c.text} for c in chunks],
        "correlation_id": event["correlation_id"],
    })


@app.task(bind=True, max_retries=3, retry_backoff=True)
def generate_embeddings(self, event: dict):
    """Step 3: Generate embeddings."""
    texts = [c["text"] for c in event["chunks"]]
    embeddings = embedding_model.embed(texts)
    publish_event(IndexingEvent.EMBEDDINGS_GENERATED, {
        "document_id": event["document_id"],
        "chunks_with_embeddings": [
            {**chunk, "embedding": emb}
            for chunk, emb in zip(event["chunks"], embeddings)
        ],
        "correlation_id": event["correlation_id"],
    })


@app.task(bind=True, max_retries=3, retry_backoff=True)
def update_index(self, event: dict):
    """Step 4: Upsert into vector store."""
    vector_store.upsert(event["chunks_with_embeddings"])
    publish_event(IndexingEvent.INDEX_UPDATED, {
        "document_id": event["document_id"],
        "chunk_count": len(event["chunks_with_embeddings"]),
        "correlation_id": event["correlation_id"],
    })
```

### GenAI-Specific: Agent Coordination Events

```python
"""Self-contained events para coordinación multi-agent."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentEvent(str, Enum):
    TASK_ASSIGNED = "agent.task_assigned"
    TASK_COMPLETED = "agent.task_completed"
    TASK_FAILED = "agent.task_failed"
    HANDOFF_REQUESTED = "agent.handoff_requested"
    HUMAN_REVIEW_NEEDED = "agent.human_review_needed"


@dataclass
class AgentTaskEvent:
    """Self-contained event — carries all context needed for processing."""
    event_type: AgentEvent
    agent_id: str
    task_id: str
    correlation_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    input_summary: str = ""
    output_summary: str = ""
    next_agent: str | None = None  # For handoff
    confidence: float = 0.0
    token_usage: int = 0
    cost_usd: float = 0.0
    payload: dict = field(default_factory=dict)
```

### Celery Task with Retry and Dead Letter Queue

```python
from celery import Celery
from celery.utils.log import get_task_logger

app = Celery("worker", broker="redis://localhost:6379/0")
logger = get_task_logger(__name__)

# Configure DLQ routing
app.conf.task_routes = {
    "tasks.process_document": {"queue": "documents"},
}
app.conf.task_reject_on_worker_lost = True
app.conf.task_acks_late = True


class MaxRetriesExceededError(Exception):
    """Raised when task exceeds max retry attempts."""


@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    acks_late=True,
)
def process_document(self, document_id: str, correlation_id: str) -> dict:
    """Process a document with bounded retries and DLQ on final failure."""
    logger.info(
        "Processing document",
        extra={"document_id": document_id, "correlation_id": correlation_id},
    )
    try:
        result = _do_processing(document_id)
        return {"status": "completed", "document_id": document_id}
    except TransientError as exc:
        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            # Route to DLQ by publishing to dead letter exchange
            _send_to_dlq(document_id, correlation_id, str(exc))
            raise
    except PermanentError as exc:
        # No retry — send directly to DLQ
        _send_to_dlq(document_id, correlation_id, str(exc))
        logger.error("Permanent failure", extra={"document_id": document_id, "error": str(exc)})
        raise
```

### Kafka Consumer with Idempotent Processing

```python
import asyncio
import json
from contextlib import asynccontextmanager

from aiokafka import AIOKafkaConsumer
from sqlalchemy.ext.asyncio import AsyncSession


class IdempotentKafkaConsumer:
    """Kafka consumer with deduplication and exactly-once processing semantics."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        session_factory,
    ) -> None:
        self._consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        self._session_factory = session_factory

    async def start(self) -> None:
        await self._consumer.start()
        try:
            async for message in self._consumer:
                await self._process_with_dedup(message)
        finally:
            await self._consumer.stop()

    async def _process_with_dedup(self, message) -> None:
        """Process message only if not already processed (idempotency)."""
        message_id = message.headers.get("message-id", message.offset)
        correlation_id = message.headers.get("correlation-id", "unknown")

        async with self._session_factory() as session:
            # Check if already processed
            if await self._is_processed(session, message_id):
                await self._consumer.commit()
                return

            try:
                # Process the event
                await self._handle_event(session, message.value, correlation_id)

                # Mark as processed in same transaction
                await self._mark_processed(session, message_id)
                await session.commit()

                # Commit offset only after successful DB commit
                await self._consumer.commit()
            except Exception:
                await session.rollback()
                raise

    async def _is_processed(self, session: AsyncSession, message_id: str) -> bool:
        result = await session.execute(
            text("SELECT 1 FROM processed_messages WHERE message_id = :id"),
            {"id": str(message_id)},
        )
        return result.scalar() is not None

    async def _mark_processed(self, session: AsyncSession, message_id: str) -> None:
        await session.execute(
            text("INSERT INTO processed_messages (message_id) VALUES (:id)"),
            {"id": str(message_id)},
        )

    async def _handle_event(self, session: AsyncSession, payload: dict, correlation_id: str) -> None:
        """Override in subclasses to implement event-specific logic."""
        raise NotImplementedError
```

### Outbox Pattern with SQLAlchemy

```python
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.data.models import Base


class OutboxMessage(Base):
    """Transactional outbox table for reliable event publishing."""

    __tablename__ = "outbox_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    aggregate_type = Column(String(255), nullable=False, index=True)
    aggregate_id = Column(String(255), nullable=False)
    event_type = Column(String(255), nullable=False)
    payload = Column(Text, nullable=False)
    correlation_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    published = Column(Boolean, default=False, index=True)
    retry_count = Column(Integer, default=0)


class OutboxPublisher:
    """Writes events to outbox table within the same DB transaction."""

    async def publish(
        self,
        session: AsyncSession,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict,
        correlation_id: str,
    ) -> str:
        message = OutboxMessage(
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type=event_type,
            payload=json.dumps(payload),
            correlation_id=correlation_id,
        )
        session.add(message)
        return message.id


class OutboxRelay:
    """Polls outbox table and forwards unpublished messages to the broker."""

    def __init__(self, session_factory, broker_client, batch_size: int = 100) -> None:
        self._session_factory = session_factory
        self._broker = broker_client
        self._batch_size = batch_size

    async def relay_pending(self) -> int:
        """Publish pending outbox messages. Returns count of published messages."""
        async with self._session_factory() as session:
            stmt = (
                select(OutboxMessage)
                .where(OutboxMessage.published == False)
                .order_by(OutboxMessage.created_at)
                .limit(self._batch_size)
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

            published_count = 0
            for msg in messages:
                try:
                    await self._broker.publish(
                        topic=msg.aggregate_type,
                        key=msg.aggregate_id,
                        value=msg.payload,
                        headers={"correlation-id": msg.correlation_id, "event-type": msg.event_type},
                    )
                    msg.published = True
                    published_count += 1
                except Exception:
                    msg.retry_count += 1

            await session.commit()
            return published_count
```

### Circuit Breaker Decorator with Tenacity

```python
import logging
from enum import Enum
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker that prevents cascading failures to downstream services."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(seconds=30),
        expected_exceptions: tuple = (Exception,),
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._expected_exceptions = expected_exceptions
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: datetime | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = datetime.now(timezone.utc) - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Recovery in {self._recovery_timeout.total_seconds()}s."
                )

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self._expected_exceptions as exc:
                self._on_failure()
                raise

        return wrapper

    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker OPENED after %d failures", self._failure_count)


class CircuitOpenError(Exception):
    """Raised when a call is attempted on an open circuit."""


# Usage
circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=timedelta(seconds=60))

@circuit
async def call_llm_provider(prompt: str) -> str:
    """Call to external LLM provider protected by circuit breaker."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post("https://api.llm-provider.com/v1/chat", json={"prompt": prompt})
        response.raise_for_status()
        return response.json()["content"]
```

## Anti-Patterns to Avoid

### :x: Synchronous Calls Between Services in Request Path

**Problem:** Service A calls Service B synchronously during an HTTP request. If Service B is slow or down, Service A's response time degrades or fails. Under load, thread pools exhaust and the entire system cascades.

**Example:**
```python
# BAD: sync inter-service call in request handler
@app.post("/orders")
async def create_order(order: OrderRequest):
    # This blocks if inventory service is slow
    inventory = await httpx.get(f"http://inventory-service/check/{order.product_id}")
    payment = await httpx.post("http://payment-service/charge", json=order.dict())
    return {"status": "created"}
```

**Solution:** Use async messaging for operations that do not need immediate response.
```python
@app.post("/orders")
async def create_order(order: OrderRequest):
    order_id = await order_repo.save(order)
    await event_publisher.publish("order.created", {"order_id": order_id, **order.dict()})
    return {"status": "accepted", "order_id": order_id}  # 202 Accepted
```

### :x: No Dead Letter Queue for Failed Messages

**Problem:** Failed messages are silently dropped or retried infinitely. Data is lost and the system has no mechanism for recovery or investigation.

**Example:**
```python
# BAD: no DLQ, no max retries
@app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True)
def process_event(self, payload):
    do_something(payload)  # retries forever on failure
```

**Solution:** Configure max retries and route failures to a DLQ.
```python
@app.task(bind=True, max_retries=3, retry_backoff=True, retry_backoff_max=60)
def process_event(self, payload):
    try:
        do_something(payload)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            send_to_dlq(payload, str(exc))
            return
        self.retry(exc=exc)
```

### :x: Unbounded Retries Without Backoff

**Problem:** A failing consumer retries immediately and infinitely, creating a tight loop that wastes resources and can amplify the problem (e.g., hammering a recovering database).

**Solution:** Always configure: max retries (3-5), exponential backoff (base 2-5s), jitter (randomized delay), and a DLQ as the terminal destination.

### :x: Tight Coupling via Shared Database

**Problem:** Multiple services read and write to the same database. Schema changes in one service break others. No service can be deployed independently. This is a distributed monolith.

**Example:**
```
Service A --> [Shared PostgreSQL] <-- Service B
                                  <-- Service C
```

**Solution:** Each service owns its data. Services communicate through events or APIs. If Service B needs data from Service A, it subscribes to Service A's events and maintains its own read model.

### :x: Fire-and-Forget Without Confirmation

**Problem:** Publishing an event without waiting for broker acknowledgment. If the broker is down or the network fails, the event is silently lost.

**Example:**
```python
# BAD: no delivery confirmation
await producer.send("events", value=payload)
# Message may not have been persisted by the broker
```

**Solution:** Wait for broker acknowledgment and handle failures.
```python
future = await producer.send_and_wait("events", value=payload)
# Message is confirmed persisted by the broker
```

## Event-Driven Systems Checklist

### Message Reliability
- [ ] All queues have Dead Letter Queues configured
- [ ] Max retry count set for every consumer (typically 3-5)
- [ ] Exponential backoff with jitter configured for retries
- [ ] Idempotency keys used in all message handlers
- [ ] Message ordering guarantees documented per topic/queue
- [ ] Outbox pattern used for DB + event consistency

### Service Resilience
- [ ] Circuit breakers configured for all external service calls
- [ ] Timeouts set on every HTTP and gRPC client
- [ ] Fallback behavior defined for circuit-open scenarios
- [ ] Health check endpoints exposed by every service
- [ ] Graceful shutdown handles in-flight messages before stopping
- [ ] Bulkhead pattern isolates critical from non-critical consumers

### Observability
- [ ] Correlation IDs propagated through all service boundaries
- [ ] Every message publish and consume is logged with correlation ID
- [ ] Queue depth metrics exported to monitoring (Prometheus/CloudWatch)
- [ ] DLQ depth alerts configured with escalation
- [ ] Distributed tracing (OpenTelemetry) spans cover async message flows
- [ ] Consumer lag monitoring for Kafka consumer groups

### Data Consistency
- [ ] Saga pattern used for multi-service transactions
- [ ] Compensating actions defined and tested for each saga step
- [ ] Event schema versioning strategy documented (e.g., Avro, JSON Schema)
- [ ] Backward-compatible schema changes enforced
- [ ] Events are self-contained (no callback required to producer)
- [ ] At-least-once delivery assumed; exactly-once enforced at application level

## Additional References

- [Celery Best Practices — Full Stack Python](https://www.fullstackpython.com/celery.html)
- [Kafka Consumer Design — Confluent](https://docs.confluent.io/platform/current/clients/consumer.html)
- [Saga Pattern — Microservices.io](https://microservices.io/patterns/data/saga.html)
- [Outbox Pattern — Debezium CDC](https://debezium.io/blog/2019/02/19/reliable-microservices-data-exchange-with-the-outbox-pattern/)
- [Temporal Workflows vs Sagas](https://temporal.io/blog/workflow-engine-principles)
