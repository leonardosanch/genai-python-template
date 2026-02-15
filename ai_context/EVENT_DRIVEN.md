# Sistemas Distribuidos & Event-Driven

## Más Allá de HTTP

No todo es request-response. Los sistemas GenAI en producción necesitan workers, colas, eventos y procesamiento asíncrono.

---

## Arquitectura Event-Driven

```
Productor → Message Broker → Consumidor(es)
              (Kafka/RabbitMQ)
```

**Ventajas:**
- Desacoplamiento entre servicios
- Escalabilidad independiente de productores y consumidores
- Resiliencia — si un consumidor cae, los mensajes esperan
- Replay de eventos

---

## Celery — Task Queue

Procesamiento asíncrono de tareas en background.

```python
# tasks.py
from celery import Celery

celery_app = Celery("genai", broker="redis://localhost:6379/0")

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_document(self, document_id: str):
    """Procesar documento en background: chunk, embed, index."""
    try:
        document = db.get_document(document_id)
        chunks = chunk_document(document)
        embeddings = embed_batch([c.content for c in chunks])
        vector_store.upsert(chunks, embeddings)
        return {"status": "completed", "chunks": len(chunks)}
    except Exception as exc:
        self.retry(exc=exc)

@celery_app.task
def generate_report(report_id: str):
    """Generar reporte con LLM (puede tardar minutos)."""
    data = gather_report_data(report_id)
    report = llm.generate(build_report_prompt(data))
    save_report(report_id, report)
    notify_user(report_id, "Report ready")

# Llamar tareas
process_document.delay("doc-123")  # Fire and forget
result = generate_report.apply_async(args=["report-456"], countdown=10)  # Con delay
```

### Celery Beat — Scheduled Tasks

```python
celery_app.conf.beat_schedule = {
    "sync-documents-every-hour": {
        "task": "tasks.sync_documents",
        "schedule": 3600.0,
    },
    "daily-evaluation": {
        "task": "tasks.run_evaluation",
        "schedule": crontab(hour=2, minute=0),
    },
}
```

---

## RQ (Redis Queue)

Alternativa simple a Celery. Menos features, menos complejidad.

```python
from redis import Redis
from rq import Queue

redis_conn = Redis()
queue = Queue(connection=redis_conn)

# Encolar tarea
job = queue.enqueue(process_document, "doc-123")
print(job.id)  # Job ID para tracking

# Workers separados procesan las tareas
# $ uv run rq worker
```

---

## RabbitMQ

Message broker para comunicación entre servicios.

```python
import aio_pika

async def publish_event(exchange_name: str, routing_key: str, message: dict):
    connection = await aio_pika.connect_robust("amqp://localhost")
    async with connection:
        channel = await connection.channel()
        exchange = await channel.declare_exchange(exchange_name, aio_pika.ExchangeType.TOPIC)
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                content_type="application/json",
            ),
            routing_key=routing_key,
        )

async def consume_events(queue_name: str, routing_key: str, handler):
    connection = await aio_pika.connect_robust("amqp://localhost")
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)
        queue = await channel.declare_queue(queue_name, durable=True)
        exchange = await channel.declare_exchange("events", aio_pika.ExchangeType.TOPIC)
        await queue.bind(exchange, routing_key=routing_key)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body)
                    await handler(data)

# Uso
await publish_event("events", "document.created", {"id": "123", "title": "New doc"})
await consume_events("indexing-queue", "document.*", handle_document_event)
```

---

## Kafka

Streaming de eventos a gran escala. Ver [DATA_ENGINEERING.md](DATA_ENGINEERING.md) para implementación completa.

---

## gRPC

Comunicación inter-servicio de alta performance con contratos tipados.

```protobuf
// protos/llm_service.proto
syntax = "proto3";

service LLMService {
    rpc Generate (GenerateRequest) returns (GenerateResponse);
    rpc StreamGenerate (GenerateRequest) returns (stream GenerateChunk);
}

message GenerateRequest {
    string prompt = 1;
    string model = 2;
    float temperature = 3;
}

message GenerateResponse {
    string content = 1;
    int32 tokens_used = 2;
}

message GenerateChunk {
    string content = 1;
}
```

```python
# Server
import grpc
from concurrent import futures
import llm_service_pb2_grpc as pb2_grpc
import llm_service_pb2 as pb2

class LLMServicer(pb2_grpc.LLMServiceServicer):
    async def Generate(self, request, context):
        result = await llm.generate(request.prompt, model=request.model)
        return pb2.GenerateResponse(content=result, tokens_used=len(result.split()))

    async def StreamGenerate(self, request, context):
        async for chunk in llm.stream(request.prompt, model=request.model):
            yield pb2.GenerateChunk(content=chunk)

server = grpc.aio.server()
pb2_grpc.add_LLMServiceServicer_to_server(LLMServicer(), server)
await server.start()

# Client
async with grpc.aio.insecure_channel("localhost:50051") as channel:
    stub = pb2_grpc.LLMServiceStub(channel)
    response = await stub.Generate(pb2.GenerateRequest(prompt="Hello", model="gpt-4o"))
    print(response.content)

    # Streaming
    async for chunk in stub.StreamGenerate(pb2.GenerateRequest(prompt="Hello")):
        print(chunk.content, end="")
```

---

## Event Sourcing

Almacenar eventos en lugar de estado actual. Ideal para audit trails y replay.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Event:
    event_type: str
    aggregate_id: str
    data: dict
    timestamp: datetime
    version: int

class EventStore:
    async def append(self, event: Event) -> None: ...
    async def get_events(self, aggregate_id: str) -> list[Event]: ...

# Eventos de un chat session
events = [
    Event("chat.started", "session-1", {"user": "u1"}, now, 1),
    Event("message.sent", "session-1", {"content": "Hello"}, now, 2),
    Event("llm.responded", "session-1", {"content": "Hi!", "tokens": 5}, now, 3),
    Event("message.sent", "session-1", {"content": "Explain RAG"}, now, 4),
    Event("rag.retrieved", "session-1", {"docs": 5, "latency_ms": 120}, now, 5),
    Event("llm.responded", "session-1", {"content": "RAG is...", "tokens": 150}, now, 6),
]
```

---

## Patterns

### Saga Pattern

Transacciones distribuidas como secuencia de eventos.

```python
class DocumentIndexingSaga:
    """Saga para indexar un documento de forma distribuida."""

    steps = [
        ("validate", validate_document),
        ("chunk", chunk_document),
        ("embed", generate_embeddings),
        ("store", store_in_vector_db),
        ("notify", notify_completion),
    ]
    compensations = {
        "store": rollback_vector_store,
        "embed": cleanup_embeddings,
    }

    async def execute(self, document: Document):
        completed = []
        try:
            for step_name, step_fn in self.steps:
                await step_fn(document)
                completed.append(step_name)
        except Exception as e:
            # Compensar en orden inverso
            for step_name in reversed(completed):
                if comp := self.compensations.get(step_name):
                    await comp(document)
            raise
```

### Circuit Breaker

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_llm_service(prompt: str) -> str:
    """LLM call con circuit breaker.

    Si falla 5 veces consecutivas, el circuit se abre
    y rechaza calls durante 30 segundos.
    """
    return await llm.generate(prompt)
```

### Outbox Pattern

Garantizar publicación de eventos junto con cambios en DB.

```python
async def save_and_publish(session, document: Document, event: dict):
    """Guardar documento y evento en la misma transacción."""
    session.add(DocumentModel.from_entity(document))
    session.add(OutboxEvent(
        event_type="document.created",
        payload=json.dumps(event),
        published=False,
    ))
    await session.commit()
    # Un worker separado lee la outbox y publica a Kafka/RabbitMQ
```

---

## Reglas

1. **Celery para task queues** en la mayoría de casos
2. **RQ cuando Celery es overkill** — scripts simples, prototipos
3. **Kafka para streaming de eventos** a gran escala
4. **RabbitMQ para messaging** entre servicios
5. **gRPC para inter-service calls** de alta performance
6. **Circuit breakers** en toda llamada a servicio externo
7. **Idempotencia** — todo consumidor debe ser idempotente
8. **Dead letter queues** — mensajes que no se pueden procesar van a DLQ
9. **Bounded retries** — nunca retry infinito

Ver también: [STREAMING.md](STREAMING.md), [DEPLOYMENT.md](DEPLOYMENT.md), [OBSERVABILITY.md](OBSERVABILITY.md)
