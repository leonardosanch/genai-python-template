# Streaming

## Por qué Streaming

Los LLMs generan tokens secuencialmente. Sin streaming, el usuario espera hasta que toda la respuesta esté completa. Con streaming, ve los tokens a medida que se generan.

---

## Patrones

### SSE (Server-Sent Events)

Comunicación unidireccional servidor → cliente. Ideal para streaming de LLM.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

async def generate_stream(query: str):
    async for chunk in llm.stream(query):
        yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

**Cliente (JavaScript):**

```javascript
const eventSource = new EventSource("/api/chat/stream");
eventSource.onmessage = (event) => {
    if (event.data === "[DONE]") {
        eventSource.close();
        return;
    }
    const data = JSON.parse(event.data);
    appendToUI(data.content);
};
```

### WebSockets

Comunicación bidireccional. Para chat interactivo con múltiples turnos, notificaciones en tiempo real, y colaboración.

#### Básico — Chat con LLM

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            async for chunk in llm.stream(data["message"]):
                await websocket.send_json({"type": "chunk", "content": chunk})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
```

#### Connection Manager — Múltiples clientes

```python
class ConnectionManager:
    """Gestiona múltiples conexiones WebSocket."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self._connections.pop(client_id, None)

    async def send_to(self, client_id: str, message: dict):
        if ws := self._connections.get(client_id):
            await ws.send_json(message)

    async def broadcast(self, message: dict):
        for ws in self._connections.values():
            await ws.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Procesar y responder
            result = await process_message(data)
            await manager.send_to(client_id, result)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

#### Rooms / Channels

```python
class RoomManager:
    """WebSocket rooms para colaboración multi-usuario."""

    def __init__(self):
        self._rooms: dict[str, set[WebSocket]] = {}

    async def join(self, room_id: str, websocket: WebSocket):
        if room_id not in self._rooms:
            self._rooms[room_id] = set()
        self._rooms[room_id].add(websocket)

    async def leave(self, room_id: str, websocket: WebSocket):
        if room_id in self._rooms:
            self._rooms[room_id].discard(websocket)

    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocket | None = None):
        for ws in self._rooms.get(room_id, set()):
            if ws != exclude:
                await ws.send_json(message)
```

#### Authentication en WebSockets

```python
from fastapi import WebSocket, Query, status

@app.websocket("/ws/chat")
async def authenticated_ws(
    websocket: WebSocket,
    token: str = Query(...),
):
    user = await verify_token(token)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    # ... chat con user autenticado
```

#### Heartbeat / Keep-alive

```python
import asyncio

async def websocket_with_heartbeat(websocket: WebSocket):
    await websocket.accept()

    async def heartbeat():
        while True:
            try:
                await websocket.send_json({"type": "ping"})
                await asyncio.sleep(30)
            except Exception:
                break

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        while True:
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=60,  # Timeout si no hay actividad
            )
            if data.get("type") == "pong":
                continue
            result = await process(data)
            await websocket.send_json(result)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        heartbeat_task.cancel()
```

#### WebSockets con Django Channels

```python
# consumers.py
from channels.generic.websocket import AsyncJsonWebsocketConsumer

class ChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group = f"chat_{self.room_name}"

        await self.channel_layer.group_add(self.room_group, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group, self.channel_name)

    async def receive_json(self, content):
        message = content["message"]

        # Stream LLM response al grupo
        async for chunk in llm.stream(message):
            await self.channel_layer.group_send(
                self.room_group,
                {"type": "chat.message", "content": chunk, "sender": self.scope["user"].id},
            )

    async def chat_message(self, event):
        await self.send_json(event)

# routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/chat/(?P<room_name>\w+)/$", consumers.ChatConsumer.as_asgi()),
]
```

---

## Async Generators

Patrón base para streaming en Python.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_llm(prompt: str, model: str = "gpt-4o"):
    """Async generator que yield chunks del LLM."""
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### Composición de Streams

```python
async def stream_with_metadata(query: str):
    """Stream que incluye metadata al final."""
    total_tokens = 0
    full_response = []

    async for chunk in stream_llm(query):
        full_response.append(chunk)
        yield {"type": "chunk", "content": chunk}

    # Metadata al final del stream
    yield {
        "type": "metadata",
        "total_length": len("".join(full_response)),
        "model": "gpt-4o",
    }
```

---

## Streaming con RAG

```python
async def stream_rag(query: str):
    """RAG pipeline con streaming de la respuesta."""
    # 1. Retrieve (no se streamea)
    docs = await vector_store.search(query, top_k=5)
    context = format_context(docs)

    # 2. Enviar fuentes primero
    yield {
        "type": "sources",
        "sources": [{"title": d.metadata.get("title"), "url": d.metadata.get("url")} for d in docs],
    }

    # 3. Stream de la generación
    prompt = build_rag_prompt(query, context)
    async for chunk in stream_llm(prompt):
        yield {"type": "chunk", "content": chunk}

    yield {"type": "done"}
```

---

## Streaming con Agentes

```python
async def stream_agent(query: str):
    """Stream de un agente multi-step."""
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=query)]},
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            yield {"type": "token", "content": event["data"]["chunk"].content}
        elif kind == "on_tool_start":
            yield {"type": "tool_start", "tool": event["name"]}
        elif kind == "on_tool_end":
            yield {"type": "tool_end", "tool": event["name"], "result": str(event["data"])}
```

---

## Async / Concurrency

### asyncio Fundamentals

```python
import asyncio

# Ejecutar múltiples LLM calls en paralelo
async def parallel_generate(prompts: list[str]) -> list[str]:
    tasks = [llm.generate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Con semáforo para rate limiting
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent calls

async def rate_limited_generate(prompt: str) -> str:
    async with semaphore:
        return await llm.generate(prompt)
```

### Task Queues

Para procesamiento asíncrono de larga duración.

```python
# Con asyncio.Queue
queue: asyncio.Queue[Task] = asyncio.Queue()

async def worker():
    while True:
        task = await queue.get()
        try:
            result = await process(task)
            await task.set_result(result)
        except Exception as e:
            await task.set_error(e)
        finally:
            queue.task_done()

# Iniciar workers
workers = [asyncio.create_task(worker()) for _ in range(3)]
```

---

## Reglas

1. **SSE para streaming unidireccional** (respuestas de LLM)
2. **WebSockets para chat bidireccional** (múltiples turnos)
3. **Siempre async** — nunca bloquear el event loop con LLM calls
4. **Timeouts en todo stream** — evitar conexiones colgadas
5. **Graceful shutdown** — cerrar streams activos limpiamente
6. **Backpressure** — no generar más rápido de lo que el cliente consume

Ver también: [ARCHITECTURE.md](ARCHITECTURE.md), [OBSERVABILITY.md](OBSERVABILITY.md)
