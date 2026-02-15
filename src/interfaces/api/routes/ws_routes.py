"""WebSocket routes — bidirectional chat streaming with JWT auth."""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.infrastructure.container import Container
from src.infrastructure.security.jwt_handler import JWTHandler

router = APIRouter(prefix="/api/v1", tags=["websocket"])
logger = logging.getLogger(__name__)

_jwt = JWTHandler()

_WS_AUTH_CLOSE_CODE = 4001


async def _authenticate_ws(websocket: WebSocket) -> dict[str, object] | None:
    """Validate JWT from query param ``token`` before accepting the connection.

    Returns decoded claims on success, or ``None`` after closing the socket.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=_WS_AUTH_CLOSE_CODE, reason="Missing token")
        return None

    from jose import JWTError

    try:
        claims = _jwt.decode_token(token)
    except JWTError:
        await websocket.close(code=_WS_AUTH_CLOSE_CODE, reason="Invalid or expired token")
        return None

    if claims.get("type") != "access":
        await websocket.close(code=_WS_AUTH_CLOSE_CODE, reason="Invalid token type")
        return None

    return claims


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Bidirectional WebSocket chat with LLM streaming.

    Requires ``?token=<JWT>`` query parameter.
    Client sends JSON: {"message": "..."}
    Server streams back JSON: {"token": "..."} per chunk, then {"done": true}.
    """
    claims = await _authenticate_ws(websocket)
    if claims is None:
        return

    await websocket.accept()
    container: Container = websocket.app.state.container
    llm = container.llm_adapter

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                await websocket.send_json({"error": "Empty message"})
                continue

            try:
                async for token in llm.stream(message):
                    await websocket.send_json({"token": token})
                await websocket.send_json({"done": True})
            except Exception as e:
                logger.error("LLM streaming error in WebSocket: %s", e, exc_info=True)
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
