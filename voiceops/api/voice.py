"""WebSocket endpoint for the voice channel. Protocol: voiceops/voice/session.py."""

import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voiceops.voice import VoiceSession

router = APIRouter(tags=["voice"])


class _WebSocketOutbound:
    def __init__(self, ws: WebSocket):
        self.ws = ws

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.ws.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        await self.ws.send_bytes(data)


@router.websocket("/v1/voice")
async def voice(ws: WebSocket) -> None:
    """One connection = one call. Send `session.start` first, then binary audio
    frames and control messages; see the module docstring of voiceops.voice.session."""
    await ws.accept()
    session = VoiceSession(ws.app.state.voice, _WebSocketOutbound(ws))
    graceful = False
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.handle_audio(data)
            elif (text := message.get("text")) is not None and not await session.handle_message(text):
                graceful = True
                break
    except WebSocketDisconnect:
        pass
    finally:
        await session.close(graceful=graceful)
        if graceful:
            with contextlib.suppress(Exception):
                await ws.close()
