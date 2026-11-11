"""Gateway behaviour of the mock systems: idempotency keys and fault injection.

Request path (outermost first):

    FaultMiddleware        injected 5xx / 429 / latency, or "response lost after commit"
      IdempotencyMiddleware  replay the stored response for a repeated Idempotency-Key
        routes

Fault injection is how the retry and idempotency logic is demonstrated and tested
end to end: e.g. `error_after_commit` lets the ticket be created, then replaces
the response with a 503, which is exactly the case where a naive retry would open
a duplicate ticket.

Simplification versus production: the idempotency record is written after the
route has committed, not in the same database transaction. A crash between the
two would let a retry run again. Card blocking and disputes are also idempotent
by their nature (state transition; one dispute per transaction).
"""

import asyncio
import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from voiceops.db import session_factory

FaultMode = Literal["error", "error_after_commit", "delay"]


def _path(scope: Scope) -> str:
    path, root = scope["path"], scope.get("root_path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _header(scope: Scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v.decode("latin-1")
    return None


async def _send_json(send: Send, status: int, payload: Any, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *(headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------- faults
class FaultRuleIn(BaseModel):
    path_prefix: str = Field(description="e.g. /helpdesk/tickets")
    method: str | None = Field(default=None, description="GET, POST, ... or any")
    mode: FaultMode = "error"
    status: int = Field(default=503, ge=400, le=599)
    delay_ms: int = Field(default=0, ge=0, le=60000)
    retry_after_s: float | None = Field(default=None, ge=0)
    times: int = Field(default=1, ge=1, le=1000, description="how many matching requests to affect")


class FaultRule(FaultRuleIn):
    id: str = Field(default_factory=lambda: f"FLT-{uuid.uuid4().hex[:6].upper()}")
    hits: int = 0


class FaultInjector:
    def __init__(self) -> None:
        self.rules: list[FaultRule] = []

    def add(self, rule: FaultRuleIn) -> FaultRule:
        r = FaultRule(**rule.model_dump())
        self.rules.append(r)
        return r

    def clear(self) -> None:
        self.rules.clear()

    def take(self, method: str, path: str) -> FaultRule | None:
        for r in self.rules:
            if r.hits < r.times and path.startswith(r.path_prefix) and (r.method in (None, "", method)):
                r.hits += 1
                return r
        return None


class FaultMiddleware:
    def __init__(self, app: ASGIApp, injector: FaultInjector):
        self.app = app
        self.injector = injector

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _path(scope).startswith("/_faults"):
            await self.app(scope, receive, send)
            return
        rule = self.injector.take(scope["method"], _path(scope))
        if rule is None:
            await self.app(scope, receive, send)
            return
        if rule.mode == "delay":
            await asyncio.sleep(rule.delay_ms / 1000)
            await self.app(scope, receive, send)
            return
        if rule.mode == "error_after_commit":

            async def discard(_: Message) -> None:
                return None

            await self.app(scope, receive, discard)  # the work is done, the response is "lost"
        headers = [(b"retry-after", str(rule.retry_after_s).encode())] if rule.retry_after_s is not None else []
        await _send_json(
            send,
            rule.status,
            {"detail": {"code": "injected_fault", "message": f"Injected {rule.status} ({rule.mode}, {rule.id})"}},
            headers,
        )


# --------------------------------------------------------------------------- idempotency
class IdempotencyMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        is_post = scope["type"] == "http" and scope["method"] == "POST" and not _path(scope).startswith("/_faults")
        key = _header(scope, b"idempotency-key") if is_post else None
        if not key:
            await self.app(scope, receive, send)
            return
        if len(key) > 255:
            await _send_json(send, 400, {"detail": {"code": "invalid_idempotency_key", "message": "Key too long"}})
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        path = _path(scope)
        request_hash = hashlib.sha256(b"POST " + path.encode() + b"\n" + body).hexdigest()

        from voiceops.mock_systems.models import IdempotencyRecord

        async with session_factory()() as session:
            rec = await session.get(IdempotencyRecord, key)
        if rec is not None:
            if rec.request_hash != request_hash:
                await _send_json(
                    send,
                    422,
                    {"detail": {"code": "idempotency_key_reused", "message": "Key was used with a different request"}},
                )
                return
            await _send_json(send, rec.status_code, json.loads(rec.response_body), [(b"idempotent-replayed", b"true")])
            return

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        start: Message | None = None
        chunks: list[bytes] = []

        async def capture(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await self.app(scope, replay_receive, capture)
        assert start is not None
        response_body = b"".join(chunks)
        if start["status"] < 500:  # server errors stay retryable
            async with session_factory()() as session:
                session.add(
                    IdempotencyRecord(
                        key=key,
                        method="POST",
                        path=path,
                        request_hash=request_hash,
                        status_code=start["status"],
                        response_body=response_body.decode() or "null",
                    )
                )
                try:
                    await session.commit()
                except IntegrityError:  # a concurrent duplicate stored it first
                    await session.rollback()
        await send(start)
        await send({"type": "http.response.body", "body": response_body})
