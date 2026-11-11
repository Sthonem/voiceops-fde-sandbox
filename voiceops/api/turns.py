"""Turn execution shared by the text API and the voice WebSocket.

One function owns the critical section of a turn: per-conversation lock,
idempotent replay, load history, run the orchestrator, persist everything in one
transaction, and build the public `TurnOut`. Both channels therefore get the same
guards, transcripts and traces.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from voiceops.agent.orchestrator import EventSink, Orchestrator, TraceRecord, TurnResult
from voiceops.agent.state import ConversationState
from voiceops.api.locks import lock_conversation
from voiceops.api.persistence import persist_turn, to_chat
from voiceops.config import get_settings
from voiceops.models import Conversation, TurnIdempotencyKey
from voiceops.observability import redact
from voiceops.pricing import estimate_cost_usd


# --------------------------------------------------------------------------- schemas
class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    status: str
    latency_ms: float
    attempts: int = 1  # most HTTP attempts any single integration call needed
    replayed: bool = False  # a response came back from an idempotency key


class RoutingOut(BaseModel):
    intent: str
    confidence: float
    source: str
    sensitive_category: str | None = None
    refusal_category: str | None = None


class UsageOut(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    estimated: bool
    estimated_cost_usd: float


class TurnOut(BaseModel):
    conversation_id: str
    turn: int
    trace_id: str
    reply: str
    intent: str | None
    routing: RoutingOut | None
    language: str
    status: str
    customer_id: str | None
    verified: bool
    awaiting_confirmation: dict[str, Any] | None = None
    approval_ids: list[str] = []
    citations: list[str] = []
    tool_calls: list[ToolCallOut]
    latency_ms: float
    usage: UsageOut
    # Which prompt produced this reply, so evals can attribute a regression to a prompt change.
    prompt_version: str | None = None
    prompt_hash: str | None = None


# --------------------------------------------------------------------------- helpers
async def load_conversation(session: AsyncSession, conversation_id: str, with_traces: bool = False) -> Conversation:
    opts = [selectinload(Conversation.messages)]
    if with_traces:
        opts.append(selectinload(Conversation.traces))
    conv = await session.scalar(select(Conversation).where(Conversation.id == conversation_id).options(*opts))
    if conv is None:
        raise HTTPException(404, detail={"code": "conversation_not_found", "message": "Unknown conversation"})
    return conv


def new_conversation(channel: str, language: str, provider: str) -> Conversation:
    return Conversation(
        channel=channel,
        language=language,
        llm_provider=provider,
        state=ConversationState(language=language).model_dump(),  # type: ignore[arg-type]
    )


def build_turn_out(conversation_id: str, result: TurnResult, provider: str, cost: float) -> TurnOut:
    final_meta = result.new_messages[-1].meta
    pending = result.state.pending_action
    routing = result.routing
    return TurnOut(
        conversation_id=conversation_id,
        turn=result.state.turn,
        trace_id=result.trace_id,
        reply=result.reply,
        intent=result.intent,
        routing=RoutingOut(
            intent=routing.intent,
            confidence=routing.confidence,
            source=routing.source,
            sensitive_category=routing.sensitive_category,
            refusal_category=routing.refusal_category,
        )
        if routing
        else None,
        language=result.state.language,
        status=result.state.status,
        customer_id=result.state.customer_id,
        verified=result.state.verified,
        awaiting_confirmation={"tool": pending["tool"], "summary": pending["summary"]}
        if pending and pending.get("kind") == "confirm"
        else None,
        approval_ids=[r["id"] for r in result.approval_requests],
        citations=final_meta.get("citations", []),
        tool_calls=[
            ToolCallOut(
                name=t.name,
                arguments=redact(t.input),  # responses leave the system of record too
                status=t.status,
                latency_ms=t.latency_ms,
                attempts=max((c.get("attempts", 1) for c in t.output.get("http", [])), default=1),
                replayed=any(c.get("replayed") for c in t.output.get("http", [])),
            )
            for t in result.traces
            if t.kind == "tool"
        ],
        latency_ms=result.latency_ms,
        prompt_version=result.prompt_version,
        prompt_hash=result.prompt_hash,
        usage=UsageOut(
            provider=provider,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_read_tokens=result.usage.cache_read_tokens,
            estimated=result.usage.estimated,
            estimated_cost_usd=cost,
        ),
    )


@dataclass
class TurnExecution:
    out: TurnOut
    result: TurnResult | None = None  # None when replayed from an idempotency key
    final_seq: int | None = None
    replayed: bool = False


def _request_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def execute_turn(
    session: AsyncSession,
    orchestrator: Orchestrator,
    conversation_id: str,
    text: str,
    *,
    idempotency_key: str | None = None,
    user_meta: dict[str, Any] | None = None,
    spans: list[TraceRecord] | None = None,
    on_event: EventSink | None = None,
) -> TurnExecution:
    async with lock_conversation(session, conversation_id):
        if idempotency_key:
            rec = await session.get(TurnIdempotencyKey, f"{conversation_id}:{idempotency_key}")
            if rec is not None:
                if rec.request_hash != _request_hash(text):
                    raise HTTPException(
                        422,
                        detail={"code": "idempotency_key_reused", "message": "Key was used with a different message"},
                    )
                return TurnExecution(TurnOut.model_validate(rec.response), replayed=True)

        conv = await load_conversation(session, conversation_id)
        if conv.status == "closed":
            raise HTTPException(409, detail={"code": "conversation_closed", "message": "Conversation is closed"})
        result = await orchestrator.run_turn(
            conversation_id=conv.id,
            state=ConversationState.model_validate(conv.state),
            history=[to_chat(m) for m in conv.messages],
            user_text=text,
            channel=conv.channel,
            user_meta=user_meta,
            spans=spans,
            on_event=on_event,
        )
        cost = estimate_cost_usd(result.model, result.usage, get_settings().anthropic_model)
        final_seq = await persist_turn(session, conv, result, {"estimated_cost_usd": cost, "model": result.model})
        out = build_turn_out(conv.id, result, orchestrator.llm.name, cost)
        if idempotency_key:
            session.add(
                TurnIdempotencyKey(
                    id=f"{conversation_id}:{idempotency_key}",
                    conversation_id=conv.id,
                    request_hash=_request_hash(text),
                    response=out.model_dump(mode="json"),
                )
            )
        await session.commit()
    return TurnExecution(out, result, final_seq)
