"""Write-side helpers shared by the conversation and approval APIs."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voiceops.agent.orchestrator import TraceRecord, TurnResult
from voiceops.agent.state import ChatMessage, ToolCallRequest
from voiceops.models import ApprovalRequest, Conversation, Message, TraceEvent
from voiceops.observability import redact


def to_chat(m: Message) -> ChatMessage:
    return ChatMessage(
        role=m.role,  # type: ignore[arg-type]
        content=m.content,
        tool_calls=[ToolCallRequest(c["id"], c["name"], c["arguments"]) for c in m.tool_calls],
        tool_call_id=m.tool_call_id,
        tool_name=m.tool_name,
        is_error=m.is_error,
        meta=m.meta,
    )


async def next_seq(session: AsyncSession, conversation_id: str) -> int:
    return (await session.scalar(select(func.max(Message.seq)).where(Message.conversation_id == conversation_id)) or 0) + 1


def add_message(session: AsyncSession, conversation_id: str, seq: int, turn: int, m: ChatMessage) -> None:
    session.add(
        Message(
            conversation_id=conversation_id,
            seq=seq,
            turn=turn,
            role=m.role,
            content=m.content,
            tool_calls=[c.to_dict() for c in m.tool_calls],
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            is_error=m.is_error,
            meta=m.meta,
        )
    )


def add_trace(session: AsyncSession, conversation_id: str, turn: int, trace_id: str, t: TraceRecord) -> None:
    """Trace payloads are redacted on the way in: they feed dashboards and exports."""
    session.add(
        TraceEvent(
            conversation_id=conversation_id,
            turn=turn,
            trace_id=trace_id,
            span_id=t.span_id,
            parent_span_id=t.parent_id,
            kind=t.kind,
            name=t.name,
            status=t.status,
            input=redact(t.input),
            output=redact(t.output),
            latency_ms=t.latency_ms,
            input_tokens=t.input_tokens,
            output_tokens=t.output_tokens,
        )
    )


async def persist_turn(session: AsyncSession, conv: Conversation, result: TurnResult, extra: dict[str, Any]) -> int:
    """Stage the turn's messages, spans, approval requests and state. Returns the
    seq of the final assistant message (the voice channel annotates it later)."""
    seq = await next_seq(session, conv.id)
    turn = result.state.turn
    for i, m in enumerate(result.new_messages):
        add_message(session, conv.id, seq + i, turn, m)
    for t in result.traces:
        if t.kind == "turn":
            t.output = {**t.output, **extra}
        add_trace(session, conv.id, turn, result.trace_id, t)
    for req in result.approval_requests:
        session.add(
            ApprovalRequest(
                id=req["id"],
                conversation_id=conv.id,
                customer_id=req["customer_id"],
                tool=req["tool"],
                arguments=req["arguments"],
                summary=req["summary"],
                reason=req["reason"],
            )
        )
    conv.state = result.state.model_dump()
    conv.status = result.state.status
    conv.language = result.state.language
    conv.customer_id = result.state.customer_id
    return seq + len(result.new_messages) - 1
