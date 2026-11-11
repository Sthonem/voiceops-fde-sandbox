"""Agent-side persistence: conversations, transcript messages and trace events."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from voiceops.db import Base, utcnow


def _uuid() -> str:
    return str(uuid.uuid4())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    channel: Mapped[str] = mapped_column(String(16), default="text")  # text | voice
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | escalated | closed
    language: Mapped[str] = mapped_column(String(2), default="tr")
    customer_id: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    llm_provider: Mapped[str] = mapped_column(String(32), default="mock")
    # Serialized ConversationState (agent working memory between turns).
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.seq", cascade="all, delete-orphan"
    )
    traces: Mapped[list["TraceEvent"]] = relationship(
        back_populates="conversation", order_by="TraceEvent.id", cascade="all, delete-orphan"
    )


class Message(Base):
    """One transcript entry. Tool calls and tool results are stored too, so a
    conversation can be replayed to an LLM provider exactly as it happened."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    turn: Mapped[int] = mapped_column(Integer)  # user turn number this message belongs to
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool | event
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class TraceEvent(Base):
    """A span-like record of one LLM call, tool call or guard decision."""

    __tablename__ = "trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    turn: Mapped[int] = mapped_column(Integer)
    trace_id: Mapped[str] = mapped_column(String(32), index=True, default="")
    span_id: Mapped[str] = mapped_column(String(16), default="")
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kind: Mapped[str] = mapped_column(String(16))  # turn | router | llm | tool | guard | approval | stt | tts
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24))  # ok | error | blocked | needs_confirmation | pending_approval
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float] = mapped_column(default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="traces")


class ApprovalRequest(Base):
    """A sensitive action the caller confirmed that still needs a supervisor."""

    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # APR-1A2B3C4D
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    customer_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    tool: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reason: Mapped[str] = mapped_column(Text, default="")
    # pending | executed | rejected | failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    reviewer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TurnIdempotencyKey(Base):
    """A client retried POST /messages with the same Idempotency-Key: replay the
    stored reply instead of running the turn (and its side effects) twice."""

    __tablename__ = "turn_idempotency_keys"

    id: Mapped[str] = mapped_column(String(320), primary_key=True)  # "{conversation_id}:{key}"
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
