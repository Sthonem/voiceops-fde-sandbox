"""Public agent API: create a conversation, send a turn, read the transcript and traces."""

from collections import defaultdict
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from voiceops.agent.orchestrator import Orchestrator
from voiceops.api.turns import TurnOut, execute_turn, load_conversation, new_conversation
from voiceops.db import as_utc, get_session
from voiceops.models import Conversation

__all__ = ["router", "get_orchestrator", "load_conversation"]

router = APIRouter(prefix="/v1", tags=["conversations"])
Session = Annotated[AsyncSession, Depends(get_session)]


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


# --------------------------------------------------------------------------- schemas
class ConversationCreate(BaseModel):
    channel: Literal["text", "voice"] = "text"
    language: Literal["tr", "en"] = "tr"


class ConversationOut(BaseModel):
    id: str
    channel: str
    status: str
    language: str
    customer_id: str | None
    llm_provider: str
    created_at: datetime


class TurnIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class MessageOut(BaseModel):
    seq: int
    turn: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]]
    tool_name: str | None
    is_error: bool
    meta: dict[str, Any]
    created_at: datetime


class SpanOut(BaseModel):
    span_id: str
    parent_span_id: str | None
    kind: str
    name: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    input_tokens: int
    output_tokens: int


class TurnTraceOut(BaseModel):
    turn: int
    trace_id: str
    spans: list[SpanOut]


class TranscriptOut(ConversationOut):
    state: dict[str, Any]
    messages: list[MessageOut]
    traces: list[TurnTraceOut]


# --------------------------------------------------------------------------- helpers
def _conversation_out(c: Conversation) -> dict[str, Any]:
    return {
        "id": c.id,
        "channel": c.channel,
        "status": c.status,
        "language": c.language,
        "customer_id": c.customer_id,
        "llm_provider": c.llm_provider,
        "created_at": as_utc(c.created_at),
    }


def _grouped_traces(conv: Conversation) -> list[dict[str, Any]]:
    turns: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for t in conv.traces:
        turns[(t.turn, t.trace_id)].append(
            {
                "span_id": t.span_id,
                "parent_span_id": t.parent_span_id,
                "kind": t.kind,
                "name": t.name,
                "status": t.status,
                "input": t.input,
                "output": t.output,
                "latency_ms": t.latency_ms,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
            }
        )
    return [{"turn": turn, "trace_id": tid, "spans": spans} for (turn, tid), spans in sorted(turns.items())]


# --------------------------------------------------------------------------- routes
@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(body: ConversationCreate, session: Session, request: Request):
    conv = new_conversation(body.channel, body.language, request.app.state.orchestrator.llm.name)
    session.add(conv)
    await session.commit()
    return _conversation_out(conv)


@router.post("/conversations/{conversation_id}/messages", response_model=TurnOut)
async def send_message(
    conversation_id: str,
    body: TurnIn,
    session: Session,
    response: Response,
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
    idempotency_key: Annotated[str | None, Header(max_length=255)] = None,
):
    """Run one caller turn. Send an `Idempotency-Key` header to make client retries
    safe: a repeated key returns the stored reply without running the turn again."""
    turn = await execute_turn(session, orchestrator, conversation_id, body.text.strip(), idempotency_key=idempotency_key)
    if turn.replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return turn.out


@router.get("/conversations/{conversation_id}", response_model=TranscriptOut)
async def get_conversation(conversation_id: str, session: Session):
    conv = await load_conversation(session, conversation_id, with_traces=True)
    return {
        **_conversation_out(conv),
        "state": conv.state,
        "messages": [
            {
                "seq": m.seq,
                "turn": m.turn,
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_name": m.tool_name,
                "is_error": m.is_error,
                "meta": m.meta,
                "created_at": as_utc(m.created_at),
            }
            for m in conv.messages
        ],
        "traces": _grouped_traces(conv),
    }


@router.get("/conversations/{conversation_id}/traces", response_model=list[TurnTraceOut])
async def get_traces(conversation_id: str, session: Session):
    """Span tree per turn (root `turn` span → router / llm / tool / guard children)."""
    return _grouped_traces(await load_conversation(session, conversation_id, with_traces=True))
