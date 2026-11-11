"""Supervisor queue for high-value sensitive actions (human-in-the-loop).

The agent can *request* a high-value dispute; only a named human can approve it.
On approval the action runs immediately against the core system (with the
approver recorded there too) and the caller is told on their next turn.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voiceops.agent.orchestrator import Orchestrator, TraceRecord
from voiceops.agent.state import ChatMessage, ConversationState
from voiceops.agent.tools import ToolContext
from voiceops.api.conversations import get_orchestrator, load_conversation
from voiceops.api.locks import lock_conversation
from voiceops.api.persistence import add_message, add_trace, next_seq
from voiceops.api.security import require_supervisor
from voiceops.db import as_utc, get_session, utcnow
from voiceops.models import ApprovalRequest

router = APIRouter(prefix="/v1/approvals", tags=["approvals"], dependencies=[Depends(require_supervisor)])
Session = Annotated[AsyncSession, Depends(get_session)]


class ApprovalOut(BaseModel):
    id: str
    conversation_id: str
    customer_id: str | None
    tool: str
    summary: dict[str, Any]
    reason: str
    status: str
    reviewer: str | None
    review_note: str | None
    result: dict[str, Any]
    created_at: datetime
    decided_at: datetime | None


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=2, max_length=64)
    note: str | None = Field(default=None, max_length=500)


def _out(a: ApprovalRequest) -> ApprovalOut:
    return ApprovalOut(
        id=a.id,
        conversation_id=a.conversation_id,
        customer_id=a.customer_id,
        tool=a.tool,
        summary=a.summary,
        reason=a.reason,
        status=a.status,
        reviewer=a.reviewer,
        review_note=a.review_note,
        result=a.result,
        created_at=as_utc(a.created_at),
        decided_at=as_utc(a.decided_at) if a.decided_at else None,
    )


@router.get("", response_model=list[ApprovalOut])
async def list_approvals(session: Session, status: str | None = "pending"):
    q = select(ApprovalRequest).order_by(ApprovalRequest.created_at)
    if status:
        q = q.where(ApprovalRequest.status == status)
    return [_out(a) for a in await session.scalars(q)]


@router.get("/{approval_id}", response_model=ApprovalOut)
async def get_approval(approval_id: str, session: Session):
    a = await session.get(ApprovalRequest, approval_id)
    if a is None:
        raise HTTPException(404, detail={"code": "approval_not_found", "message": "Unknown approval"})
    return _out(a)


@router.post("/{approval_id}/decision", response_model=ApprovalOut)
async def decide(
    approval_id: str,
    body: DecisionIn,
    session: Session,
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
):
    a = await session.get(ApprovalRequest, approval_id)
    if a is None:
        raise HTTPException(404, detail={"code": "approval_not_found", "message": "Unknown approval"})
    async with lock_conversation(session, a.conversation_id):
        await session.refresh(a)
        if a.status != "pending":
            raise HTTPException(409, detail={"code": "already_decided", "message": f"Approval is {a.status}"})
        conv = await load_conversation(session, a.conversation_id)
        state = ConversationState.model_validate(conv.state)

        span = TraceRecord("approval", f"{a.tool}:{body.decision}", "ok", input={"approval_id": a.id, "reviewer": body.reviewer})
        if body.decision == "approve":
            ctx = ToolContext(orchestrator.integrations, state, conv.id)
            outcome = await orchestrator.registry.execute_approved(
                a.tool, a.arguments, ctx, approver=body.reviewer, scope=f"approval:{a.id}"
            )
            a.status = "executed" if outcome.status == "ok" else "failed"
            a.result = outcome.payload
            span.status, span.output = outcome.status, {**outcome.payload, "http": outcome.calls}
            notice = {
                "approval_id": a.id,
                "decision": "approved" if outcome.status == "ok" else "failed",
                "reference": outcome.payload.get("id"),
                "reason": outcome.payload.get("error"),
            }
        else:
            a.status = "rejected"
            notice = {"approval_id": a.id, "decision": "rejected"}
        a.reviewer, a.review_note, a.decided_at = body.reviewer, body.note, utcnow()

        if state.pending_action and state.pending_action.get("approval_id") == a.id:
            state.pending_action = None
        if state.status == "active":
            state.notifications.append(notice)
        conv.state = state.model_dump()

        event = ChatMessage(
            role="event",
            content=f"Supervisor {body.reviewer} {a.status} {a.tool} ({a.id})",
            meta={"approval_id": a.id, "decision": body.decision, "status": a.status},
        )
        add_message(session, conv.id, await next_seq(session, conv.id), state.turn, event)
        add_trace(session, conv.id, state.turn, uuid.uuid4().hex, span)
        await session.commit()
    return _out(a)
