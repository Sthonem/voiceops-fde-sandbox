"""Mock enterprise APIs.

Each router imitates one external system a contact-center agent has to touch:

    /crm             customer master data
    /payments        core-banking transaction status
    /helpdesk        support tickets
    /scheduling      callback appointments
    /contact-center  live-agent handoff queue
    /_faults         fault injection for resilience demos and tests

All POSTs accept an `Idempotency-Key` header (see middleware.py).

Run standalone:  uvicorn voiceops.mock_systems.app:mock_app --port 8001
"""

import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voiceops.config import get_settings
from voiceops.db import as_utc, get_session, init_db, utcnow
from voiceops.mock_systems.middleware import FaultInjector, FaultMiddleware, FaultRule, FaultRuleIn, IdempotencyMiddleware
from voiceops.mock_systems.models import Callback, Card, Customer, Dispute, Handoff, Ticket, Transaction

Session = Annotated[AsyncSession, Depends(get_session)]


def normalize_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 10:
        return None
    return "+90" + digits[-10:]


def _not_found(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": code, "message": message})


async def _next_id(session: AsyncSession, model: type, prefix: str) -> str:
    n = await session.scalar(select(func.count()).select_from(model))
    return f"{prefix}-{(n or 0) + 1:06d}"


# ---------------------------------------------------------------- CRM
crm = APIRouter(prefix="/crm", tags=["mock: crm"])


class CustomerOut(BaseModel):
    id: str
    full_name: str
    phone: str
    email: str
    preferred_language: str
    segment: str


@crm.get("/customers", response_model=CustomerOut)
async def find_customer(session: Session, phone: str | None = None, customer_id: str | None = None):
    if customer_id:
        cust = await session.get(Customer, customer_id.upper())
    elif phone and (norm := normalize_phone(phone)):
        cust = await session.scalar(select(Customer).where(Customer.phone == norm))
    else:
        raise HTTPException(422, detail={"code": "missing_lookup_key", "message": "phone or customer_id required"})
    if cust is None:
        raise _not_found("customer_not_found", "No customer matches the given identifier")
    return cust


@crm.get("/customers/{customer_id}", response_model=CustomerOut)
async def get_customer(customer_id: str, session: Session):
    cust = await session.get(Customer, customer_id.upper())
    if cust is None:
        raise _not_found("customer_not_found", "No customer with this id")
    return cust


class VerifyIn(BaseModel):
    date_of_birth: date


@crm.post("/customers/{customer_id}/verify")
async def verify_customer(customer_id: str, body: VerifyIn, session: Session):
    """Step-up knowledge check. Returns only a boolean: the stored value is never echoed."""
    cust = await session.get(Customer, customer_id.upper())
    if cust is None:
        raise _not_found("customer_not_found", "No customer with this id")
    return {"customer_id": cust.id, "verified": cust.date_of_birth == body.date_of_birth}


# ---------------------------------------------------------------- Payments / core banking
payments = APIRouter(prefix="/payments", tags=["mock: payments"])


class TransactionOut(BaseModel):
    id: str
    customer_id: str
    kind: str
    amount: Decimal
    currency: str
    counterparty: str
    status: str
    status_reason: str | None
    created_at: datetime


def _tx_out(tx: Transaction) -> TransactionOut:
    out = TransactionOut.model_validate(tx, from_attributes=True)
    out.created_at = as_utc(tx.created_at)
    return out


@payments.get("/customers/{customer_id}/transactions", response_model=list[TransactionOut])
async def list_transactions(customer_id: str, session: Session, limit: int = Query(5, ge=1, le=50)):
    rows = await session.scalars(
        select(Transaction)
        .where(Transaction.customer_id == customer_id.upper())
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    return [_tx_out(t) for t in rows]


@payments.get("/transactions/{transaction_id}", response_model=TransactionOut)
async def get_transaction(transaction_id: str, session: Session, customer_id: str):
    """customer_id is mandatory: a caller may only read their own transactions, and a
    mismatch returns 404 (not 403) so transaction ids cannot be enumerated."""
    tx = await session.get(Transaction, transaction_id.upper())
    if tx is None or tx.customer_id != customer_id.upper():
        raise _not_found("transaction_not_found", "No such transaction for this customer")
    return _tx_out(tx)


class CardOut(BaseModel):
    id: str
    customer_id: str
    kind: str
    last4: str
    status: str


@payments.get("/customers/{customer_id}/cards", response_model=list[CardOut])
async def list_cards(customer_id: str, session: Session):
    rows = await session.scalars(select(Card).where(Card.customer_id == customer_id.upper()).order_by(Card.id))
    return list(rows)


class BlockIn(BaseModel):
    customer_id: str
    reason: Literal["lost", "stolen", "lost_or_stolen", "fraud_suspected"] = "lost_or_stolen"


@payments.post("/cards/{card_id}/block", response_model=CardOut)
async def block_card(card_id: str, body: BlockIn, session: Session):
    card = await session.get(Card, card_id.upper())
    if card is None or card.customer_id != body.customer_id.upper():
        raise _not_found("card_not_found", "No such card for this customer")
    if card.status == "blocked":  # idempotent: blocking twice is not an error
        return card
    card.status, card.blocked_reason, card.blocked_at = "blocked", body.reason, utcnow()
    await session.commit()
    return card


class DisputeIn(BaseModel):
    customer_id: str
    transaction_id: str
    reason: str = Field(min_length=3, max_length=500)
    approved_by: str | None = None


class DisputeOut(BaseModel):
    id: str
    transaction_id: str
    status: str
    review_days: int


@payments.post("/disputes", response_model=DisputeOut, status_code=201)
async def open_dispute(body: DisputeIn, session: Session):
    tx = await session.get(Transaction, body.transaction_id.upper())
    if tx is None or tx.customer_id != body.customer_id.upper():
        raise _not_found("transaction_not_found", "No such transaction for this customer")
    if tx.kind != "card":
        raise HTTPException(422, detail={"code": "not_disputable", "message": "Only card transactions can be disputed"})
    existing = await session.scalar(select(Dispute).where(Dispute.transaction_id == tx.id))
    if existing:
        raise HTTPException(409, detail={"code": "dispute_exists", "message": f"Dispute {existing.id} already open"})
    if tx.amount >= Decimal("10000") and not body.approved_by:
        raise HTTPException(
            403, detail={"code": "approval_required", "message": "Disputes of 10,000 TL or more need supervisor approval"}
        )
    dispute = Dispute(
        id=await _next_id(session, Dispute, "DSP"),
        customer_id=tx.customer_id,
        transaction_id=tx.id,
        reason=body.reason,
        approved_by=body.approved_by,
    )
    session.add(dispute)
    await session.commit()
    return DisputeOut(id=dispute.id, transaction_id=tx.id, status=dispute.status, review_days=30)


@payments.get("/customers/{customer_id}/disputes", response_model=list[DisputeOut])
async def list_disputes(customer_id: str, session: Session):
    rows = await session.scalars(select(Dispute).where(Dispute.customer_id == customer_id.upper()).order_by(Dispute.id))
    return [DisputeOut(id=d.id, transaction_id=d.transaction_id, status=d.status, review_days=30) for d in rows]


# ---------------------------------------------------------------- Helpdesk
helpdesk = APIRouter(prefix="/helpdesk", tags=["mock: helpdesk"])

TicketCategory = Literal["card", "payment", "account", "app", "complaint", "other"]
Priority = Literal["low", "normal", "high", "urgent"]


class TicketIn(BaseModel):
    customer_id: str
    category: TicketCategory = "other"
    priority: Priority = "normal"
    summary: str = Field(min_length=3, max_length=2000)
    conversation_id: str | None = None


class TicketOut(BaseModel):
    id: str
    customer_id: str
    category: str
    priority: str
    summary: str
    status: str
    sla_hours: int
    created_at: datetime | None = None


SLA_HOURS = {"urgent": 4, "high": 24, "normal": 48, "low": 72}


def _ticket_out(t: Ticket) -> TicketOut:
    return TicketOut(
        id=t.id,
        customer_id=t.customer_id,
        category=t.category,
        priority=t.priority,
        summary=t.summary,
        status=t.status,
        sla_hours=SLA_HOURS[t.priority],
        created_at=t.created_at,
    )


@helpdesk.post("/tickets", response_model=TicketOut, status_code=201)
async def create_ticket(body: TicketIn, session: Session):
    if await session.get(Customer, body.customer_id.upper()) is None:
        raise _not_found("customer_not_found", "Unknown customer")
    ticket = Ticket(
        id=await _next_id(session, Ticket, "TCK"),
        customer_id=body.customer_id.upper(),
        category=body.category,
        priority=body.priority,
        summary=body.summary,
        source_conversation_id=body.conversation_id,
    )
    session.add(ticket)
    await session.commit()
    return _ticket_out(ticket)


@helpdesk.get("/tickets", response_model=list[TicketOut])
async def list_tickets(session: Session, customer_id: str):
    rows = await session.scalars(select(Ticket).where(Ticket.customer_id == customer_id.upper()).order_by(Ticket.id))
    return [_ticket_out(t) for t in rows]


# ---------------------------------------------------------------- Scheduling
scheduling = APIRouter(prefix="/scheduling", tags=["mock: scheduling"])

BUSINESS_START_HOUR, BUSINESS_END_HOUR, MAX_DAYS_AHEAD = 9, 18, 14


class CallbackIn(BaseModel):
    customer_id: str
    scheduled_for: datetime  # must include a UTC offset
    reason: str = Field(min_length=3, max_length=500)


class CallbackOut(BaseModel):
    id: str
    customer_id: str
    phone_last4: str
    scheduled_for: datetime
    reason: str
    status: str


@scheduling.post("/callbacks", response_model=CallbackOut, status_code=201)
async def create_callback(body: CallbackIn, session: Session):
    cust = await session.get(Customer, body.customer_id.upper())
    if cust is None:
        raise _not_found("customer_not_found", "Unknown customer")
    if body.scheduled_for.tzinfo is None:
        raise HTTPException(422, detail={"code": "timezone_required", "message": "scheduled_for needs an offset"})
    when_utc = as_utc(body.scheduled_for)
    local = when_utc.astimezone(get_settings().tz)
    now = utcnow()
    if when_utc <= now:
        raise HTTPException(422, detail={"code": "in_past", "message": "Callback time is in the past"})
    if when_utc > now + timedelta(days=MAX_DAYS_AHEAD):
        raise HTTPException(422, detail={"code": "too_far", "message": f"Max {MAX_DAYS_AHEAD} days ahead"})
    if not (BUSINESS_START_HOUR <= local.hour < BUSINESS_END_HOUR):
        raise HTTPException(
            422,
            detail={
                "code": "outside_business_hours",
                "message": f"Callbacks run {BUSINESS_START_HOUR:02d}:00-{BUSINESS_END_HOUR:02d}:00 local time",
            },
        )
    cb = Callback(
        id=await _next_id(session, Callback, "CB"),
        customer_id=cust.id,
        phone=cust.phone,
        scheduled_for=when_utc,
        reason=body.reason,
    )
    session.add(cb)
    await session.commit()
    return CallbackOut(
        id=cb.id,
        customer_id=cb.customer_id,
        phone_last4=cb.phone[-4:],
        scheduled_for=when_utc,
        reason=cb.reason,
        status=cb.status,
    )


@scheduling.get("/callbacks", response_model=list[CallbackOut])
async def list_callbacks(session: Session, customer_id: str):
    rows = await session.scalars(select(Callback).where(Callback.customer_id == customer_id.upper()).order_by(Callback.id))
    return [
        CallbackOut(id=c.id, customer_id=c.customer_id, phone_last4=c.phone[-4:], scheduled_for=as_utc(c.scheduled_for),
                    reason=c.reason, status=c.status)  # fmt: skip
        for c in rows
    ]


# ---------------------------------------------------------------- Contact center
contact_center = APIRouter(prefix="/contact-center", tags=["mock: contact center"])


class HandoffIn(BaseModel):
    conversation_id: str
    customer_id: str | None = None
    reason: str = Field(min_length=2, max_length=500)
    language: Literal["tr", "en"] = "tr"
    priority: Priority = "normal"


class HandoffOut(BaseModel):
    id: str
    conversation_id: str
    queue_position: int
    estimated_wait_minutes: int
    status: str


@contact_center.post("/handoffs", response_model=HandoffOut, status_code=201)
async def create_handoff(body: HandoffIn, session: Session):
    queued = await session.scalar(select(func.count()).select_from(Handoff).where(Handoff.status == "queued"))
    position = 1 if body.priority == "urgent" else min((queued or 0) + 1, 9)
    ho = Handoff(
        id=await _next_id(session, Handoff, "HO"),
        conversation_id=body.conversation_id,
        customer_id=body.customer_id,
        reason=body.reason,
        language=body.language,
        priority=body.priority,
        queue_position=position,
    )
    session.add(ho)
    await session.commit()
    return HandoffOut(
        id=ho.id,
        conversation_id=ho.conversation_id,
        queue_position=position,
        estimated_wait_minutes=position * 2,
        status=ho.status,
    )


# ---------------------------------------------------------------- Fault injection
fault_injector = FaultInjector()
faults = APIRouter(prefix="/_faults", tags=["mock: fault injection"])


@faults.post("", response_model=FaultRule, status_code=201)
async def add_fault(rule: FaultRuleIn):
    """Make the next N matching requests fail, slow down, or lose their response after commit."""
    return fault_injector.add(rule)


@faults.get("", response_model=list[FaultRule])
async def list_faults():
    return fault_injector.rules


@faults.delete("", status_code=204)
async def clear_faults():
    fault_injector.clear()


# ---------------------------------------------------------------- app
@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Only runs when the mock app is served standalone (its own container, own
    # database); when mounted under the main app, that app's lifespan does it.
    await init_db(seed=get_settings().seed_on_startup, agent=False, mocks=True)
    yield


def create_mock_app() -> FastAPI:
    app = FastAPI(title="VoiceOps mock enterprise systems", version="0.1.0", lifespan=_lifespan)
    for router in (crm, payments, helpdesk, scheduling, contact_center, faults):
        app.include_router(router)
    app.add_middleware(IdempotencyMiddleware)
    app.add_middleware(FaultMiddleware, injector=fault_injector)  # added last = outermost

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "service": "mock-systems"}

    return app


mock_app = create_mock_app()
