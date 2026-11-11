"""Tables owned by the mock enterprise systems. The agent must not import these."""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from voiceops.db import Base, utcnow


class Customer(Base):  # CRM
    __tablename__ = "crm_customers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # C-1001
    full_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # E.164, synthetic
    email: Mapped[str] = mapped_column(String(128))
    preferred_language: Mapped[str] = mapped_column(String(2))  # tr | en
    segment: Mapped[str] = mapped_column(String(16))  # retail | premium | sme
    date_of_birth: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Transaction(Base):  # Core banking / payments
    __tablename__ = "core_transactions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # TX-50001
    customer_id: Mapped[str] = mapped_column(ForeignKey("crm_customers.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # transfer | card | bill
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3), default="TRY")
    counterparty: Mapped[str] = mapped_column(String(128))
    # completed | pending | failed | reversed | under_review
    status: Mapped[str] = mapped_column(String(16))
    status_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Card(Base):  # Core banking / cards
    __tablename__ = "core_cards"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # CRD-100101
    customer_id: Mapped[str] = mapped_column(ForeignKey("crm_customers.id"), index=True)
    kind: Mapped[str] = mapped_column(String(8))  # debit | credit
    last4: Mapped[str] = mapped_column(String(4))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | blocked
    blocked_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    blocked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Dispute(Base):  # Core banking / chargebacks
    __tablename__ = "core_disputes"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # DSP-000001
    customer_id: Mapped[str] = mapped_column(ForeignKey("crm_customers.id"), index=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("core_transactions.id"), unique=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="open")
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Ticket(Base):  # Helpdesk
    __tablename__ = "helpdesk_tickets"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # TCK-000123
    customer_id: Mapped[str] = mapped_column(ForeignKey("crm_customers.id"), index=True)
    category: Mapped[str] = mapped_column(String(32))
    priority: Mapped[str] = mapped_column(String(8))  # low | normal | high | urgent
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="open")
    source_conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Callback(Base):  # Scheduling
    __tablename__ = "scheduling_callbacks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # CB-000045
    customer_id: Mapped[str] = mapped_column(ForeignKey("crm_customers.id"), index=True)
    phone: Mapped[str] = mapped_column(String(20))
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Handoff(Base):  # Contact-center queue (human agents)
    __tablename__ = "contact_center_handoffs"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # HO-000007
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    customer_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(2))
    priority: Mapped[str] = mapped_column(String(8))
    queue_position: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column(String(16), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdempotencyRecord(Base):  # Shared API gateway of the mock systems
    """Stored response for a POST that carried an Idempotency-Key (Stripe-style)."""

    __tablename__ = "mock_idempotency_keys"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    method: Mapped[str] = mapped_column(String(8))
    path: Mapped[str] = mapped_column(String(255))
    request_hash: Mapped[str] = mapped_column(String(64))
    status_code: Mapped[int] = mapped_column()
    response_body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
