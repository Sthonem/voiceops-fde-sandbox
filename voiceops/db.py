"""Async SQLAlchemy engine/session management shared by the agent API and the mock systems.

The two bounded contexts (agent vs. mock enterprise systems) share one SQLite
file in local development and tests; in Docker Compose each has its own Postgres
database. Either way the agent never queries mock-system tables: it only reaches
them through their REST APIs (see voiceops/integrations.py).
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from voiceops.config import get_settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def configure_engine(url: str | None = None) -> AsyncEngine:
    """(Re)create the global engine. Tests call this with a temporary database URL."""
    global _engine, _sessionmaker
    url = url or get_settings().database_url
    _engine = create_async_engine(url, future=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        configure_engine()
    assert _engine is not None
    return _engine


def session_factory() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        configure_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with session_factory()() as session:
        yield session


def agent_tables() -> list[Any]:
    from voiceops import models

    return [m.__table__ for m in (models.Conversation, models.Message, models.TraceEvent, models.ApprovalRequest,
                                  models.TurnIdempotencyKey)]  # fmt: skip


def mock_tables() -> list[Any]:
    from voiceops.mock_systems import models as mm

    return [m.__table__ for m in (mm.Customer, mm.Transaction, mm.Card, mm.Dispute, mm.Ticket, mm.Callback,
                                  mm.Handoff, mm.IdempotencyRecord)]  # fmt: skip


async def init_db(seed: bool = True, *, agent: bool = True, mocks: bool = True) -> None:
    """Create tables and load synthetic seed data if the mock tables are empty.

    In Docker the agent's schema is owned by Alembic migrations (agent=False here)
    and the mock systems run as their own service with their own database."""
    tables = (agent_tables() if agent else []) + (mock_tables() if mocks else [])
    if tables:
        async with get_engine().begin() as conn:
            await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=tables))
    if seed and mocks:
        from voiceops.mock_systems.seed import seed_if_empty

        async with session_factory()() as session:
            await seed_if_empty(session)


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo; treat naive datetimes read back from the DB as UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
