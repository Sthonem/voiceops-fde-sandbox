"""Per-conversation serialisation: a caller can't talk over themselves, and a
supervisor decision can't race a caller turn.

Two layers:

* an in-process `asyncio.Lock` per conversation (cheap, and the only layer SQLite needs)
* on Postgres, a transaction-scoped advisory lock keyed by the conversation id, so
  turns stay serialised when several API replicas run behind a load balancer. It is
  released automatically at commit or rollback, so a crashed replica can't leave it
  held. This replaces the Redis lock the roadmap originally planned: one fewer
  moving part to operate at a customer site.
"""

import asyncio
import hashlib
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

conversation_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def advisory_key(conversation_id: str) -> int:
    """Stable signed 64-bit key (Postgres advisory locks take a bigint)."""
    return int.from_bytes(hashlib.blake2b(conversation_id.encode(), digest_size=8).digest(), "big", signed=True)


@asynccontextmanager
async def lock_conversation(session: AsyncSession, conversation_id: str) -> AsyncIterator[None]:
    async with conversation_locks[conversation_id]:
        if session.bind.dialect.name == "postgresql":
            await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": advisory_key(conversation_id)})
        yield
