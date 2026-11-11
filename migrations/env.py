"""Alembic environment: async engine, agent tables only."""

import asyncio
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from voiceops.config import Settings
from voiceops.db import Base, agent_tables

config = context.config
if config.config_file_name and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

AGENT_TABLES = {t.name for t in agent_tables()}
target_metadata = Base.metadata


def include_object(obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any) -> bool:
    """Only the agent's tables are migrated; mock-system tables are out of scope."""
    if type_ == "table":
        return name in AGENT_TABLES
    table = getattr(obj, "table", None)
    return table is None or table.name in AGENT_TABLES


def database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or Settings().database_url


def _configure(**kw: Any) -> None:
    context.configure(target_metadata=target_metadata, include_object=include_object, compare_type=True, **kw)


def run_migrations_offline() -> None:
    url = database_url()
    _configure(url=url, literal_binds=True, render_as_batch=url.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection) -> None:
    _configure(connection=connection, render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(database_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run_sync)
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(_run_async())
