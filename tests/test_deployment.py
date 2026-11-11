"""Deployment concerns: migrations match the models, the layered conversation lock,
supervisor auth, split mock systems, and the demo endpoints."""

import asyncio
import os

import httpx
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from tests.conftest import TEST_DATABASE_URL
from voiceops.api.locks import advisory_key, lock_conversation
from voiceops.config import Settings
from voiceops.db import Base, agent_tables, configure_engine, get_engine, mock_tables, session_factory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alembic(url: str) -> Config:
    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    return cfg


def _sync_url(url: str) -> str:
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg") if "postgresql" in url else url.replace("+aiosqlite", "")


def test_migrations_create_exactly_the_agent_schema(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'migrated.db'}"
    cfg = _alembic(url)
    command.upgrade(cfg, "head")
    engine = create_engine(_sync_url(url))
    agent = {t.name for t in agent_tables()}
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == agent | {"alembic_version"}  # mock systems are not migrated
        ctx = MigrationContext.configure(conn, opts={"include_object": lambda o, n, t, r, c: t != "table" or n in agent,
                                                      "compare_type": True})  # fmt: skip
        assert compare_metadata(ctx, Base.metadata) == []  # models and migrations agree
    command.downgrade(cfg, "base")
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) == {"alembic_version"}
    engine.dispose()


async def test_app_with_migrated_schema_and_external_mocks(tmp_path):
    """The Docker topology in one process: agent DB created by Alembic, mock systems
    in their own database behind HTTP, no /mock mount on the agent."""
    from voiceops.main import create_app
    from voiceops.mock_systems.app import mock_app

    agent_url = f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}"
    await asyncio.to_thread(command.upgrade, _alembic(agent_url), "head")

    # Mock systems: their own database, seeded through their own lifespan.
    mocks_engine = configure_engine(f"sqlite+aiosqlite:///{tmp_path / 'mocks.db'}")
    async with mock_app.router.lifespan_context(mock_app):
        mocks_maker = session_factory()
        async with mocks_engine.connect() as conn:
            tables = set(await conn.run_sync(lambda c: inspect(c).get_table_names()))
        assert tables == {t.name for t in mock_tables()}  # no agent tables in the mock systems' database

    settings = Settings(database_url=agent_url, db_auto_create=False, mock_api_base_url="http://mock-systems", log_level="WARNING")
    app = create_app(settings)
    configure_engine(agent_url)
    async with app.router.lifespan_context(app):
        # Point the agent's integration client at the mock app over (in-memory) HTTP; the mock
        # routes resolve their sessions through the same global engine, so swap it per call.
        import voiceops.db as dbmod

        agent_maker = dbmod._sessionmaker

        class SwitchingTransport(httpx.ASGITransport):
            async def handle_async_request(self, request):
                dbmod._sessionmaker = mocks_maker
                try:
                    return await super().handle_async_request(request)
                finally:
                    dbmod._sessionmaker = agent_maker

        app.state.orchestrator.integrations._http = httpx.AsyncClient(transport=SwitchingTransport(app=mock_app), base_url="http://mock-systems")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            health = (await client.get("/health")).json()
            assert health["mock_systems"] == "http://mock-systems"
            assert (await client.get("/mock/health")).status_code == 404  # not mounted
            conv = (await client.post("/v1/conversations", json={"language": "tr"})).json()
            turn = (await client.post(f"/v1/conversations/{conv['id']}/messages", json={"text": "Müşteri numaram C-1001"})).json()
            assert turn["customer_id"] == "C-1001"
    await get_engine().dispose()
    await mocks_engine.dispose()


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="needs VOICEOPS_TEST_DATABASE_URL (Postgres)")
async def test_postgres_advisory_lock_serialises_across_sessions(db):
    maker = session_factory()
    key = advisory_key("conv-123")
    async with maker() as holder, maker() as other:
        async with lock_conversation(holder, "conv-123"):
            got = await other.scalar(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key})
            assert got is False  # a second replica would wait here
        await holder.commit()  # transaction end releases it
        assert await other.scalar(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key}) is True


def test_advisory_key_is_stable_and_signed_64_bit():
    assert advisory_key("abc") == advisory_key("abc") != advisory_key("abd")
    assert -(2**63) <= advisory_key("abc") < 2**63


async def test_supervisor_endpoints_require_the_key_when_configured(db):
    from voiceops.main import create_app

    app = create_app(Settings(supervisor_api_key="test-supervisor-key", log_level="WARNING"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            assert (await client.get("/v1/approvals")).status_code == 401
            assert (await client.get("/v1/approvals", headers={"X-API-Key": "wrong"})).status_code == 401
            assert (await client.get("/v1/approvals", headers={"X-API-Key": "test-supervisor-key"})).status_code == 200
            assert (await client.get("/v1/demo/faults")).status_code == 401
            assert (await client.get("/health")).json()["supervisor_auth"] == "api_key"
            # The caller-facing API stays open (the telephony/web gateway authenticates sessions).
            assert (await client.post("/v1/conversations", json={})).status_code == 201


async def test_demo_page_and_fault_proxy(client):
    page = await client.get("/demo")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    added = await client.post("/v1/demo/faults", json={"path_prefix": "/crm", "status": 503, "times": 1})
    assert added.status_code == 201 and added.json()["id"].startswith("FLT-")
    assert [f["path_prefix"] for f in (await client.get("/v1/demo/faults")).json()] == ["/crm"]
    assert (await client.delete("/v1/demo/faults")).status_code == 204
    assert (await client.get("/v1/demo/faults")).json() == []
    assert (await client.get("/health")).json()["supervisor_auth"].startswith("disabled")
