import os
from collections.abc import AsyncIterator

import httpx
import pytest

from voiceops.db import Base, agent_tables, configure_engine, get_engine, init_db, mock_tables

# Run the suite against Postgres with e.g.
#   VOICEOPS_TEST_DATABASE_URL=postgresql+asyncpg://voiceops@localhost:55432/voiceops_test pytest
TEST_DATABASE_URL = os.environ.get("VOICEOPS_TEST_DATABASE_URL")
from voiceops.integrations import IntegrationClient
from voiceops.mock_systems.app import mock_app


@pytest.fixture(autouse=True)
def _reset_faults():
    """Fault rules live on the module-level mock app; never leak them between tests."""
    from voiceops.mock_systems.app import fault_injector

    fault_injector.clear()
    yield
    fault_injector.clear()


@pytest.fixture
async def db(tmp_path) -> AsyncIterator[None]:
    """Fresh, seeded database per test (SQLite file, or Postgres when configured)."""
    if TEST_DATABASE_URL:
        from sqlalchemy import text

        configure_engine(TEST_DATABASE_URL)
        async with get_engine().begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            await conn.run_sync(lambda c: Base.metadata.drop_all(c, tables=agent_tables() + mock_tables()))
    else:
        configure_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(seed=True)
    yield
    await get_engine().dispose()


@pytest.fixture
async def mock_client(db) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://mock") as c:
        yield c


@pytest.fixture
async def integrations(mock_client) -> IntegrationClient:
    return IntegrationClient(mock_client)


@pytest.fixture
async def app(db):
    """The agent app with its real lifespan (orchestrator, mock LLM, mock speech, in-process integrations)."""
    from voiceops.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://voiceops") as c:
        yield c


class Chat:
    """Helper to drive one conversation through the public API."""

    def __init__(self, client: httpx.AsyncClient, conversation_id: str):
        self.client = client
        self.id = conversation_id

    async def say(self, text: str) -> dict:
        resp = await self.client.post(f"/v1/conversations/{self.id}/messages", json={"text": text})
        assert resp.status_code == 200, resp.text
        return resp.json()

    async def transcript(self) -> dict:
        resp = await self.client.get(f"/v1/conversations/{self.id}")
        assert resp.status_code == 200, resp.text
        return resp.json()


@pytest.fixture
def new_chat(client):
    async def _new(language: str = "tr") -> Chat:
        resp = await client.post("/v1/conversations", json={"language": language})
        assert resp.status_code == 201, resp.text
        return Chat(client, resp.json()["id"])

    return _new


def tool_names(turn: dict) -> list[str]:
    return [c["name"] for c in turn["tool_calls"]]
