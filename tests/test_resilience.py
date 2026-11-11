"""Retries, timeouts and idempotency, exercised end to end with the mock systems'
fault injection (voiceops/mock_systems/middleware.py)."""

import random
import time

import httpx
import pytest

from voiceops.integrations import IntegrationClient, IntegrationError, derive_idempotency_key, integration_scope
from voiceops.mock_systems.app import fault_injector
from voiceops.mock_systems.middleware import FaultRuleIn
from voiceops.retry import RetryPolicy, parse_retry_after
from tests.conftest import tool_names


class Sleeps:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


def fault(path_prefix: str, **kw) -> None:
    fault_injector.add(FaultRuleIn(path_prefix=path_prefix, **kw))


def _client(mock_client, sleeps=None, **policy) -> IntegrationClient:
    return IntegrationClient(mock_client, RetryPolicy(**policy), sleep=sleeps or Sleeps(), rng=random.Random(7))


TICKET = {"customer_id": "C-1003", "category": "app", "priority": "normal", "summary": "Uygulama açılmıyor"}


async def _tickets(mock_client, customer="C-1003") -> list[dict]:
    return (await mock_client.get("/helpdesk/tickets", params={"customer_id": customer})).json()


# --------------------------------------------------------------------------- policy
def test_backoff_is_full_jitter_and_capped():
    policy = RetryPolicy(base_delay_s=0.2, max_delay_s=1.0)
    rng = random.Random(1)
    for attempt, cap in [(1, 0.2), (2, 0.4), (3, 0.8), (6, 1.0)]:
        delays = [policy.backoff(attempt, rng) for _ in range(200)]
        assert 0 <= min(delays) and max(delays) <= cap


def test_next_delay_respects_attempts_retry_after_and_deadline():
    policy = RetryPolicy(max_attempts=3, max_delay_s=2.0, deadline_s=5.0)
    assert policy.next_delay(3, 0.0, None) is None  # out of attempts
    assert policy.next_delay(1, 0.0, 1.5) == 1.5  # Retry-After honoured
    assert policy.next_delay(1, 0.0, 30.0) is None  # longer than we may wait
    assert policy.next_delay(1, 4.5, 1.0) is None  # would blow the deadline
    assert parse_retry_after("2") == 2.0 and parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


# --------------------------------------------------------------------------- reads
async def test_get_is_retried_on_503(mock_client):
    fault("/crm/customers", method="GET", status=503, times=2)
    sleeps = Sleeps()
    client = _client(mock_client, sleeps)
    with integration_scope("conv:call") as calls:
        cust = await client.get("/crm/customers", params={"customer_id": "C-1001"})
    assert cust["id"] == "C-1001"
    assert calls == [{**calls[0], "method": "GET", "status": 200, "attempts": 3}]
    assert len(sleeps.calls) == 2


async def test_client_errors_are_not_retried(mock_client):
    client = _client(mock_client)
    with integration_scope("s") as calls, pytest.raises(IntegrationError) as exc:
        await client.get("/crm/customers", params={"customer_id": "C-9999"})
    assert exc.value.status == 404 and exc.value.attempts == 1
    assert calls[0]["error"] == "customer_not_found"


async def test_429_retry_after_is_honoured(mock_client):
    fault("/payments", status=429, retry_after_s=0.5)
    sleeps = Sleeps()
    await _client(mock_client, sleeps).get("/payments/customers/C-1001/cards")
    assert sleeps.calls == [0.5]


async def test_retry_after_beyond_the_budget_fails_fast(mock_client):
    fault("/payments", status=429, retry_after_s=30)
    sleeps = Sleeps()
    with pytest.raises(IntegrationError) as exc:
        await _client(mock_client, sleeps).get("/payments/customers/C-1001/cards")
    assert exc.value.status == 429 and exc.value.attempts == 1 and sleeps.calls == []


async def test_slow_attempt_times_out_and_the_retry_succeeds(mock_client):
    fault("/crm", mode="delay", delay_ms=500)
    client = _client(mock_client, attempt_timeout_s=0.1)
    with integration_scope("s") as calls:
        cust = await client.get("/crm/customers/C-1002")
    assert cust["id"] == "C-1002" and calls[0]["attempts"] == 2


async def test_total_deadline_bounds_the_call(mock_client):
    fault("/crm", mode="delay", delay_ms=1000, times=10)
    client = IntegrationClient(mock_client, RetryPolicy(max_attempts=10, base_delay_s=0.01, attempt_timeout_s=0.08, deadline_s=0.3))
    started = time.monotonic()
    with pytest.raises(IntegrationError) as exc:
        await client.get("/crm/customers/C-1002")
    assert exc.value.code == "timeout"
    assert time.monotonic() - started < 0.6


# --------------------------------------------------------------------------- writes
async def test_lost_response_without_idempotency_key_duplicates_the_ticket(mock_client):
    """The failure mode idempotency keys exist for: the write committed, the
    response was lost, and a naive retry writes again."""
    fault("/helpdesk/tickets", method="POST", mode="error_after_commit")
    first = await mock_client.post("/helpdesk/tickets", json=TICKET)
    assert first.status_code == 503
    await mock_client.post("/helpdesk/tickets", json=TICKET)
    assert len(await _tickets(mock_client)) == 2


async def test_lost_response_with_idempotency_key_is_replayed_not_duplicated(mock_client):
    fault("/helpdesk/tickets", method="POST", mode="error_after_commit")
    client = _client(mock_client)
    with integration_scope("conv-1:call-1") as calls:
        ticket = await client.post("/helpdesk/tickets", json=TICKET)
    assert calls[0]["attempts"] == 2 and calls[0]["replayed"] is True
    assert [t["id"] for t in await _tickets(mock_client)] == [ticket["id"]]


async def test_same_tool_call_executed_twice_hits_the_same_key(mock_client):
    client = _client(mock_client)
    with integration_scope("conv-1:call-9"):
        a = await client.post("/helpdesk/tickets", json=TICKET)
    with integration_scope("conv-1:call-9") as calls:
        b = await client.post("/helpdesk/tickets", json=TICKET)
    assert a == b and calls[0]["replayed"] is True
    assert len(await _tickets(mock_client)) == 1
    assert derive_idempotency_key("conv-1:call-9", "POST", "/helpdesk/tickets") != derive_idempotency_key(
        "conv-1:call-10", "POST", "/helpdesk/tickets"
    )


async def test_reusing_a_key_for_a_different_request_is_rejected(mock_client):
    headers = {"Idempotency-Key": "key-123"}
    assert (await mock_client.post("/helpdesk/tickets", json=TICKET, headers=headers)).status_code == 201
    resp = await mock_client.post("/helpdesk/tickets", json={**TICKET, "summary": "başka bir şey"}, headers=headers)
    assert resp.status_code == 422 and resp.json()["detail"]["code"] == "idempotency_key_reused"


async def test_every_post_carries_a_key():
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key"))
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x") as http:
        client = IntegrationClient(http)
        await client.post("/a", json={})  # outside a tool call: random key
        with integration_scope("conv:call"):
            await client.post("/a", json={})
            await client.post("/a", json={})  # second POST to the same path gets its own key
    assert all(seen) and len(set(seen)) == 3


async def test_fault_admin_api(mock_client):
    resp = await mock_client.post("/_faults", json={"path_prefix": "/crm", "status": 502, "times": 2})
    assert resp.status_code == 201 and resp.json()["id"].startswith("FLT-")
    assert (await mock_client.get("/crm/customers/C-1001")).status_code == 502
    assert (await mock_client.get("/_faults")).json()[0]["hits"] == 1
    assert (await mock_client.delete("/_faults")).status_code == 204
    assert (await mock_client.get("/crm/customers/C-1001")).status_code == 200


# --------------------------------------------------------------------------- through the agent
async def test_agent_survives_a_flaky_crm_and_the_span_shows_the_retry(new_chat, client):
    fault("/crm/customers", method="GET", status=503)
    chat = await new_chat()
    turn = await chat.say("Müşteri numaram C-1001")
    assert turn["customer_id"] == "C-1001"
    spans = (await client.get(f"/v1/conversations/{chat.id}/traces")).json()[0]["spans"]
    lookup = next(s for s in spans if s["name"] == "lookup_customer")
    assert lookup["output"]["http"][0]["attempts"] == 2


async def test_agent_ticket_is_not_duplicated_when_the_response_is_lost(new_chat, mock_client):
    fault("/helpdesk/tickets", method="POST", mode="error_after_commit")
    chat = await new_chat()
    await chat.say("müşteri numaram C-1003")
    turn = await chat.say("Mobil uygulama sürekli hata veriyor, şikayet kaydı açın lütfen")
    assert tool_names(turn) == ["create_ticket"] and turn["tool_calls"][0]["status"] == "ok"
    tickets = await _tickets(mock_client)
    assert len(tickets) == 1 and tickets[0]["id"] in turn["reply"]


async def test_agent_api_idempotency_key(new_chat, client):
    chat = await new_chat()
    url = f"/v1/conversations/{chat.id}/messages"
    headers = {"Idempotency-Key": "client-retry-1"}
    first = await client.post(url, json={"text": "Müşteri numaram C-1001"}, headers=headers)
    second = await client.post(url, json={"text": "Müşteri numaram C-1001"}, headers=headers)
    assert first.status_code == second.status_code == 200
    assert second.headers["idempotent-replayed"] == "true"
    assert first.json() == second.json()
    messages = (await chat.transcript())["messages"]
    assert [m["role"] for m in messages].count("user") == 1  # the turn ran once
    reused = await client.post(url, json={"text": "başka bir şey"}, headers=headers)
    assert reused.status_code == 422 and reused.json()["detail"]["code"] == "idempotency_key_reused"
