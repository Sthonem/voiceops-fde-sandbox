from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Europe/Istanbul")


def _tomorrow_at(hour: int) -> str:
    d = datetime.now(IST) + timedelta(days=1)
    return d.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


async def test_health(mock_client):
    assert (await mock_client.get("/health")).json()["status"] == "ok"


async def test_crm_lookup_by_phone_any_format(mock_client):
    for phone in ("+905550101003", "0555 010 1003", "5550101003"):
        resp = await mock_client.get("/crm/customers", params={"phone": phone})
        assert resp.status_code == 200
        assert resp.json()["id"] == "C-1003"


async def test_crm_lookup_by_id_case_insensitive(mock_client):
    resp = await mock_client.get("/crm/customers", params={"customer_id": "c-1005"})
    assert resp.json()["full_name"] == "John Carter"


async def test_crm_not_found_and_missing_key(mock_client):
    resp = await mock_client.get("/crm/customers", params={"phone": "0555 999 9999"})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "customer_not_found"
    assert (await mock_client.get("/crm/customers")).status_code == 422


async def test_transactions_sorted_newest_first(mock_client):
    txs = (await mock_client.get("/payments/customers/C-1001/transactions", params={"limit": 10})).json()
    assert [t["id"] for t in txs] == ["TX-50001", "TX-50002", "TX-50003"]


async def test_transaction_ownership_is_enforced(mock_client):
    ok = await mock_client.get("/payments/transactions/TX-50004", params={"customer_id": "C-1002"})
    assert ok.status_code == 200 and ok.json()["status"] == "under_review"
    # Same transaction, different customer -> indistinguishable from "does not exist".
    other = await mock_client.get("/payments/transactions/TX-50004", params={"customer_id": "C-1001"})
    assert other.status_code == 404


async def test_create_ticket_assigns_id_and_sla(mock_client):
    resp = await mock_client.post(
        "/helpdesk/tickets",
        json={"customer_id": "C-1001", "category": "card", "priority": "high", "summary": "Card declined abroad"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == "TCK-000004"  # three tickets are seeded
    assert body["sla_hours"] == 24
    listed = (await mock_client.get("/helpdesk/tickets", params={"customer_id": "C-1001"})).json()
    assert [t["id"] for t in listed] == ["TCK-000004"]


async def test_create_ticket_unknown_customer(mock_client):
    resp = await mock_client.post("/helpdesk/tickets", json={"customer_id": "C-9999", "summary": "hello there"})
    assert resp.status_code == 404


async def test_callback_happy_path(mock_client):
    resp = await mock_client.post(
        "/scheduling/callbacks",
        json={"customer_id": "C-1002", "scheduled_for": _tomorrow_at(11), "reason": "card limit"},
    )
    assert resp.status_code == 201
    assert resp.json()["phone_last4"] == "1002"


async def test_callback_validation_errors(mock_client):
    cases = {
        "outside_business_hours": _tomorrow_at(20),
        "in_past": (datetime.now(IST) - timedelta(hours=1)).isoformat(),
        "too_far": (datetime.now(IST) + timedelta(days=30)).replace(hour=11).isoformat(),
        "timezone_required": "2030-01-01T11:00:00",
    }
    for code, when in cases.items():
        resp = await mock_client.post(
            "/scheduling/callbacks", json={"customer_id": "C-1002", "scheduled_for": when, "reason": "test"}
        )
        assert resp.status_code == 422, code
        assert resp.json()["detail"]["code"] == code


async def test_handoff_queue(mock_client):
    first = (
        await mock_client.post("/contact-center/handoffs", json={"conversation_id": "c1", "reason": "caller_request"})
    ).json()
    second = (
        await mock_client.post("/contact-center/handoffs", json={"conversation_id": "c2", "reason": "caller_request"})
    ).json()
    urgent = (
        await mock_client.post(
            "/contact-center/handoffs", json={"conversation_id": "c3", "reason": "fraud", "priority": "urgent"}
        )
    ).json()
    assert (first["queue_position"], second["queue_position"], urgent["queue_position"]) == (1, 2, 1)
