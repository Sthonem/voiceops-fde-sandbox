"""Sensitive actions end to end: step-up verification, caller confirmation, supervisor approval,
policy-mandated handoffs and grounded policy answers."""

import pytest
from conftest import tool_names


async def _verified(new_chat, customer: str, dob: str, opener: str, lang: str = "tr"):
    chat = await new_chat(lang)
    t1 = await chat.say(f"{customer} {opener}")
    assert tool_names(t1) == ["lookup_customer"]
    t2 = await chat.say(dob)
    assert t2["verified"] is True
    return chat, t2


# ------------------------------------------------------------------ card block
async def test_block_card_full_flow(new_chat, mock_client):
    chat, t2 = await _verified(new_chat, "C-1001", "12.04.1988", "kredi kartım çalındı")
    assert tool_names(t2) == ["verify_identity", "block_card"]
    assert t2["tool_calls"][1]["status"] == "needs_confirmation"
    assert t2["awaiting_confirmation"]["summary"] == {"card_last4": "7710", "card_kind": "credit"}
    assert "7710" in t2["reply"] and "Onaylıyor musunuz" in t2["reply"]
    # Nothing has happened in the core system yet.
    cards = (await mock_client.get("/payments/customers/C-1001/cards")).json()
    assert all(c["status"] == "active" for c in cards)

    t3 = await chat.say("Evet")
    assert t3["routing"]["source"] == "confirmation"
    assert t3["tool_calls"][0]["status"] == "ok"
    assert "bloke ettim" in t3["reply"]
    cards = {c["last4"]: c["status"] for c in (await mock_client.get("/payments/customers/C-1001/cards")).json()}
    assert cards == {"4821": "active", "7710": "blocked"}


async def test_block_card_asks_which_card_when_ambiguous(new_chat):
    chat, t2 = await _verified(new_chat, "C-1001", "12.04.1988", "kartımı kaybettim")
    assert t2["tool_calls"][1]["status"] == "error"
    assert "4821" in t2["reply"] and "7710" in t2["reply"]
    t3 = await chat.say("sonu 4821 olan")
    assert t3["awaiting_confirmation"]["summary"]["card_last4"] == "4821"


async def test_declining_cancels_without_touching_the_card(new_chat, mock_client):
    chat, _ = await _verified(new_chat, "C-1003", "27.01.1996", "kartımı bloke edin")
    t3 = await chat.say("Hayır, vazgeçtim")
    assert t3["tool_calls"] == [] and "yapmadım" in t3["reply"]
    tr = await chat.transcript()
    assert tr["state"]["pending_action"] is None
    cards = (await mock_client.get("/payments/customers/C-1003/cards")).json()
    assert cards[0]["status"] == "active"


async def test_unclear_answer_re_asks_instead_of_acting(new_chat):
    chat, _ = await _verified(new_chat, "C-1003", "27.01.1996", "kartımı bloke edin")
    t3 = await chat.say("hmm")
    assert t3["tool_calls"][0]["status"] == "needs_confirmation"
    assert t3["awaiting_confirmation"] is not None


async def test_changing_topic_expires_the_confirmation(new_chat):
    chat, _ = await _verified(new_chat, "C-1003", "27.01.1996", "kartımı bloke edin")
    await chat.say("Ödememin durumu ne?")
    t4 = await chat.say("evet")
    assert "block_card" not in tool_names(t4)


async def test_wrong_birth_date_twice_hands_off(new_chat):
    chat = await new_chat("tr")
    await chat.say("C-1003 kartımı bloke edin")
    t2 = await chat.say("01.01.2000")
    assert "eşleşmedi" in t2["reply"] and t2["verified"] is False
    t3 = await chat.say("02.02.2000")
    assert tool_names(t3) == ["verify_identity", "escalate_to_human"]
    assert t3["status"] == "escalated"


async def test_verification_does_not_survive_identity_switch(new_chat):
    chat, _ = await _verified(new_chat, "C-1003", "27.01.1996", "kartımı bloke edin")
    t = await chat.say("aslında müşteri numaram C-1001, kartım çalındı")
    assert t["customer_id"] == "C-1003"  # identity already set; the switch is not silently accepted
    assert t["verified"] is True


# ------------------------------------------------------------------ disputes + supervisor
async def test_small_dispute_needs_only_caller_confirmation(new_chat):
    chat, t2 = await _verified(
        new_chat, "C-1005", "19.06.1979", "I want to dispute TX-50017, I don't recognise it", lang="en"
    )
    assert t2["awaiting_confirmation"]["summary"]["transaction_id"] == "TX-50017"
    t3 = await chat.say("yes")
    assert t3["tool_calls"][0]["status"] == "ok" and "DSP-" in t3["reply"]
    assert t3["approval_ids"] == []


async def test_large_dispute_goes_to_supervisor_then_caller_is_told(new_chat, client, mock_client):
    chat, t2 = await _verified(new_chat, "C-1002", "03.09.1975", "TX-50015 işlemine itiraz ediyorum, ben yapmadım")
    assert "yetkili onayından sonra" in t2["reply"]
    t3 = await chat.say("evet onaylıyorum")
    assert t3["tool_calls"][0]["status"] == "pending_approval"
    [approval_id] = t3["approval_ids"]

    t4 = await chat.say("Talebim ne durumda?")
    assert approval_id in t4["reply"] and "bekliyor" in t4["reply"]

    queue = (await client.get("/v1/approvals")).json()
    assert [a["id"] for a in queue] == [approval_id]
    assert queue[0]["summary"]["amount"] == "12450.00"

    decided = await client.post(f"/v1/approvals/{approval_id}/decision", json={"decision": "approve", "reviewer": "sup.ayse"})
    assert decided.status_code == 200 and decided.json()["status"] == "executed"
    dispute_id = decided.json()["result"]["id"]
    again = await client.post(f"/v1/approvals/{approval_id}/decision", json={"decision": "reject", "reviewer": "sup.ayse"})
    assert again.status_code == 409

    t5 = await chat.say("Teşekkürler")
    assert t5["reply"].startswith(f"Bilginiz olsun: {approval_id}") and dispute_id in t5["reply"]
    t6 = await chat.say("Teşekkürler")
    assert approval_id not in t6["reply"]  # relayed exactly once

    tr = await chat.transcript()
    assert any(m["role"] == "event" and approval_id in m["content"] for m in tr["messages"])


async def test_supervisor_rejection_is_relayed(new_chat, client):
    chat, _ = await _verified(new_chat, "C-1005", "19.06.1979", "I want to dispute TX-50016", lang="en")
    t3 = await chat.say("yes, go ahead")
    [approval_id] = t3["approval_ids"]
    r = await client.post(f"/v1/approvals/{approval_id}/decision", json={"decision": "reject", "reviewer": "sup.john", "note": "merchant evidence"})
    assert r.json()["status"] == "rejected"
    t4 = await chat.say("thanks")
    assert "was not approved" in t4["reply"]


async def test_core_system_refuses_large_dispute_without_approver(mock_client):
    """Defence in depth: even a direct call that skips the agent's gate is refused."""
    r = await mock_client.post(
        "/payments/disputes", json={"customer_id": "C-1002", "transaction_id": "TX-50015", "reason": "test"}
    )
    assert r.status_code == 403 and r.json()["detail"]["code"] == "approval_required"


async def test_transfer_cannot_be_disputed(new_chat):
    chat, t2 = await _verified(new_chat, "C-1001", "12.04.1988", "TX-50001 işlemine itiraz etmek istiyorum")
    assert t2["tool_calls"][1]["status"] == "error"
    assert "kart harcaması olmadığı" in t2["reply"]


# ------------------------------------------------------------------ policy handoffs
@pytest.mark.parametrize(
    ("text", "category", "priority"),
    [
        ("Dolandırıldım, biri hesabımdan para çekti", "fraud_report", "urgent"),
        ("I want to close my account", "account_closure", "high"),
        ("Babam vefat etti", "bereavement", "high"),
    ],
)
async def test_sensitive_topics_go_straight_to_a_human(new_chat, text, category, priority):
    chat = await new_chat("tr")
    t = await chat.say(text)
    assert t["routing"]["sensitive_category"] == category
    assert t["routing"]["source"] == "policy"
    assert tool_names(t) == ["escalate_to_human"]
    assert t["tool_calls"][0]["arguments"]["priority"] == priority
    assert t["usage"]["input_tokens"] == 0  # no model in the loop
    assert t["status"] == "escalated"


# ------------------------------------------------------------------ grounded answers
async def test_policy_answer_cites_source(new_chat):
    chat = await new_chat("tr")
    t = await chat.say("Kredi kartı yıllık ücreti ne kadar?")
    assert tool_names(t) == ["search_policies"]
    assert "750 TL" in t["reply"] and "Kaynak: Kartlar" in t["reply"]
    assert t["citations"][0] == "cards#yillik-kart-ucreti"


async def test_policy_question_without_grounding_does_not_guess(new_chat):
    chat = await new_chat("en")
    t = await chat.say("What is the interest rate on gold savings accounts?")
    assert tool_names(t) == ["search_policies"]
    assert "couldn't find" in t["reply"] and t["citations"] == []


async def test_turn_reports_estimated_cost(new_chat):
    chat = await new_chat("en")
    t = await chat.say("EFT cut-off time?")
    assert t["usage"]["estimated"] is True and t["usage"]["estimated_cost_usd"] > 0


async def test_two_approvals_for_one_transaction_second_hears_the_real_reason(new_chat, client):
    """Race: two callers' requests for the same charge reach the queue before either runs.
    The second execution fails as dispute_exists, and that caller is told why (not "system issue")."""
    chats = []
    for _ in range(2):
        chat, _ = await _verified(new_chat, "C-1002", "03.09.1975", "TX-50015 işlemine itiraz ediyorum")
        assert (await chat.say("evet onaylıyorum"))["tool_calls"][0]["status"] == "pending_approval"
        chats.append(chat)
    pending = (await client.get("/v1/approvals")).json()
    results = [(await client.post(f"/v1/approvals/{a['id']}/decision", json={"decision": "approve", "reviewer": "sup"})).json() for a in pending]
    assert [r["status"] for r in results] == ["executed", "failed"] and results[1]["result"]["error"] == "dispute_exists"
    second = await chats[1].say("Teşekkürler")
    assert "zaten açık bir itiraz" in second["reply"] and "sistemsel" not in second["reply"]


async def test_open_dispute_is_caught_before_confirmation(new_chat):
    chat, t2 = await _verified(new_chat, "C-1005", "19.06.1979", "I want to dispute TX-50017", lang="en")
    assert (await chat.say("yes"))["tool_calls"][0]["status"] == "ok"
    again = await chat.say("Please dispute TX-50017 again")
    assert again["tool_calls"][0]["status"] == "error" and again["awaiting_confirmation"] is None
    assert "already an open dispute" in again["reply"]
