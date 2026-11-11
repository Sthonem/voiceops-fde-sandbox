"""Claude provider, tested offline: the SDK runs for real, only the HTTP transport is fake."""

import json

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from voiceops.agent.llm.anthropic import RAW_KEY, AnthropicProvider, to_api_messages
from voiceops.agent.orchestrator import Orchestrator
from voiceops.agent.state import ChatMessage, ConversationState, ToolCallRequest
from voiceops.agent.tools import build_default_registry


def _response(content, stop_reason="end_turn", model="claude-opus-5-5", **usage):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 1000),
            "output_tokens": usage.get("output_tokens", 50),
            "cache_read_input_tokens": usage.get("cache_read", 0),
            "cache_creation_input_tokens": 0,
        },
    }


THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx2.Response(200, json=self.responses.pop(0))

    def provider(self) -> AnthropicProvider:
        client = AsyncAnthropic(
            api_key="test-key-not-real",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler)),
        )
        return AnthropicProvider(client=client)


def test_message_translation_groups_tool_results_and_replays_raw_blocks():
    raw = [THINKING, {"type": "tool_use", "id": "t1", "name": "lookup_customer", "input": {"customer_id": "C-1001"}}]
    history = [
        ChatMessage("user", "C-1001 kartım çalındı", meta={"context": "Session state: ..."}),
        ChatMessage(
            "assistant",
            tool_calls=[ToolCallRequest("t1", "lookup_customer", {"customer_id": "C-1001"})],
            meta={RAW_KEY: raw},
        ),
        ChatMessage("tool", '{"customer_id": "C-1001"}', tool_call_id="t1", tool_name="lookup_customer"),
        ChatMessage("tool", '{"error": "x"}', tool_call_id="t2", tool_name="other", is_error=True),
        ChatMessage("assistant", "Doğum tarihiniz?"),
        ChatMessage("event", "Supervisor approved APR-1"),
        ChatMessage("user", "12.04.1988"),
    ]
    out = to_api_messages(history)
    assert [m["role"] for m in out] == ["user", "system", "assistant", "user", "assistant", "user"]
    assert out[1] == {"role": "system", "content": "Session state: ...", "clear_at": "next_user_message"}
    assert out[2]["content"] == raw  # thinking + tool_use replayed byte-for-byte
    assert [b["tool_use_id"] for b in out[3]["content"]] == ["t1", "t2"]
    assert out[3]["content"][1]["is_error"] is True
    assert out[4]["content"] == [{"type": "text", "text": "Doğum tarihiniz?"}]


async def test_request_shape_and_tool_call_parsing():
    api = FakeAPI(
        [_response([THINKING, {"type": "tool_use", "id": "toolu_1", "name": "lookup_customer", "input": {"phone": "05550101001"}}], "tool_use", cache_read=800)]
    )
    provider = api.provider()
    res = await provider.complete(
        system="SYS",
        messages=[ChatMessage("user", "0555 010 1001", meta={"context": "state"})],
        tools=build_default_registry().specs(),
        state=ConversationState(),
    )
    req = api.requests[0]
    body = req["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "low"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] is True and t["input_schema"]["additionalProperties"] is False for t in body["tools"])
    assert "thinking" not in body  # Opus 5.5: always adaptive; disabling it is a 400
    assert "stream" not in body and all("eager_input_streaming" not in t for t in body["tools"])
    assert "server-side-fallback-2026-07-01" in req["headers"]["anthropic-beta"]
    assert "mid-conversation-system-clear-at-2026-08-21" in req["headers"]["anthropic-beta"]
    assert res.stop_reason == "tool_use"
    assert res.tool_calls[0].name == "lookup_customer" and res.tool_calls[0].arguments == {"phone": "05550101001"}
    assert res.usage.cache_read_tokens == 800
    assert res.meta[RAW_KEY][0]["signature"] == "sig-abc"


async def test_full_turn_with_claude_and_real_tools(integrations):
    api = FakeAPI(
        [
            _response([THINKING, {"type": "tool_use", "id": "toolu_1", "name": "lookup_customer", "input": {"customer_id": "C-1001"}}], "tool_use"),
            _response([THINKING, {"type": "tool_use", "id": "toolu_2", "name": "get_transaction_status", "input": {"limit": 3}}], "tool_use"),
            _response([THINKING, {"type": "text", "text": "Ayşe Hanım, 2.500 TL'lik EFT'niz yarın sabah işlenecek."}]),
        ]
    )
    orch = Orchestrator(llm=api.provider(), registry=build_default_registry(), integrations=integrations)
    res = await orch.run_turn(
        conversation_id="conv-claude", state=ConversationState(), history=[], user_text="C-1001, EFT'm nerede?"
    )
    assert res.reply.startswith("Ayşe Hanım")
    assert res.state.customer_id == "C-1001"
    assert res.intent == "payment_status"  # from the router, provider-independent
    assert res.usage.input_tokens == 3000 and res.model == "claude-opus-5-5"
    third = api.requests[2]["body"]["messages"]
    roles = [m["role"] for m in third]
    assert roles == ["user", "system", "assistant", "user", "assistant", "user"]
    assert "caller identified: no" in third[1]["content"]  # state at the start of the turn
    assert third[2]["content"][0] == THINKING  # thinking block replayed unchanged
    assert third[5]["content"][0]["type"] == "tool_result"


async def test_guards_hold_against_claude(integrations):
    """The model skips verification and tries to block a card directly: the registry refuses."""
    api = FakeAPI(
        [
            _response([{"type": "tool_use", "id": "t1", "name": "block_card", "input": {"card_last4": "7710"}}], "tool_use"),
            _response([{"type": "text", "text": "Önce doğum tarihinizi alabilir miyim?"}]),
        ]
    )
    orch = Orchestrator(llm=api.provider(), registry=build_default_registry(), integrations=integrations)
    state = ConversationState(customer_id="C-1001", customer_first_name="Ayşe")
    res = await orch.run_turn(conversation_id="c", state=state, history=[], user_text="Kartımı hemen bloke et")
    tool = next(t for t in res.traces if t.kind == "tool")
    assert tool.status == "blocked" and tool.output["error"] == "verification_required"
    sent_result = api.requests[1]["body"]["messages"][-1]["content"][0]
    assert sent_result["is_error"] is True


async def test_refusal_becomes_safe_reply(integrations):
    api = FakeAPI([_response([{"type": "text", "text": ""}], "refusal")])
    orch = Orchestrator(llm=api.provider(), registry=build_default_registry(), integrations=integrations)
    res = await orch.run_turn(conversation_id="c", state=ConversationState(), history=[], user_text="Merhaba")
    assert "ulaşamıyorum" in res.reply
    assert res.new_messages[-1].meta["offer"] == "escalation"


@pytest.mark.parametrize("provider_name", ["mock", "anthropic"])
def test_factory_builds_both_providers(provider_name, monkeypatch):
    from voiceops.agent.llm import build_llm_provider
    from voiceops.config import Settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")  # constructor only; no request is made
    p = build_llm_provider(Settings(llm_provider=provider_name))
    assert p.name == provider_name
