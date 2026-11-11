"""Streaming: the orchestrator's event contract (what the voice channel speaks) and
the Claude provider's streaming path, tested offline with a fake SSE transport."""

import json

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from voiceops.agent.llm.anthropic import RAW_KEY, AnthropicProvider
from voiceops.agent.llm.base import LLMProvider, LLMResult, LLMUsage
from voiceops.agent.llm.mock import MockLLMProvider
from voiceops.agent.orchestrator import Orchestrator
from voiceops.agent.state import ChatMessage, ConversationState, ToolCallRequest
from voiceops.agent.tools import build_default_registry
from voiceops.speech.text import SentenceChunker

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-stream"}


# --------------------------------------------------------------------------- fake SSE API
def stream_body(blocks: list[dict], stop_reason: str = "end_turn") -> bytes:
    events: list[dict] = [
        {"type": "message_start", "message": {"id": "msg_s", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 900, "output_tokens": 1}}}  # fmt: skip
    ]
    for i, b in enumerate(blocks):
        if b["type"] == "thinking":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            events.append({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": b["signature"]}})
        elif b["type"] == "text":
            events.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            for chunk in b["chunks"]:
                events.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": chunk}})
        elif b["type"] == "tool_use":
            start = {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}
            events.append({"type": "content_block_start", "index": i, "content_block": start})
            for frag in b["json"]:
                events.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": frag}})
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 40}})
    events.append({"type": "message_stop"})
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


class StreamingAPI:
    def __init__(self, bodies: list[bytes]):
        self.bodies = list(bodies)
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=self.bodies.pop(0))

    def provider(self) -> AnthropicProvider:
        client = AsyncAnthropic(
            api_key="test-key-not-real",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler)),
        )
        return AnthropicProvider(client=client)


class Collector:
    def __init__(self):
        self.events: list[dict] = []

    async def __call__(self, event: dict) -> None:
        self.events.append(event)

    def spoken(self) -> list[str]:
        """Sentences as the voice channel would queue them for TTS."""
        chunker, out = SentenceChunker(), []
        for e in self.events:
            if e["type"] == "text":
                out += chunker.feed(e["text"])
            elif e["type"] == "text_break":
                out += chunker.flush()
        return out


# --------------------------------------------------------------------------- provider
async def test_stream_forwards_text_deltas_and_sends_eager_tools():
    chunks = ["Merhaba ", "Ayşe Hanım. Size ", "nasıl yardımcı olabilirim?"]
    api = StreamingAPI([stream_body([THINKING, {"type": "text", "chunks": chunks}])])
    deltas: list[str] = []

    async def on_text(d: str) -> None:
        deltas.append(d)

    res = await api.provider().complete(
        system="SYS", messages=[ChatMessage("user", "Merhaba")], tools=build_default_registry().specs(),
        state=ConversationState(), on_text=on_text,
    )  # fmt: skip
    assert deltas == chunks
    assert res.text == "Merhaba Ayşe Hanım. Size nasıl yardımcı olabilirim?"
    body = api.requests[0]
    assert body["stream"] is True
    assert all(t["eager_input_streaming"] is True and t["strict"] is True for t in body["tools"])
    assert res.meta[RAW_KEY][0] == THINKING  # signature kept for replay
    assert res.usage.output_tokens == 40


async def test_unparseable_tool_json_is_reissued_when_nothing_was_spoken():
    bad = stream_body([{"type": "tool_use", "id": "t1", "name": "lookup_customer", "json": ['{"customer_id": ]']}], "tool_use")
    good = stream_body([{"type": "tool_use", "id": "t2", "name": "lookup_customer", "json": ['{"customer_', 'id": "C-1001"}']}], "tool_use")
    api = StreamingAPI([bad, good])

    async def on_text(_: str) -> None:
        return None

    res = await api.provider().complete(
        system="S", messages=[ChatMessage("user", "C-1001")], tools=build_default_registry().specs(),
        state=ConversationState(), on_text=on_text,
    )  # fmt: skip
    assert len(api.requests) == 2
    assert res.tool_calls == [ToolCallRequest("t2", "lookup_customer", {"customer_id": "C-1001"})]


async def test_streamed_turn_speaks_preamble_then_answer(integrations):
    api = StreamingAPI(
        [
            stream_body(
                [{"type": "text", "chunks": ["Hemen ", "bakıyorum."]},
                 {"type": "tool_use", "id": "t1", "name": "lookup_customer", "json": ['{"customer_id"', ': "C-1001"}']}],
                "tool_use",
            ),
            stream_body([{"type": "text", "chunks": ["Teşekkürler Ayşe Hanım. ", "Kaydınızı buldum."]}]),
        ]
    )  # fmt: skip
    orch = Orchestrator(llm=api.provider(), registry=build_default_registry(), integrations=integrations)
    sink = Collector()
    res = await orch.run_turn(conversation_id="c", state=ConversationState(), history=[], user_text="C-1001 benim", on_event=sink)
    assert sink.spoken() == ["Hemen bakıyorum.", "Teşekkürler Ayşe Hanım.", "Kaydınızı buldum."]
    assert res.reply == "Hemen bakıyorum. Teşekkürler Ayşe Hanım. Kaydınızı buldum."
    tools = [e for e in sink.events if e["type"] == "tool"]
    assert [(e["phase"], e["name"]) for e in tools] == [("start", "lookup_customer"), ("end", "lookup_customer")]
    assert res.state.customer_id == "C-1001"
    assert "phone call" not in api.requests[0]["messages"][1]["content"]  # text channel by default


# --------------------------------------------------------------------------- orchestrator contract
def _mock_orch(integrations) -> Orchestrator:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/Istanbul")
    return Orchestrator(llm=MockLLMProvider(tz), registry=build_default_registry(), integrations=integrations, tz=tz)


async def test_events_add_up_to_the_reply_with_notices_first(integrations):
    state = ConversationState(
        customer_id="C-1002", customer_first_name="Mehmet",
        notifications=[{"approval_id": "APR-1", "decision": "approved", "reference": "DSP-000001"}],
    )  # fmt: skip
    sink = Collector()
    res = await _mock_orch(integrations).run_turn(conversation_id="c", state=state, history=[], user_text="EFT kesim saati kaç?", on_event=sink)
    spoken = sink.spoken()
    assert " ".join(spoken) == res.reply
    assert "APR-1" in spoken[0]  # the supervisor decision is heard before the answer
    assert sum(e["type"] == "text" for e in sink.events) > len(spoken)  # the mock streams word groups


async def test_template_replies_are_emitted_too(integrations):
    sink = Collector()
    res = await _mock_orch(integrations).run_turn(
        conversation_id="c", state=ConversationState(), history=[], user_text="Dolandırıldım, biri hesabımdan para çekti", on_event=sink
    )
    assert res.intent == "human_escalation" and " ".join(sink.spoken()) == res.reply


async def test_non_streaming_provider_text_is_emitted_and_kept(integrations):
    class Preamble(LLMProvider):
        name, model = "preamble", "p1"

        def __init__(self):
            self.n = 0

        async def complete(self, **kw):
            self.n += 1
            if self.n == 1:
                call = ToolCallRequest("c1", "lookup_customer", {"customer_id": "C-1001"})
                return LLMResult("Bir bakayım.", [call], self.model, LLMUsage(1, 1), "tool_use")
            return LLMResult("Buldum.", [], self.model, LLMUsage(1, 1), "end_turn")

    sink = Collector()
    orch = Orchestrator(llm=Preamble(), registry=build_default_registry(), integrations=integrations)
    res = await orch.run_turn(conversation_id="c", state=ConversationState(), history=[], user_text="C-1001", on_event=sink)
    assert sink.spoken() == ["Bir bakayım.", "Buldum."] and res.reply == "Bir bakayım. Buldum."


async def test_a_broken_listener_does_not_break_the_turn(integrations):
    async def broken(_: dict) -> None:
        raise ConnectionResetError("socket closed")

    res = await _mock_orch(integrations).run_turn(
        conversation_id="c", state=ConversationState(), history=[], user_text="Müşteri numaram C-1001", on_event=broken
    )
    assert res.state.customer_id == "C-1001" and "Ayşe" in res.reply
