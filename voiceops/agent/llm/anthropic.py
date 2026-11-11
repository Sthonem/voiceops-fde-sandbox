"""Claude provider (Anthropic Messages API, tool use).

One `complete()` call = one Messages API request; the orchestrator owns the tool
loop, guards and persistence, so this class is only translation:

* provider-neutral `ChatMessage` history  -> Messages API `messages`
* tool specs                              -> `tools` with `strict: true`
* per-turn session state                  -> a mid-conversation `role: "system"`
  message after each caller turn, with `clear_at: "next_user_message"` so it is
  shown for that turn only but stays in the transcript unchanged (append-only
  history keeps prompt caching and thinking-block replay valid)
* response                                -> `LLMResult`; the raw content blocks
  (thinking, fallback, tool_use, text) are kept in `meta["anthropic_content"]` and
  replayed verbatim on later requests.

Streaming (voice channel): when the orchestrator passes `on_text`, the request is
streamed and text deltas are forwarded as they arrive, so the first sentence can
be synthesised while the model is still writing. Streamed requests send tools
with `eager_input_streaming`; the registry validates every tool input against its
schema, and the orchestrator never runs tools from a `refusal` or `max_tokens` stop.

Defaults: Claude Opus 5.5, effort `low` (routine, latency-sensitive voice turns),
server-side refusal fallbacks enabled (`fallbacks="default"`).
"""

import json
from typing import Any

import anthropic

from voiceops.agent.llm.base import LLMProvider, LLMResult, LLMUsage, TextSink, ToolSpec
from voiceops.agent.router import RoutingDecision
from voiceops.agent.state import ChatMessage, ConversationState, ToolCallRequest

BETAS = ["server-side-fallback-2026-07-01", "mid-conversation-system-clear-at-2026-08-21"]
RAW_KEY = "anthropic_content"


MAX_STREAM_REISSUES = 2


def to_api_tools(tools: list[ToolSpec], eager_input_streaming: bool = False) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        tool: dict[str, Any] = {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
        if eager_input_streaming:
            tool["eager_input_streaming"] = True
        out.append(tool)
    return out


def to_api_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Translate the transcript. Consecutive tool results are grouped into one user
    message (one assistant turn's parallel calls must be answered together)."""
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        if m.role == "tool":
            block: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if m.is_error:
                block["is_error"] = True
            pending_results.append(block)
            continue
        flush()
        if m.role == "event":
            continue  # out-of-band events reach the model through the session-state note
        if m.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.content}]})
            if note := m.meta.get("context"):
                out.append({"role": "system", "content": note, "clear_at": "next_user_message"})
        elif m.role == "assistant":
            if raw := m.meta.get(RAW_KEY):
                out.append({"role": "assistant", "content": raw})
                continue
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "…"}]})
    flush()
    return out


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",
        max_tokens: int = 16000,
        timeout_s: float = 30.0,
        client: anthropic.AsyncAnthropic | None = None,
        eager_tool_streaming: bool = True,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.eager_tool_streaming = eager_tool_streaming
        # Credentials resolve from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
        self.client = client or anthropic.AsyncAnthropic(timeout=timeout_s, max_retries=2)

    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        state: ConversationState,
        routing: RoutingDecision | None = None,
        on_text: TextSink | None = None,
    ) -> LLMResult:
        streaming = on_text is not None
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system}],
            "messages": to_api_messages(messages),
            "tools": to_api_tools(tools, eager_input_streaming=streaming and self.eager_tool_streaming),
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
            "fallbacks": "default",
            "betas": BETAS,
        }
        if on_text is None:
            resp = await self.client.beta.messages.create(**params)
        else:
            resp = await self._stream(params, on_text)
        return self._parse(resp, routing)

    async def _stream(self, params: dict[str, Any], on_text: TextSink) -> Any:
        for attempt in range(MAX_STREAM_REISSUES + 1):
            emitted = False
            try:
                async with self.client.beta.messages.stream(**params) as stream:
                    async for event in stream:
                        if event.type == "text" and event.text:
                            emitted = True
                            await on_text(event.text)
                    return await stream.get_final_message()
            except ValueError:
                # Tool-input JSON the SDK could not parse at all (eager input streaming).
                # No tool_use id exists to answer, so re-issue the request, unless the
                # caller already heard part of this answer. API errors are not ValueError.
                if emitted or attempt == MAX_STREAM_REISSUES:
                    raise
        raise AssertionError("unreachable")

    def _parse(self, resp: Any, routing: RoutingDecision | None) -> LLMResult:
        raw = [b.to_dict(exclude_none=True) for b in resp.content]
        text = "".join(b.text for b in resp.content if b.type == "text").strip() or None
        calls = [
            ToolCallRequest(b.id, b.name, b.input if isinstance(b.input, dict) else json.loads(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        u = resp.usage
        usage = LLMUsage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
        )
        meta: dict[str, Any] = {RAW_KEY: raw, "served_by": resp.model}
        if routing is not None:
            meta["intent"] = routing.intent
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            meta["refusal_category"] = getattr(details, "category", None)
        return LLMResult(
            text=text,
            tool_calls=calls,
            model=resp.model,
            usage=usage,
            stop_reason=resp.stop_reason or "end_turn",
            meta=meta,
        )
