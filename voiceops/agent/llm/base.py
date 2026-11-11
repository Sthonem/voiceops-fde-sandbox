"""Provider-neutral LLM interface.

The orchestrator only depends on this module, so the mock provider (offline, CI,
eval baseline) and real providers (Claude in M2) are interchangeable.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from voiceops.agent.state import ChatMessage, ConversationState, ToolCallRequest


# Receives text deltas as the model generates them (voice: speak before the turn ends).
TextSink = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False  # True when tokens are approximated (mock provider)

    def add(self, other: "LLMUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.estimated = self.estimated or other.estimated


@dataclass
class LLMResult:
    text: str | None
    tool_calls: list[ToolCallRequest]
    model: str
    usage: LLMUsage
    stop_reason: str  # "tool_use" | "end_turn" | "refusal" | ...
    # Provider annotations persisted on the assistant message, e.g. the intent the
    # turn was routed to and which slot the agent is waiting for.
    meta: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        state: ConversationState,
        routing: "RoutingDecision | None" = None,
        on_text: TextSink | None = None,
    ) -> LLMResult:
        """One model step. When `on_text` is given, providers that can stream call
        it with text deltas as they arrive; the returned `LLMResult.text` is still
        the complete text of the step."""


if TYPE_CHECKING:
    from voiceops.agent.router import RoutingDecision
