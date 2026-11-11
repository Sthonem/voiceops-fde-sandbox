"""Shared agent types: conversation state, transcript messages, tool calls."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["tr", "en"]

# Intents the agent understands. "identify" = caller only gave an identifier.
INTENTS = (
    "greeting",
    "identify",
    "payment_status",
    "create_ticket",
    "schedule_callback",
    "policy_question",
    "ticket_status",
    "card_block",
    "dispute_transaction",
    "human_escalation",
    "goodbye",
    "out_of_scope",  # understood and refused (investment advice, PINs, prompt injection, ...)
    "unknown",
)
INTENTS_REQUIRING_CUSTOMER = {
    "payment_status", "ticket_status", "create_ticket", "schedule_callback", "card_block", "dispute_transaction",
}  # fmt: skip
# A customer ID or phone number identifies the caller; it does not authenticate them.
# Reading account data and sensitive actions need step-up verification (date of
# birth) on top of identification. Callbacks (to the registered number) and new
# tickets disclose nothing, so identification is enough for them.
INTENTS_REQUIRING_VERIFICATION = {"payment_status", "ticket_status", "card_block", "dispute_transaction"}


class ConversationState(BaseModel):
    """Working memory persisted between turns (JSON column on Conversation).

    Identity lives here, not in the model's context: customer-scoped tools take
    the customer id from this state, never from LLM-generated arguments.
    """

    language: Language = "tr"
    status: Literal["active", "escalated", "closed"] = "active"
    customer_id: str | None = None
    customer_first_name: str | None = None
    verified: bool = False  # step-up verification for sensitive actions
    handoff_id: str | None = None
    # A sensitive action waiting for the caller's "yes" (kind=confirm) or a
    # supervisor's decision (kind=approval). See voiceops/agent/tools.py.
    pending_action: dict[str, Any] | None = None
    # Out-of-band updates (e.g. supervisor decisions) to relay on the next turn.
    notifications: list[dict[str, Any]] = Field(default_factory=list)
    turn: int = 0


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ChatMessage:
    """Provider-neutral transcript entry. Providers translate to their own wire format.

    role "event" records something that happened outside the conversation (a
    supervisor decision); providers do not replay it as a chat turn.
    """

    role: Literal["user", "assistant", "tool", "event"]
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
