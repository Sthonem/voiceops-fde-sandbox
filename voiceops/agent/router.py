"""Intent router: runs before the LLM on every turn.

Responsibilities that must not depend on the model:
* label every turn with an intent + confidence (the basis of intent-accuracy evals),
* carry the conversation forward when the caller is answering our question
  ("0555 010 1001" after "what's your phone number?"),
* recognise "yes"/"no" to a pending sensitive action,
* flag sensitive topics (fraud, account closure, bereavement, legal threats) that
  policy says must go to a human immediately,
* flag requests that must be refused (investment advice, PIN/password disclosure,
  prompt injection, other customers' data), so the refusal is explicit and no
  tool or model ever runs for them.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from voiceops.agent.state import ChatMessage, ConversationState, Language
from voiceops.agent.text import analyze

SENSITIVE_PRIORITY = {"fraud_report": "urgent", "account_closure": "high", "bereavement": "high", "legal_threat": "high"}


@dataclass
class RoutingDecision:
    intent: str
    confidence: float
    language: Language
    slots: dict[str, Any] = field(default_factory=dict)
    source: Literal["keywords", "carry_over", "confirmation", "policy"] = "keywords"
    sensitive_category: str | None = None
    mandatory_escalation: bool = False
    refusal_category: str | None = None
    confirmation: Literal["confirmed", "declined"] | None = None
    scores: dict[str, float] = field(default_factory=dict)

    def to_trace(self) -> dict[str, Any]:
        def safe(v: Any) -> Any:
            return v.isoformat() if isinstance(v, (date, datetime)) else v

        d = asdict(self)
        d["slots"] = {k: safe(v) for k, v in self.slots.items() if v not in (None, False)}
        return d


class IntentRouter:
    def __init__(self, tz: ZoneInfo, clock: Callable[[], datetime] | None = None):
        self.tz = tz
        self.clock = clock or (lambda: datetime.now(tz))

    def route(self, text: str, *, state: ConversationState, history: list[ChatMessage]) -> RoutingDecision:
        a = analyze(text, now=self.clock(), default_language=state.language)
        d = RoutingDecision(a.intent, a.confidence, a.language, a.slots, scores=a.scores)

        # 1. Policy: sensitive topics always go to a human, whatever else was said.
        if a.sensitive_category:
            d.intent, d.confidence, d.source = "human_escalation", 1.0, "policy"
            d.sensitive_category, d.mandatory_escalation = a.sensitive_category, True
            return d

        # 2. Policy: understood, out of scope or not allowed -> explicit refusal.
        if a.refusal_category:
            d.intent, d.confidence, d.source = "out_of_scope", 1.0, "policy"
            d.refusal_category = a.refusal_category
            return d

        # 3. Answer to "shall I go ahead?" for a pending sensitive action.
        pending = state.pending_action
        if pending and pending.get("kind") == "confirm" and not pending.get("confirmed"):
            same_or_none = a.intent in ("unknown", "greeting", pending["intent"])
            if a.slots["affirmative"] and same_or_none:
                d.intent, d.confidence, d.source, d.confirmation = pending["intent"], 0.95, "confirmation", "confirmed"
                return d
            if a.slots["negative"]:
                d.intent, d.confidence, d.source, d.confirmation = pending["intent"], 0.95, "confirmation", "declined"
                return d
            if a.intent in ("unknown", "identify", "greeting"):
                # Neither yes nor no ("the one ending 4821"): stay on the action and ask again.
                d.intent, d.confidence, d.source = pending["intent"], 0.7, "carry_over"
                return d

        # 4. Carry-over: the caller is answering the question we just asked.
        finals = [m for m in history if m.role == "assistant" and not m.tool_calls]
        prev: dict[str, Any] = finals[-1].meta if finals else {}
        awaiting, carried = prev.get("awaiting"), prev.get("intent")
        weak = a.intent in ("unknown", "identify", "greeting")
        new_intent = None
        if prev.get("offer") == "escalation" and a.intent == "unknown" and a.slots["affirmative"]:
            new_intent = "human_escalation"
        elif awaiting == "identity" and a.intent in ("identify", "unknown") and carried:
            new_intent = carried
        elif awaiting == "callback_time" and weak:
            new_intent = "schedule_callback"
        elif awaiting in ("dob", "card", "transaction") and weak and carried:
            new_intent = carried
        if new_intent:
            d.intent, d.source, d.confidence = new_intent, "carry_over", max(d.confidence, 0.85)
        return d
