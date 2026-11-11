"""Deterministic stand-in for an LLM with tool calling.

It honours the same contract as a real provider: read the transcript, either
request one tool call or return a final answer. That lets the orchestrator,
guards, persistence, voice pipeline and eval harness be built and tested without
API keys, and gives the eval a rules baseline to compare a real model against.

Intent comes from the orchestrator's IntentRouter; this class decides *which
tool next* and *how to phrase the answer*.
"""

import json
import re
import uuid
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from voiceops.agent import templates as tpl
from voiceops.agent.llm.base import LLMProvider, LLMResult, LLMUsage, TextSink, ToolSpec
from voiceops.agent.router import IntentRouter, RoutingDecision
from voiceops.agent.state import (
    INTENTS_REQUIRING_CUSTOMER,
    INTENTS_REQUIRING_VERIFICATION,
    ChatMessage,
    ConversationState,
    Language,
    ToolCallRequest,
)
from voiceops.agent.text import extract_priority, extract_ticket_category, fold
from voiceops.rag.store import Passage, best_sentences

THANKS: dict[Language, str] = {"tr": "Teşekkürler {name}.", "en": "Thanks {name}."}
MAX_IDENTITY_ATTEMPTS = 2  # failed lookups before handing off to a human
MAX_DOB_ATTEMPTS = 2  # failed date-of-birth checks before handing off
MAX_UNKNOWN_STREAK = 2  # consecutive misunderstandings before handing off
# Slots worth remembering while we ask follow-up questions (identity, DOB, ...).
CARRIED_SLOTS = ("transaction_id", "ticket_id", "card_last4", "card_kind", "reason_text", "card_reason")
# Reads ask for the date of birth with their own wording ("before I share account details").
READ_INTENTS = {"payment_status", "ticket_status"}
_CONFIRM_WORDS = re.compile(r"\b(onayliyorum|onayla|onaylarim|onay veriyorum|i confirm|confirmed|confirm)\b")


class MockLLMProvider(LLMProvider):
    name = "mock"
    model = "mock-rules-v1"

    def __init__(self, tz: ZoneInfo, clock: Callable[[], datetime] | None = None):
        self.tz = tz
        self.clock = clock or (lambda: datetime.now(tz))

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
        user_idx = max(i for i, m in enumerate(messages) if m.role == "user")
        user_text = messages[user_idx].content
        results = [m for m in messages[user_idx + 1 :] if m.role == "tool"]
        finals = [m for m in messages[:user_idx] if m.role == "assistant" and not m.tool_calls]
        prev: dict[str, Any] = finals[-1].meta if finals else {}
        r = routing or IntentRouter(self.tz, self.clock).route(user_text, state=state, history=messages[:user_idx])
        intent = r.intent

        unknown_streak = 0
        for m in reversed(finals):
            if m.meta.get("intent") != "unknown":
                break
            unknown_streak += 1
        attempts = {
            "identity": int(prev.get("identity_attempts", 0)) if prev.get("awaiting") == "identity" else 0,
            "dob": int(prev.get("dob_attempts", 0)) if prev.get("awaiting") == "dob" else 0,
        }
        slots = self._slots(r, user_text, prev)

        meta: dict[str, Any] = {"intent": intent, "confidence": r.confidence}
        if intent == "out_of_scope" and r.refusal_category in tpl.REFUSALS:
            # The orchestrator refuses these before any model runs; this keeps the
            # provider correct when it is driven on its own.
            text = tpl.REFUSALS[r.refusal_category][state.language]
            return self._result(text, [], {**meta, "refused": r.refusal_category}, system, messages, tools)
        call = self._next_call(intent, r, slots, state, results, unknown_streak, attempts, user_text)
        if call is not None:
            return self._result(None, [call], meta, system, messages, tools)
        text, extra = self._render(intent, slots, state, results, attempts, user_text)
        if extra.get("awaiting"):
            extra["slots"] = {k: slots[k] for k in CARRIED_SLOTS if slots.get(k)}
        if on_text is not None and text:
            # Imitate token streaming (a few words per delta) so the voice pipeline's
            # sentence chunking runs the same way it does with a real model.
            words = re.findall(r"\S+\s*", text)
            for i in range(0, len(words), 3):
                await on_text("".join(words[i : i + 3]))
        return self._result(text, [], {**meta, **extra}, system, messages, tools)

    # ------------------------------------------------------------------ slots
    @staticmethod
    def _slots(r: RoutingDecision, user_text: str, prev: dict[str, Any]) -> dict[str, Any]:
        t = fold(user_text)
        slots = {**prev.get("slots", {}), **{k: v for k, v in r.slots.items() if v not in (None, False)}}
        if re.search(r"\b(kredi|credit)\b", t):
            slots["card_kind"] = "credit"
        elif re.search(r"\b(banka karti|debit|bankamatik)\b", t):
            slots["card_kind"] = "debit"
        if re.search(r"\b(calindi|calinti|stolen)\b", t):
            slots["card_reason"] = "stolen"
        elif re.search(r"\b(kayip|kayboldu|kaybettim|lost)\b", t):
            slots.setdefault("card_reason", "lost")
        if r.intent == "dispute_transaction" and r.source == "keywords":
            slots.setdefault("reason_text", user_text.strip()[:200])
        return slots

    # ------------------------------------------------------------------ policy
    def _next_call(
        self,
        intent: str,
        r: RoutingDecision,
        slots: dict[str, Any],
        state: ConversationState,
        results: list[ChatMessage],
        unknown_streak: int,
        attempts: dict[str, int],
        user_text: str,
    ) -> ToolCallRequest | None:
        done = {m.tool_name for m in results}
        last = results[-1] if results else None

        def call(name: str, **args: Any) -> ToolCallRequest | None:
            return None if name in done else ToolCallRequest(f"call_{uuid.uuid4().hex[:12]}", name, args)

        if last is not None and last.is_error:
            if last.tool_name == "lookup_customer" and attempts["identity"] >= MAX_IDENTITY_ATTEMPTS:
                return call("escalate_to_human", reason="identity_not_established")
            if last.tool_name == "verify_identity" and attempts["dob"] + 1 >= MAX_DOB_ATTEMPTS:
                return call("escalate_to_human", reason="verification_failed", priority="high")
            return None
        if last is not None and last.tool_name in ("block_card", "dispute_transaction"):
            return None  # confirmation / approval / done: time to talk to the caller
        if intent == "human_escalation":
            return call("escalate_to_human", reason="caller_requested_human", priority=extract_priority(user_text))
        if intent == "unknown" and unknown_streak >= MAX_UNKNOWN_STREAK:
            return call("escalate_to_human", reason="repeated_low_confidence")
        if (intent in INTENTS_REQUIRING_CUSTOMER or intent == "identify") and state.customer_id is None:
            if slots.get("customer_id"):
                return call("lookup_customer", customer_id=slots["customer_id"])
            if slots.get("phone"):
                return call("lookup_customer", phone=slots["phone"])
            return None
        if intent in INTENTS_REQUIRING_VERIFICATION and not state.verified:
            dob: date | None = r.slots.get("date_of_birth")
            return call("verify_identity", date_of_birth=dob.isoformat()) if dob else None
        if intent == "policy_question":
            return call("search_policies", query=user_text)
        if intent == "payment_status":
            args = {"transaction_id": slots["transaction_id"]} if slots.get("transaction_id") else {"limit": 3}
            return call("get_transaction_status", **args)
        if intent == "ticket_status":
            return call("get_tickets", **({"ticket_id": slots["ticket_id"]} if slots.get("ticket_id") else {}))
        if intent == "create_ticket":
            return call(
                "create_ticket",
                category=extract_ticket_category(user_text),
                priority=extract_priority(user_text),
                summary=user_text.strip()[:500],
            )
        if intent == "schedule_callback" and slots.get("callback_time"):
            return call("schedule_callback", preferred_time=slots["callback_time"].isoformat(), reason=user_text[:200])

        pending = state.pending_action or {}
        resume = r.confirmation == "confirmed" and pending.get("intent") == intent
        if intent == "card_block":
            if resume:
                return call("block_card", **pending.get("requested", pending["arguments"]))
            args = {k: slots[k] for k in ("card_last4", "card_kind") if slots.get(k)}
            return call("block_card", **args, **({"reason": slots["card_reason"]} if slots.get("card_reason") else {}))
        if intent == "dispute_transaction":
            if resume:
                return call("dispute_transaction", **pending.get("requested", pending["arguments"]))
            if slots.get("transaction_id"):
                reason = slots.get("reason_text") or user_text
                return call("dispute_transaction", transaction_id=slots["transaction_id"], reason=reason[:200])
        return None

    # ------------------------------------------------------------------ rendering
    def _render(
        self,
        intent: str,
        slots: dict[str, Any],
        state: ConversationState,
        results: list[ChatMessage],
        attempts: dict[str, int],
        user_text: str,
    ) -> tuple[str, dict[str, Any]]:
        lang = state.language
        out = {m.tool_name: json.loads(m.content) for m in results}
        last = results[-1] if results else None

        if "escalate_to_human" in out and not out["escalate_to_human"].get("error"):
            ho = out["escalate_to_human"]
            return tpl.render(
                "handoff", lang, position=ho["queue_position"], wait=ho["estimated_wait_minutes"], handoff_id=ho["id"]
            ), {}

        if last is not None and last.is_error:
            return self._render_error(last.tool_name or "", out[last.tool_name], lang, attempts)

        prefix = ""
        if "lookup_customer" in out and intent != "identify":
            prefix = THANKS[lang].format(name=state.customer_first_name) + " "

        if intent == "greeting":
            return tpl.render("greeting", lang), {}
        if intent == "goodbye":
            return tpl.render("goodbye", lang), {}
        pending = state.pending_action or {}
        if (
            intent == "unknown"
            and pending.get("kind") == "approval"
            and re.search(r"\b(talep|basvuru|itiraz|request|dispute|durum|status)", fold(user_text))
        ):
            return tpl.render("approval_waiting", lang, approval_id=pending["approval_id"]), {}
        if intent in INTENTS_REQUIRING_CUSTOMER and state.customer_id is None:
            return tpl.render("ask_identity", lang), {
                "awaiting": "identity",
                "identity_attempts": attempts["identity"] + 1,
            }
        if intent == "identify":
            return tpl.render("identified", lang, name=state.customer_first_name), {}
        if intent in INTENTS_REQUIRING_VERIFICATION and not state.verified:
            key = "ask_dob_read" if intent in READ_INTENTS else "ask_dob"
            return prefix + tpl.render(key, lang), {"awaiting": "dob", "dob_attempts": attempts["dob"]}
        if intent == "policy_question":
            return self._render_policy(out.get("search_policies", {}), user_text, lang)
        if intent == "payment_status":
            txs = out["get_transaction_status"]["transactions"]
            offer = {"offer": "escalation"} if any(t["status"] == "under_review" for t in txs) else {}
            return prefix + self._render_transactions(txs, lang), offer
        if intent == "ticket_status":
            return prefix + self._render_tickets(out["get_tickets"], lang), {}
        if intent == "create_ticket":
            t = out["create_ticket"]
            return prefix + tpl.render("ticket_created", lang, ticket_id=t["id"], sla=t["sla_hours"]), {}
        if intent == "schedule_callback":
            if "schedule_callback" not in out:
                return prefix + tpl.render("ask_callback_time", lang), {"awaiting": "callback_time"}
            cb = out["schedule_callback"]
            date_s, time_s = tpl.format_datetime(datetime.fromisoformat(cb["scheduled_for"]), self.tz, lang)
            return prefix + tpl.render(
                "callback_scheduled", lang, date=date_s, time=time_s, last4=cb["phone_last4"], callback_id=cb["id"]
            ), {}
        if intent == "card_block" and "block_card" in out:
            res = out["block_card"]
            if res.get("confirmation_required"):
                s = res["summary"]
                kind = tpl.CARD_KIND[s["card_kind"]][lang]
                return tpl.render("confirm_block", lang, last4=s["card_last4"], kind=kind), {"awaiting": "confirmation"}
            kind = tpl.CARD_KIND[res["card_kind"]][lang]
            return tpl.render("card_blocked", lang, last4=res["card_last4"], kind=kind), {}
        if intent == "dispute_transaction":
            if "dispute_transaction" not in out:
                return prefix + tpl.render("ask_dispute_tx", lang), {"awaiting": "transaction"}
            res = out["dispute_transaction"]
            if res.get("confirmation_required"):
                s = res["summary"]
                date_s, _ = tpl.format_datetime(datetime.fromisoformat(s["created_at"]), self.tz, lang)
                text = tpl.render(
                    "confirm_dispute",
                    lang,
                    date=date_s,
                    counterparty=s["counterparty"],
                    amount=tpl.format_amount(s["amount"], s["currency"], lang),
                )
                if res.get("requires_supervisor"):
                    text += tpl.T["supervisor_note"][lang]
                return text, {"awaiting": "confirmation"}
            if res.get("approval_id"):
                return tpl.render("dispute_pending_approval", lang, approval_id=res["approval_id"]), {}
            return tpl.render("dispute_opened", lang, dispute_id=res["id"], days=res["review_days"]), {}
        if intent == "unknown" and not pending and _CONFIRM_WORDS.search(fold(user_text)):
            # "Evet, onaylıyorum" after an action was already refused (e.g. a dispute that
            # exists): say plainly that nothing is waiting, instead of "I didn't catch that".
            return tpl.render("nothing_to_confirm", lang), {}
        return tpl.render("unknown", lang), {}

    def _render_error(
        self, tool: str, err: dict[str, Any], lang: Language, attempts: dict[str, int]
    ) -> tuple[str, dict[str, Any]]:
        code = err.get("error")
        if tool == "lookup_customer" and code == "customer_not_found":
            return tpl.render("customer_not_found", lang), {
                "awaiting": "identity",
                "identity_attempts": attempts["identity"] + 1,
            }
        if tool == "verify_identity" and code == "verification_failed":
            return tpl.render("dob_mismatch", lang), {"awaiting": "dob", "dob_attempts": attempts["dob"] + 1}
        if tool == "get_transaction_status" and code == "transaction_not_found":
            return tpl.render("tx_not_found", lang), {}
        if tool == "get_tickets" and code == "ticket_not_found":
            return tpl.render("ticket_not_found", lang), {}
        if tool == "schedule_callback" and err.get("status") == 422:
            reason = tpl.CALLBACK_ERRORS.get(code or "", {}).get(lang, err.get("message", ""))
            return tpl.render("callback_invalid", lang, reason=reason), {"awaiting": "callback_time"}
        if tool == "block_card" and code == "multiple_cards":
            joiner = " ya da " if lang == "tr" else " or "
            opts = [
                (f"sonu {o['last4']} olan {tpl.CARD_KIND[o['kind']]['tr']} kartı")
                if lang == "tr"
                else f"the {tpl.CARD_KIND[o['kind']]['en']} card ending in {o['last4']}"
                for o in err.get("details", {}).get("options", [])
            ]
            return tpl.render("ask_which_card", lang, options=joiner.join(opts)), {"awaiting": "card"}
        if tool == "block_card" and code == "card_not_found":
            return tpl.render("card_not_found", lang), {"offer": "escalation"}
        if tool == "dispute_transaction":
            if code == "transaction_not_found":
                return tpl.render("tx_not_found", lang), {"awaiting": "transaction"}
            if code == "not_disputable":
                return tpl.render("not_disputable", lang), {}
            if code == "dispute_exists":
                return tpl.render("dispute_exists", lang), {}
        return tpl.render("system_error", lang), {"error": code, "offer": "escalation"}

    @staticmethod
    def _render_policy(res: dict[str, Any], query: str, lang: Language) -> tuple[str, dict[str, Any]]:
        passages = res.get("passages") or []
        if not passages:
            return tpl.render("policy_not_found", lang), {"offer": "escalation", "citations": []}
        top = passages[0]
        passage = Passage(top["id"], top["id"].split("#")[0], top["title"], top["section"], lang, top["text"])
        answer = best_sentences(passage, query)
        text = tpl.render("policy_answer", lang, answer=answer, title=top["title"], section=top["section"])
        return text, {"citations": [p["id"] for p in passages]}

    def _render_tickets(self, res: dict[str, Any], lang: Language) -> str:
        tickets = res.get("tickets") or []
        if not tickets:
            return tpl.render("ticket_none", lang)
        t = tickets[0]  # newest first, or the one the caller named
        created = t.get("created_at")
        date_s = tpl.format_datetime(datetime.fromisoformat(created), self.tz, lang)[0] if created else ""
        status = tpl.TICKET_STATUS_LABELS.get(t["status"], {}).get(lang, t["status"])
        return tpl.render("ticket_status", lang, date=date_s, ticket_id=t["id"], status=status, summary=t["summary"]).strip()

    def _render_transactions(self, txs: list[dict[str, Any]], lang: Language) -> str:
        if not txs:
            return tpl.render("tx_none", lang)
        # Callers usually ask about the payment that is *not* done yet.
        tx = next((t for t in txs if t["status"] != "completed"), txs[0])
        date_s, _ = tpl.format_datetime(datetime.fromisoformat(tx["created_at"]), self.tz, lang)
        detail = tpl.REASON_DETAILS.get(tx.get("status_reason") or "", {}).get(lang, "")
        return tpl.render(
            "tx_status",
            lang,
            date=date_s,
            counterparty=tx["counterparty"],
            amount=tpl.format_amount(tx["amount"], tx["currency"], lang),
            status=tpl.STATUS_LABELS.get(tx["status"], {}).get(lang, tx["status"]),
            detail=detail,
        ).strip()

    # ------------------------------------------------------------------ usage
    def _result(
        self,
        text: str | None,
        calls: list[ToolCallRequest],
        meta: dict[str, Any],
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
    ) -> LLMResult:
        # ~4 characters per token: rough, but lets cost-per-conversation be
        # estimated end to end before a real provider is switched on.
        prompt_chars = len(system) + sum(len(m.content) + len(m.meta.get("context", "")) for m in messages)
        prompt_chars += sum(len(t.description) + len(json.dumps(t.input_schema)) for t in tools)
        output_chars = len(text or "") + sum(len(json.dumps(c.arguments)) for c in calls)
        usage = LLMUsage(input_tokens=prompt_chars // 4, output_tokens=max(1, output_chars // 4), estimated=True)
        return LLMResult(
            text=text,
            tool_calls=calls,
            model=self.model,
            usage=usage,
            stop_reason="tool_use" if calls else "end_turn",
            meta=meta,
        )
