"""Tool registry: the only way the agent can touch the customer's systems.

Each tool declares a JSON schema (what the LLM sees) and a handler (what actually
runs). The registry enforces guards before any handler executes, so a model that
hallucinates a call cannot bypass them:

* unknown tool              -> error result, nothing runs
* arguments                 -> validated against the tool's JSON schema (defence in
                               depth: streamed tool inputs are not validated server-side)
* customer-scoped tool      -> blocked until lookup_customer succeeded; the customer
                               id comes from session state, never from model arguments
* account data / sensitive  -> blocked until step-up verification (date of birth):
                               a customer ID or phone number identifies the caller,
                               it does not authenticate them
* approval gate             -> a sensitive action is first *prepared* (resolved and
                               validated), then parked as `pending_action` until the
                               caller says yes on their next turn; high-value actions
                               additionally wait for a supervisor (see api/approvals.py)
* integration failures      -> normalised error payloads the model can explain

Every tool call runs inside an integration scope: its POSTs carry idempotency keys
derived from the conversation and tool call id, and the HTTP calls it made
(attempts, latency, replays) are attached to the outcome for the tool span.
"""

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from voiceops.agent.llm.base import ToolSpec
from voiceops.agent.state import ConversationState, ToolCallRequest
from voiceops.integrations import IntegrationClient, IntegrationError, integration_scope
from voiceops.rag import PolicyStore, load_policy_store

SUPERVISOR_THRESHOLD_TRY = Decimal("10000")

OutcomeStatus = Literal["ok", "error", "blocked", "needs_confirmation", "pending_approval"]


class ToolInputError(ValueError):
    pass


@dataclass
class ToolContext:
    integrations: IntegrationClient
    state: ConversationState
    conversation_id: str
    policies: PolicyStore | None = None
    # Approval requests raised during this turn; the API layer persists them.
    approval_requests: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolOutcome:
    status: OutcomeStatus
    payload: dict[str, Any]
    # HTTP calls made while running the tool (for traces, never shown to the LLM).
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return self.status in ("error", "blocked")


Handler = Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]
# prepare(args, ctx) -> (canonical args, human-readable summary). Must be idempotent.
Preparer = Callable[[dict[str, Any], ToolContext], Awaitable[tuple[dict[str, Any], dict[str, Any]]]]


@dataclass
class Approval:
    prepare: Preparer
    key: Callable[[dict[str, Any]], str]  # what the caller is agreeing to (card id, tx id)
    level: Callable[[dict[str, Any]], Literal["confirm", "supervisor"]] = lambda summary: "confirm"


@dataclass
class Tool:
    spec: ToolSpec
    handler: Handler
    requires_customer: bool = False
    requires_verification: bool = False
    approval: Approval | None = None
    intent: str | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def execute(self, call: ToolCallRequest, ctx: ToolContext) -> ToolOutcome:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolOutcome("error", {"error": "unknown_tool", "message": f"No tool named {call.name!r}"})
        if problem := validate_arguments(tool.spec.input_schema, call.arguments):
            return ToolOutcome(
                "error", {"error": "invalid_arguments", "message": problem, "received": json_safe(call.arguments)}
            )
        with integration_scope(f"{ctx.conversation_id}:{call.id}") as calls:
            outcome = await self._execute(tool, call, ctx)
        outcome.calls = calls
        return outcome

    async def _execute(self, tool: Tool, call: ToolCallRequest, ctx: ToolContext) -> ToolOutcome:
        if (tool.requires_customer or tool.requires_verification) and ctx.state.customer_id is None:
            return ToolOutcome(
                "blocked",
                {
                    "error": "customer_not_identified",
                    "message": "Ask the caller for their registered phone number or customer ID, "
                    "then call lookup_customer before this tool.",
                },
            )
        if tool.requires_verification and not ctx.state.verified:
            return ToolOutcome(
                "blocked",
                {
                    "error": "verification_required",
                    "message": "Account data and sensitive actions need step-up verification: ask the caller for "
                    "their date of birth and call verify_identity first.",
                },
            )
        try:
            if tool.approval is not None:
                return await self._gate(tool, call, ctx)
            return ToolOutcome("ok", await tool.handler(call.arguments, ctx))
        except IntegrationError as exc:
            return ToolOutcome("error", exc.to_payload())
        except ToolInputError as exc:
            return ToolOutcome("error", {"error": "invalid_arguments", "message": str(exc)})

    async def _gate(self, tool: Tool, call: ToolCallRequest, ctx: ToolContext) -> ToolOutcome:
        assert tool.approval is not None
        args, summary = await tool.approval.prepare(call.arguments, ctx)
        key = tool.approval.key(args)
        level = tool.approval.level(summary)
        pending = ctx.state.pending_action
        confirmed = bool(
            pending
            and pending.get("kind") == "confirm"
            and pending.get("confirmed")
            and pending["tool"] == tool.spec.name
            and pending["key"] == key
        )
        if not confirmed:
            ctx.state.pending_action = {
                "id": f"act_{uuid.uuid4().hex[:10]}",
                "kind": "confirm",
                "tool": tool.spec.name,
                "intent": tool.intent,
                "arguments": args,  # canonical, resolved by prepare()
                "requested": dict(call.arguments),  # as the model asked (schema-valid)
                "key": key,
                "summary": summary,
                "level": level,
                "confirmed": False,
            }
            return ToolOutcome(
                "needs_confirmation",
                {
                    "confirmation_required": True,
                    "summary": summary,
                    "requires_supervisor": level == "supervisor",
                    "message": "Nothing has been done yet. Read the summary to the caller and ask for an explicit yes. "
                    "Call this tool again with the same target only after they confirm.",
                },
            )
        if level == "supervisor":
            approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"
            request = {
                "id": approval_id,
                "tool": tool.spec.name,
                "arguments": args,
                "summary": summary,
                "customer_id": ctx.state.customer_id,
                "reason": f"{tool.spec.name} above {SUPERVISOR_THRESHOLD_TRY} TRY",
            }
            ctx.approval_requests.append(request)
            ctx.state.pending_action = {**pending, "kind": "approval", "approval_id": approval_id}  # type: ignore[dict-item]
            return ToolOutcome(
                "pending_approval",
                {
                    "approval_id": approval_id,
                    "summary": summary,
                    "message": "Sent to a supervisor for approval. Tell the caller it is not done yet and that they will "
                    "be informed of the decision.",
                },
            )
        ctx.state.pending_action = None
        return ToolOutcome("ok", await tool.handler(args, ctx))

    async def execute_approved(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext, approver: str, scope: str | None = None
    ) -> ToolOutcome:
        """Run an action a supervisor approved. Identity guards still apply; the
        confirmation gate does not (the caller confirmed before the request was raised).
        `scope` (the approval id) makes a repeated execution idempotent downstream."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome("error", {"error": "unknown_tool", "message": name})
        if ctx.state.customer_id is None:
            return ToolOutcome("blocked", {"error": "customer_not_identified", "message": "No customer on session"})
        with integration_scope(scope) as calls:
            try:
                outcome = ToolOutcome("ok", await tool.handler({**arguments, "approved_by": approver}, ctx))
            except IntegrationError as exc:
                outcome = ToolOutcome("error", exc.to_payload())
            except ToolInputError as exc:
                outcome = ToolOutcome("error", {"error": "invalid_arguments", "message": str(exc)})
        outcome.calls = calls
        return outcome


# --------------------------------------------------------------------------- argument validation
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_arguments(schema: dict[str, Any], args: Any) -> str | None:
    """Check tool arguments against the (flat) JSON schemas used by the registry.
    Returns a message for the model, or None when the arguments are valid. Null
    values for optional properties are tolerated (some providers send them)."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if args.get(k) is None]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    if schema.get("additionalProperties") is False and (unknown := sorted(set(args) - set(props))):
        return f"unknown argument(s): {', '.join(unknown)}"
    for name, value in args.items():
        spec = props.get(name)
        if spec is None or value is None:
            continue
        expected = _JSON_TYPES.get(spec.get("type", ""))
        is_bool = isinstance(value, bool)
        if expected and (not isinstance(value, expected) or (is_bool and spec["type"] in ("integer", "number"))):
            return f"{name} must be of type {spec['type']}"
        if "enum" in spec and value not in spec["enum"]:
            return f"{name} must be one of {spec['enum']}"
    return None


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)[:300]


# --------------------------------------------------------------------------- handlers
def _tx_view(tx: dict[str, Any]) -> dict[str, Any]:
    keep = ("id", "kind", "amount", "currency", "counterparty", "status", "status_reason", "created_at")
    return {k: tx[k] for k in keep}


async def lookup_customer(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    params = {k: args[k] for k in ("phone", "customer_id") if args.get(k)}
    if not params:
        raise ToolInputError("Provide phone or customer_id")
    cust = await ctx.integrations.get("/crm/customers", params=params)
    if ctx.state.customer_id != cust["id"]:
        ctx.state.verified = False  # verification belongs to one identity
    ctx.state.customer_id = cust["id"]
    ctx.state.customer_first_name = cust["full_name"].split()[0]
    # Data minimisation: the model only needs what it will say or act on.
    return {
        "customer_id": cust["id"],
        "first_name": ctx.state.customer_first_name,
        "preferred_language": cust["preferred_language"],
        "segment": cust["segment"],
        "phone_last4": cust["phone"][-4:],
    }


async def verify_identity(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        dob = date.fromisoformat(str(args.get("date_of_birth", "")))
    except ValueError as exc:
        raise ToolInputError("date_of_birth must be YYYY-MM-DD") from exc
    res = await ctx.integrations.post(
        f"/crm/customers/{ctx.state.customer_id}/verify", json={"date_of_birth": dob.isoformat()}
    )
    if not res["verified"]:
        raise IntegrationError("crm", 401, "verification_failed", "Date of birth does not match our records")
    ctx.state.verified = True
    return {"verified": True}


async def get_transaction_status(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    cid = ctx.state.customer_id
    if tx_id := args.get("transaction_id"):
        tx = await ctx.integrations.get(f"/payments/transactions/{tx_id}", params={"customer_id": cid})
        return {"transactions": [_tx_view(tx)]}
    limit = min(max(int(args.get("limit") or 3), 1), 10)
    txs = await ctx.integrations.get(f"/payments/customers/{cid}/transactions", params={"limit": limit})
    return {"transactions": [_tx_view(t) for t in txs]}


async def get_tickets(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    tickets = await ctx.integrations.get("/helpdesk/tickets", params={"customer_id": ctx.state.customer_id})
    tickets = sorted(tickets, key=lambda t: (t.get("created_at") or "", t["id"]), reverse=True)  # newest first
    if ticket_id := str(args.get("ticket_id") or "").strip().upper():
        tickets = [t for t in tickets if t["id"] == ticket_id]
        if not tickets:
            raise IntegrationError("helpdesk", 404, "ticket_not_found", f"No ticket {ticket_id} for this customer")
    keep = ("id", "category", "status", "summary", "created_at")
    return {"tickets": [{k: t.get(k) for k in keep} for t in tickets[:5]], "total": len(tickets)}


async def create_ticket(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    summary = (args.get("summary") or "").strip()
    if len(summary) < 3:
        raise ToolInputError("summary is required")
    ticket = await ctx.integrations.post(
        "/helpdesk/tickets",
        json={
            "customer_id": ctx.state.customer_id,
            "category": args.get("category") or "other",
            "priority": args.get("priority") or "normal",
            "summary": summary,
            "conversation_id": ctx.conversation_id,
        },
    )
    return {k: ticket[k] for k in ("id", "category", "priority", "status", "sla_hours")}


async def schedule_callback(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    try:
        when = datetime.fromisoformat(args["preferred_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolInputError("preferred_time must be an ISO-8601 datetime") from exc
    if when.tzinfo is None:
        raise ToolInputError("preferred_time must include a UTC offset, e.g. 2026-11-09T14:00:00+03:00")
    cb = await ctx.integrations.post(
        "/scheduling/callbacks",
        json={
            "customer_id": ctx.state.customer_id,
            "scheduled_for": when.isoformat(),
            "reason": args.get("reason") or "callback requested",
        },
    )
    return {k: cb[k] for k in ("id", "scheduled_for", "phone_last4", "status")}


async def search_policies(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        raise ToolInputError("query is required")
    store = ctx.policies or load_policy_store()
    hits = store.search(query, lang=ctx.state.language, k=3)
    if not hits:
        return {"passages": [], "note": "No relevant policy found. Do not guess; offer a ticket or a representative."}
    return {
        "passages": [
            {"id": p.id, "title": p.title, "section": p.section, "text": p.text, "score": p.score} for p in hits
        ]
    }


async def _prepare_block_card(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], dict[str, Any]]:
    cards = await ctx.integrations.get(f"/payments/customers/{ctx.state.customer_id}/cards")
    active = [c for c in cards if c["status"] == "active"]
    if card_id := args.get("card_id"):
        match = [c for c in cards if c["id"] == card_id]
    elif last4 := args.get("card_last4"):
        match = [c for c in active if c["last4"] == str(last4)]
    elif kind := args.get("card_kind"):
        match = [c for c in active if c["kind"] == kind]
    else:
        match = active
    if not match:
        raise IntegrationError("payments", 404, "card_not_found", "No active card matches")
    if len(match) > 1:
        options = [{"last4": c["last4"], "kind": c["kind"]} for c in match]
        raise IntegrationError(
            "payments", 409, "multiple_cards", "Ask the caller which card to block", {"options": options}
        )
    card = match[0]
    reason = args.get("reason") if args.get("reason") in ("lost", "stolen", "fraud_suspected") else "lost_or_stolen"
    return {"card_id": card["id"], "reason": reason}, {"card_last4": card["last4"], "card_kind": card["kind"]}


async def block_card(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    card = await ctx.integrations.post(
        f"/payments/cards/{args['card_id']}/block",
        json={"customer_id": ctx.state.customer_id, "reason": args.get("reason", "lost_or_stolen")},
    )
    return {"card_last4": card["last4"], "card_kind": card["kind"], "status": card["status"]}


async def _prepare_dispute(args: dict[str, Any], ctx: ToolContext) -> tuple[dict[str, Any], dict[str, Any]]:
    tx_id = str(args.get("transaction_id") or "").upper()
    if not tx_id:
        raise ToolInputError("transaction_id is required (format TX-12345)")
    tx = await ctx.integrations.get(f"/payments/transactions/{tx_id}", params={"customer_id": ctx.state.customer_id})
    if tx["kind"] != "card":
        raise IntegrationError("payments", 422, "not_disputable", "Only card transactions can be disputed")
    # Fail before the caller confirms (and before a supervisor is involved), not after.
    open_disputes = await ctx.integrations.get(f"/payments/customers/{ctx.state.customer_id}/disputes")
    if existing := next((d for d in open_disputes if d["transaction_id"] == tx["id"]), None):
        raise IntegrationError("payments", 409, "dispute_exists", f"Dispute {existing['id']} is already open", {"dispute_id": existing["id"]})
    reason = (args.get("reason") or "unrecognised transaction").strip()[:300]
    summary = {
        "transaction_id": tx["id"],
        "amount": tx["amount"],
        "currency": tx["currency"],
        "counterparty": tx["counterparty"],
        "created_at": tx["created_at"],
    }
    return {"transaction_id": tx["id"], "reason": reason}, summary


async def dispute_transaction(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    body = {"customer_id": ctx.state.customer_id, "transaction_id": args["transaction_id"], "reason": args["reason"]}
    if args.get("approved_by"):
        body["approved_by"] = args["approved_by"]
    d = await ctx.integrations.post("/payments/disputes", json=body)
    return {k: d[k] for k in ("id", "transaction_id", "status", "review_days")}


async def escalate_to_human(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    ho = await ctx.integrations.post(
        "/contact-center/handoffs",
        json={
            "conversation_id": ctx.conversation_id,
            "customer_id": ctx.state.customer_id,
            "reason": args.get("reason") or "caller_request",
            "language": ctx.state.language,
            "priority": args.get("priority") or "normal",
        },
    )
    ctx.state.status = "escalated"
    ctx.state.handoff_id = ho["id"]
    ctx.state.pending_action = None
    return {k: ho[k] for k in ("id", "queue_position", "estimated_wait_minutes", "status")}


# --------------------------------------------------------------------------- registry
def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            ToolSpec(
                "lookup_customer",
                "Find and identify the caller in the CRM by registered phone number or customer ID "
                "(format C-1234). Must succeed before any account-specific tool can run.",
                _schema(
                    {
                        "phone": {"type": "string", "description": "Phone number as spoken, any format"},
                        "customer_id": {"type": "string", "description": "Customer ID such as C-1001"},
                    }
                ),
            ),
            lookup_customer,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "verify_identity",
                "Step-up verification before reading account data (transactions, tickets) or taking a sensitive "
                "action (blocking a card, disputing a transaction): check the identified caller's date of birth.",
                _schema({"date_of_birth": {"type": "string", "description": "YYYY-MM-DD"}}, ["date_of_birth"]),
            ),
            verify_identity,
            requires_customer=True,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "get_transaction_status",
                "Get the status of the verified caller's payments/transfers. Requires verify_identity first. "
                "Pass transaction_id (format TX-12345) for a specific one, otherwise the most recent ones "
                "(limit 1-10) are returned.",
                _schema({"transaction_id": {"type": "string"}, "limit": {"type": "integer"}}),
            ),
            get_transaction_status,
            requires_customer=True,
            requires_verification=True,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "get_tickets",
                "Look up the verified caller's existing support tickets, newest first (number, status, subject). "
                "Requires verify_identity first. Use it when the caller asks about a ticket they already have; "
                "never open a new ticket to answer that.",
                _schema({"ticket_id": {"type": "string", "description": "Ticket number such as TCK-000123"}}),
            ),
            get_tickets,
            requires_customer=True,
            requires_verification=True,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "search_policies",
                "Search the bank's official policy documents (fees, transfer cut-off times, limits, cards, "
                "disputes, service times, security). Answer policy questions only from the returned passages "
                "and mention the source section.",
                _schema({"query": {"type": "string"}}, ["query"]),
            ),
            search_policies,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "create_ticket",
                "Open a NEW support ticket for the identified caller describing their problem. Only when the "
                "caller asks to report a problem or open a ticket; for questions about an existing ticket use get_tickets.",
                _schema(
                    {
                        "category": {"type": "string", "enum": ["card", "payment", "account", "app", "complaint", "other"]},
                        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                        "summary": {"type": "string", "description": "One or two sentences in the caller's words"},
                    },
                    ["category", "summary"],
                ),
            ),
            create_ticket,
            requires_customer=True,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "schedule_callback",
                "Schedule a phone callback to the identified caller's registered number. Callbacks run "
                "09:00-18:00 Europe/Istanbul, up to 14 days ahead.",
                _schema(
                    {
                        "preferred_time": {"type": "string", "description": "ISO-8601 with offset, e.g. 2026-11-09T14:00:00+03:00"},
                        "reason": {"type": "string"},
                    },
                    ["preferred_time", "reason"],
                ),
            ),
            schedule_callback,
            requires_customer=True,
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "block_card",
                "Permanently block one of the caller's cards (lost/stolen). Requires verify_identity first. "
                "The first call only prepares the action and returns a summary to confirm with the caller.",
                _schema(
                    {
                        "card_last4": {"type": "string", "description": "Last 4 digits, if the caller said them"},
                        "card_kind": {"type": "string", "enum": ["debit", "credit"]},
                        "reason": {"type": "string", "enum": ["lost", "stolen", "fraud_suspected"]},
                    }
                ),
            ),
            block_card,
            requires_customer=True,
            requires_verification=True,
            approval=Approval(prepare=_prepare_block_card, key=lambda a: a["card_id"]),
            intent="card_block",
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "dispute_transaction",
                "Open a dispute (chargeback) for one of the caller's card transactions. Requires verify_identity "
                "first. The first call only prepares the action; disputes of 10,000 TRY or more also need "
                "supervisor approval.",
                _schema(
                    {"transaction_id": {"type": "string"}, "reason": {"type": "string"}},
                    ["transaction_id", "reason"],
                ),
            ),
            dispute_transaction,
            requires_customer=True,
            requires_verification=True,
            approval=Approval(
                prepare=_prepare_dispute,
                key=lambda a: a["transaction_id"],
                level=lambda s: "supervisor" if Decimal(str(s["amount"])) >= SUPERVISOR_THRESHOLD_TRY else "confirm",
            ),
            intent="dispute_transaction",
        )
    )
    reg.register(
        Tool(
            ToolSpec(
                "escalate_to_human",
                "Transfer the conversation to a human representative. Use when the caller asks for a "
                "person, when you cannot help, or when the request is sensitive.",
                _schema(
                    {
                        "reason": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                    },
                    ["reason"],
                ),
            ),
            escalate_to_human,
        )
    )
    return reg
