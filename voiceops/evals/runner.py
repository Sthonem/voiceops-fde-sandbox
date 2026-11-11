"""Run golden scenarios against the real app, in-process, through its public APIs.

Every scenario gets a freshly seeded database, so side effects (a blocked card,
a ticket) never leak into the next scenario and results are reproducible. Text
scenarios use the REST API exactly like a client; voice scenarios drive a
VoiceSession with synthesised caller audio; supervisor steps use the approvals
API. Faults are injected into the mock systems where a scenario asks for them.
"""

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from voiceops.config import Settings
from voiceops.db import configure_engine, get_engine, init_db
from voiceops.evals.scenarios import Scenario, Step, TurnExpect
from voiceops.mock_systems.app import fault_injector

# A tool call that did its job, even if the job was "wait for the caller's yes".
SUCCESS_STATUSES = {"ok", "needs_confirmation", "pending_approval"}


@dataclass
class Check:
    name: str
    passed: bool
    category: str
    expected: Any = None
    actual: Any = None


@dataclass
class StepResult:
    index: int
    kind: str
    input: str | None
    reply: str | None = None
    intent: str | None = None
    expected_intent: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_expectation: list[str] | None = None
    status_expectation: list[str] | None = None
    latency_ms: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False
    reprompted: bool = False
    filler: bool = False
    voice: dict[str, Any] | None = None
    checks: list[Check] = field(default_factory=list)


@dataclass
class ScenarioResult:
    id: str
    title: str
    language: str
    channel: str
    tags: list[str]
    failure_modes: list[str]
    known_limit: str | None
    expect_escalation: bool
    faulted: bool = False  # faults were injected: latency is reported separately
    run: int = 0
    conversation_id: str | None = None
    escalated: bool = False
    steps: list[StepResult] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    http: dict[str, int] = field(default_factory=lambda: {"calls": 0, "retried": 0, "replayed": 0})
    llm_calls: int = 0
    duration_ms: float = 0.0

    @property
    def failures(self) -> list[tuple[int | None, Check]]:
        out: list[tuple[int | None, Check]] = [(s.index, c) for s in self.steps for c in s.checks if not c.passed]
        return out + [(None, c) for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)


# --------------------------------------------------------------------------- checks
def check_turn(expect: TurnExpect, turn: dict[str, Any] | None, reprompted: bool = False, filler: bool = False) -> list[Check]:
    checks: list[Check] = []
    if expect.reprompt is not None:
        checks.append(Check("reprompt", reprompted == expect.reprompt, "voice_reprompt", expect.reprompt, reprompted))
    if expect.filler is not None:
        checks.append(Check("filler", filler == expect.filler, "voice_filler", expect.filler, filler))
    if turn is None:
        if not reprompted:
            checks.append(Check("turn", False, "runner_error", "an agent turn", "no turn"))
        return checks
    reply = turn["reply"]
    names = [t["name"] for t in turn["tool_calls"]]
    statuses = [t["status"] for t in turn["tool_calls"]]
    if expect.intent is not None:
        checks.append(Check("intent", turn["intent"] == expect.intent, "intent_mismatch", expect.intent, turn["intent"]))
    if expect.tools is not None:
        checks.append(Check("tools", names == expect.tools, "wrong_tools", expect.tools, names))
    if expect.tool_statuses is not None:
        checks.append(Check("tool_statuses", statuses == expect.tool_statuses, "tool_status_mismatch", expect.tool_statuses, statuses))
    elif unexpected := [f"{n}:{s}" for n, s in zip(names, statuses, strict=True) if s not in SUCCESS_STATUSES]:
        checks.append(Check("tool_failures", False, "tool_failure", "no error/blocked tool calls", unexpected))
    for text in expect.reply_contains:
        checks.append(Check(f"reply contains {text!r}", text in reply, "missing_content", text, reply))
    for text in expect.reply_not_contains:
        checks.append(Check(f"reply lacks {text!r}", text not in reply, "forbidden_content", text, reply))
    if expect.escalated is not None:
        actual = turn["status"] == "escalated"
        checks.append(Check("escalated", actual == expect.escalated, "escalation_timing", expect.escalated, actual))
    state = {
        "awaiting_confirmation": (expect.awaiting_confirmation, turn["awaiting_confirmation"] is not None),
        "verified": (expect.verified, turn["verified"]),
        "customer_id": (expect.customer_id, turn["customer_id"]),
        "language": (expect.language, turn["language"]),
        "approvals_raised": (expect.approvals_raised, len(turn["approval_ids"])),
    }
    for name, (want, got) in state.items():
        if want is not None:
            checks.append(Check(name, want == got, "state_mismatch", want, got))
    if expect.citations is not None:
        got = bool(turn["citations"])
        checks.append(Check("citations", got == expect.citations, "missing_citation", expect.citations, turn["citations"]))
    return checks


def _from_turn(sr: StepResult, turn: dict[str, Any]) -> None:
    sr.reply = turn["reply"]
    sr.intent = turn["intent"]
    sr.tools = [{"name": t["name"], "status": t["status"], "latency_ms": t["latency_ms"]} for t in turn["tool_calls"]]
    sr.latency_ms = turn["latency_ms"]
    usage = turn["usage"]
    sr.cost_usd = usage["estimated_cost_usd"]
    sr.input_tokens, sr.output_tokens, sr.estimated = usage["input_tokens"], usage["output_tokens"], usage["estimated"]


class _Collector:
    """Outbound side of a voice session, kept in memory."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.audio_bytes = 0

    async def send_json(self, data: dict[str, Any]) -> None:
        self.events.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.audio_bytes += len(data)


# --------------------------------------------------------------------------- runner
class EvalRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings(log_level="WARNING")

    async def run(
        self,
        scenarios: list[Scenario],
        repeat: int = 1,
        on_result: Callable[[ScenarioResult], None] | None = None,
    ) -> list[list[ScenarioResult]]:
        from voiceops.main import create_app

        runs: list[list[ScenarioResult]] = []
        with tempfile.TemporaryDirectory(prefix="voiceops-eval-") as tmp:
            configure_engine(f"sqlite+aiosqlite:///{Path(tmp) / 'bootstrap.db'}")
            app = create_app(self.settings)
            try:
                async with app.router.lifespan_context(app):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://voiceops", timeout=120) as client:
                        for r in range(repeat):
                            run: list[ScenarioResult] = []
                            for i, sc in enumerate(scenarios):
                                await get_engine().dispose()
                                configure_engine(f"sqlite+aiosqlite:///{Path(tmp) / f'r{r}-{i}.db'}")
                                await init_db(seed=True)
                                result = await self._scenario(app, client, sc)
                                result.run = r
                                run.append(result)
                                if on_result:
                                    on_result(result)
                            runs.append(run)
            finally:
                fault_injector.clear()
                await get_engine().dispose()
        return runs

    async def _scenario(self, app: Any, client: httpx.AsyncClient, sc: Scenario) -> ScenarioResult:
        res = ScenarioResult(sc.id, sc.title, sc.language, sc.channel, sc.tags, sc.failure_modes, sc.known_limit,
                             sc.expect_escalation, faulted=bool(sc.faults or any(st.faults for st in sc.steps)))  # fmt: skip
        started = time.perf_counter()
        fault_injector.clear()
        for f in sc.faults:
            fault_injector.add(f)
        session = collector = None
        try:
            if sc.channel == "voice":
                session, collector = await self._voice_session(app, sc.language)
                res.conversation_id = session.conversation_id
            else:
                resp = await client.post("/v1/conversations", json={"language": sc.language, "channel": "text"})
                resp.raise_for_status()
                res.conversation_id = resp.json()["id"]
            for idx, step in enumerate(sc.steps):
                for f in step.faults:
                    fault_injector.add(f)
                if step.kind == "supervisor":
                    sr = await self._supervisor(client, res.conversation_id, idx, step)
                elif sc.channel == "voice":
                    sr = await self._voice_step(session, collector, idx, step)
                else:
                    sr = await self._text_step(client, res.conversation_id, idx, step)
                res.steps.append(sr)
            fault_injector.clear()  # verification reads must not hit injected faults
            await self._conversation_checks(client, sc, res)
        except Exception as exc:  # a crashing scenario is a result, not an aborted run
            res.checks.append(Check("runner", False, "runner_error", "scenario completes", repr(exc)[:300]))
        finally:
            fault_injector.clear()
            if session is not None:
                await session.close(graceful=True)
        res.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return res

    # ------------------------------------------------------------------ steps
    async def _text_step(self, client: httpx.AsyncClient, cid: str | None, idx: int, step: Step) -> StepResult:
        sr = StepResult(idx, "say", step.say, expected_intent=step.expect.intent,
                        tool_expectation=step.expect.tools, status_expectation=step.expect.tool_statuses)  # fmt: skip
        resp = await client.post(f"/v1/conversations/{cid}/messages", json={"text": step.say})
        if resp.status_code != 200:
            sr.checks.append(Check("http", False, "runner_error", 200, f"{resp.status_code} {resp.text[:200]}"))
            return sr
        turn = resp.json()
        _from_turn(sr, turn)
        sr.checks += check_turn(step.expect, turn)
        return sr

    async def _supervisor(self, client: httpx.AsyncClient, cid: str | None, idx: int, step: Step) -> StepResult:
        sr = StepResult(idx, "supervisor", step.supervisor)
        pending = [a for a in (await client.get("/v1/approvals", params={"status": "pending"})).json() if a["conversation_id"] == cid]
        if not pending:
            sr.checks.append(Check("approval_request", False, "state_mismatch", "a pending approval", "none"))
            return sr
        latest = pending[-1]
        resp = await client.post(
            f"/v1/approvals/{latest['id']}/decision", json={"decision": step.supervisor, "reviewer": "eval.supervisor"}
        )
        want = "executed" if step.supervisor == "approve" else "rejected"
        got = resp.json().get("status") if resp.status_code == 200 else f"http {resp.status_code}"
        sr.reply = f"{latest['id']} {got}"
        sr.checks.append(Check("approval_decision", got == want, "side_effect_mismatch", want, got))
        return sr

    async def _voice_session(self, app: Any, language: str) -> tuple[Any, _Collector]:
        import dataclasses
        import json

        from voiceops.voice import VoiceServices, VoiceSession

        base: VoiceServices = app.state.voice
        services = VoiceServices(base.orchestrator, base.stt, base.tts, dataclasses.replace(base.config, greeting=False))
        collector = _Collector()
        session = VoiceSession(services, collector)
        await session.handle_message(json.dumps({"type": "session.start", "language": language, "audio": {"format": "wav"}}))
        if session.conversation_id is None:
            raise RuntimeError(f"voice session did not start: {collector.events}")
        return session, collector

    async def _voice_step(self, session: Any, collector: _Collector, idx: int, step: Step) -> StepResult:
        from voiceops.agent.text import detect_language
        from voiceops.speech.audio import build_wav, silence
        from voiceops.speech.mock import synthesize

        sr = StepResult(idx, "audio" if step.audio else "say", step.say or f"<{step.audio}>",
                        expected_intent=step.expect.intent, tool_expectation=step.expect.tools,
                        status_expectation=step.expect.tool_statuses)  # fmt: skip
        mark = len(collector.events)
        audio = build_wav(silence(800)) if step.audio else synthesize(step.say or "", detect_language(step.say or "", default=session.language))
        for i in range(0, len(audio), 3200):
            await session.handle_audio(audio[i : i + 3200])
        await session.handle_message('{"type": "audio.end"}')
        await session.drain()
        new = collector.events[mark:]
        turn = next((e for e in new if e["type"] == "agent.reply"), None)
        done = next((e for e in reversed(new) if e["type"] == "turn.done"), None)
        sr.reprompted = bool(done and done["kind"] == "prompt")
        sr.filler = any(e["type"] == "tts.start" and e["kind"] == "filler" for e in new)
        sr.voice = done["metrics"] if done else None
        if turn is not None:
            _from_turn(sr, turn)
        elif done is not None:
            sr.reply = done.get("text")
        sr.checks += check_turn(step.expect, turn, sr.reprompted, sr.filler)
        return sr

    # ------------------------------------------------------------------ conversation level
    async def _conversation_checks(self, client: httpx.AsyncClient, sc: Scenario, res: ScenarioResult) -> None:
        transcript = (await client.get(f"/v1/conversations/{res.conversation_id}")).json()
        res.escalated = transcript["status"] == "escalated"
        if res.escalated != sc.expect_escalation:
            category = "missed_escalation" if sc.expect_escalation else "false_escalation"
            res.checks.append(Check("escalation", False, category, sc.expect_escalation, res.escalated))
        else:
            res.checks.append(Check("escalation", True, "escalation", sc.expect_escalation, res.escalated))
        for turn in transcript["traces"]:
            for span in turn["spans"]:
                if span["kind"] == "llm":
                    res.llm_calls += 1
                for call in span["output"].get("http", []) if span["kind"] in ("tool", "approval") else []:
                    res.http["calls"] += 1
                    res.http["retried"] += call["attempts"] > 1
                    res.http["replayed"] += bool(call.get("replayed"))

        sys = sc.systems
        for cust, want in sys.cards.items():
            cards = (await client.get(f"/mock/payments/customers/{cust}/cards")).json()
            got = {c["last4"]: c["status"] for c in cards if c["last4"] in want}
            res.checks.append(Check(f"cards {cust}", got == want, "side_effect_mismatch", want, got))
        counts = {
            "tickets": ("/mock/helpdesk/tickets", sys.tickets),
            "callbacks": ("/mock/scheduling/callbacks", sys.callbacks),
        }
        for name, (url, want_by_cust) in counts.items():
            for cust, want in want_by_cust.items():
                got = len((await client.get(url, params={"customer_id": cust})).json())
                res.checks.append(Check(f"{name} {cust}", got == want, "side_effect_mismatch", want, got))
        for cust, want in sys.disputes.items():
            got = len((await client.get(f"/mock/payments/customers/{cust}/disputes")).json())
            res.checks.append(Check(f"disputes {cust}", got == want, "side_effect_mismatch", want, got))
        if sys.pending_approvals is not None:
            pending = [a for a in (await client.get("/v1/approvals")).json() if a["conversation_id"] == res.conversation_id]
            got = len(pending)
            res.checks.append(Check("pending approvals", got == sys.pending_approvals, "side_effect_mismatch", sys.pending_approvals, got))
