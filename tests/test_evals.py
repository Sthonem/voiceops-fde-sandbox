"""The eval harness: scenario schema, checks, metrics, report, CLI, and the golden
set itself run against the mock baseline as a regression gate."""

import json

import pytest
from pydantic import ValidationError

from voiceops.evals import EvalRunner, check_thresholds, compute_metrics, load_scenarios
from voiceops.evals.__main__ import main as cli_main
from voiceops.evals.metrics import percentile, summarize
from voiceops.evals.report import build_report, compare, render_markdown, slim
from voiceops.evals.runner import Check, ScenarioResult, StepResult, check_turn
from voiceops.evals.scenarios import Scenario, TurnExpect


# --------------------------------------------------------------------------- golden set
def test_golden_set_is_large_bilingual_and_covers_the_brief():
    scenarios = load_scenarios()
    assert len(scenarios) >= 20
    assert {s.language for s in scenarios} == {"tr", "en"}
    assert {s.channel for s in scenarios} == {"text", "voice"}
    assert sum(s.expect_escalation for s in scenarios) >= 5 and sum(not s.expect_escalation for s in scenarios) >= 5
    modes = {m for s in scenarios for m in s.failure_modes}
    assert {"integration_5xx", "lost_response_after_commit", "sensitive_topic", "verification_failed", "no_speech"} <= modes
    tags = {t for s in scenarios for t in s.tags}
    assert {"payment_status", "policy_question", "card_block", "dispute_transaction", "schedule_callback",
            "create_ticket", "human_escalation", "paraphrase", "resilience", "voice"} <= tags  # fmt: skip


def test_scenario_schema_is_strict():
    base = {"id": "x", "title": "t", "language": "tr", "steps": [{"say": "hi"}]}
    Scenario.model_validate(base)
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "steps": [{"say": "hi", "expect": {"intnet": "typo"}}]})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "steps": [{"say": "hi", "supervisor": "approve"}]})
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "steps": [{"audio": "silence"}]})  # audio needs channel voice
    with pytest.raises(ValidationError):
        Scenario.model_validate({**base, "id": "Has Spaces"})


def test_duplicate_ids_are_rejected(tmp_path):
    item = {"id": "dup", "title": "t", "language": "en", "steps": [{"say": "hi"}]}
    (tmp_path / "a.yaml").write_text(json.dumps([item]))
    (tmp_path / "b.yaml").write_text(json.dumps([item]))
    with pytest.raises(ValueError, match="duplicate"):
        load_scenarios(tmp_path)


# --------------------------------------------------------------------------- checks
TURN = {
    "reply": "Sonu 7710 olan kredi kartınızı bloke ettim.",
    "intent": "card_block",
    "tool_calls": [{"name": "block_card", "status": "ok", "latency_ms": 2.0, "arguments": {}}],
    "status": "active",
    "awaiting_confirmation": None,
    "verified": True,
    "customer_id": "C-1001",
    "language": "tr",
    "approval_ids": [],
    "citations": [],
}


def test_check_turn_passes_and_fails_with_categories():
    ok = check_turn(TurnExpect(intent="card_block", tools=["block_card"], reply_contains=["bloke ettim"], verified=True), TURN)
    assert all(c.passed for c in ok)
    bad = check_turn(TurnExpect(intent="payment_status", tools=[], reply_not_contains=["7710"], escalated=True, citations=True), TURN)
    assert {c.category for c in bad if not c.passed} == {
        "intent_mismatch", "wrong_tools", "forbidden_content", "escalation_timing", "missing_citation",
    }  # fmt: skip


def test_unlabelled_tool_errors_are_failures_but_labelled_ones_are_not():
    turn = {**TURN, "tool_calls": [{"name": "lookup_customer", "status": "error", "latency_ms": 1.0, "arguments": {}}]}
    assert [c.category for c in check_turn(TurnExpect(), turn)] == ["tool_failure"]
    assert all(c.passed for c in check_turn(TurnExpect(tool_statuses=["error"]), turn))


def test_voice_checks():
    assert check_turn(TurnExpect(reprompt=True), None, reprompted=True)[0].passed
    missing = check_turn(TurnExpect(tools=["lookup_customer"]), None, reprompted=False)
    assert missing[0].category == "runner_error"
    assert not check_turn(TurnExpect(filler=True), TURN, filler=False)[0].passed


# --------------------------------------------------------------------------- metrics
def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.95) == 95.0 and percentile(values, 0.5) == 50.0
    assert percentile([], 0.95) is None
    assert summarize([1.0, 3.0]) == {"n": 2, "avg": 2.0, "p50": 1.0, "p95": 3.0, "max": 3.0}


def _result(sid, *, escalate, escalated, passed=True, faulted=False, latency=10.0, cost=0.01, statuses=("ok",), labels=None):
    step = StepResult(0, "say", "x", intent="a", expected_intent="a" if passed else "b", latency_ms=latency, cost_usd=cost,
                      tools=[{"name": "t", "status": s, "latency_ms": 1.0} for s in statuses],
                      tool_expectation=["t"] * len(statuses), status_expectation=labels)  # fmt: skip
    if not passed:
        step.checks.append(Check("intent", False, "intent_mismatch", "b", "a"))
    return ScenarioResult(sid, sid, "tr", "text", [], [], None, escalate, faulted=faulted, escalated=escalated, steps=[step])


def test_metrics_escalation_tools_latency_and_cost():
    runs = [[
        _result("a", escalate=True, escalated=True),
        _result("b", escalate=True, escalated=False, passed=False),
        _result("c", escalate=False, escalated=True),
        _result("d", escalate=False, escalated=False, statuses=("error",), labels=["error"]),
        _result("e", escalate=False, escalated=False, statuses=("blocked",)),
        _result("f", escalate=False, escalated=False, faulted=True, latency=3000.0),
    ]]  # fmt: skip
    m = compute_metrics(runs)
    assert m["escalation"] | {} == {"tp": 1, "fp": 1, "fn": 1, "tn": 3, "accuracy": 0.6667, "precision": 0.5, "recall": 0.5}
    assert m["tools"]["call_success_rate"] == round(5 / 6, 4)  # the labelled error counts, the unlabelled block doesn't
    assert m["headline"]["p95_latency_ms"] == 10.0  # the faulted 3 s turn is reported separately
    assert m["latency_ms"]["turn_with_faults"]["max"] == 3000.0
    assert m["cost"]["per_conversation_usd"]["avg"] == 0.01
    assert m["intent"]["accuracy"] == round(5 / 6, 4) and m["failures"]["regressions"] == ["b"]


def test_thresholds_and_comparison():
    m = {"headline": {"intent_accuracy": 0.8, "p95_latency_ms": 900.0}}
    assert check_thresholds(m, {"intent_accuracy": 0.9}, {"p95_latency_ms": 1000}) == ["intent_accuracy = 0.8 < 0.9"]
    assert check_thresholds(m, {}, {"p95_latency_ms": 500}) == ["p95_latency_ms = 900.0 > 500"]
    current = {"meta": {"label": "now"}, "metrics": {"headline": {"intent_accuracy": 0.9}},
               "results": [{"id": "a", "passed": False}, {"id": "b", "passed": True}, {"id": "c", "passed": True}]}  # fmt: skip
    base = {"meta": {"label": "before"}, "metrics": {"headline": {"intent_accuracy": 0.8}},
            "results": [{"id": "a", "passed": True}, {"id": "b", "passed": False}]}  # fmt: skip
    diff = compare(current, base)
    assert (diff["regressed"], diff["fixed"], diff["new"]) == (["a"], ["b"], ["c"])
    assert diff["deltas"]["intent_accuracy"]["delta"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def golden_report():
    import asyncio

    runs = asyncio.run(EvalRunner().run(load_scenarios()))
    meta = {"label": "test", "generated_at": "now", "llm_provider": "mock", "llm_model": "mock-rules-v1",
            "stt_provider": "mock", "tts_provider": "mock", "notes": []}  # fmt: skip
    return build_report(runs, meta)


def test_golden_set_has_no_regressions_on_the_mock_baseline(golden_report):
    """CI gate: everything not marked as a known limit passes."""
    failures = golden_report["metrics"]["failures"]
    assert failures["regressions"] == [], render_markdown(golden_report)


def test_known_limits_still_fail(golden_report):
    """Like a strict xfail: if a known limit starts passing, drop its flag."""
    known = golden_report["metrics"]["failures"]["known_limits"]
    assert known and all(k["passed_runs"] == 0 for k in known.values()), known


def test_headline_metrics_meet_the_bar(golden_report):
    head = golden_report["metrics"]["headline"]
    assert check_thresholds(
        golden_report["metrics"],
        {"intent_accuracy": 0.85, "tool_selection_accuracy": 0.9, "tool_call_success_rate": 0.95,
         "escalation_accuracy": 0.95, "escalation_recall": 1.0},
        {"p95_latency_ms": 500},
    ) == [], head  # fmt: skip
    tools = golden_report["metrics"]["tools"]
    assert tools["http_retried"] >= 4 and tools["http_replayed"] >= 1  # the resilience scenarios really exercised retries


def test_report_renders_every_section(golden_report):
    md = render_markdown(golden_report)
    for heading in ("## Headline", "## Breakdown", "## Failure-mode report", "### Known limits",
                    "### Failure modes exercised", "## Scenarios", "## Definitions"):  # fmt: skip
        assert heading in md
    assert len(json.dumps(slim(golden_report))) < len(json.dumps(golden_report)) / 3


async def test_cli_subset_gate_and_baseline(tmp_path, capsys):
    out = tmp_path / "reports"
    base = tmp_path / "baseline.json"
    code = await cli_main(["--id", "tr_eft_cutoff_policy", "--id", "en_human_request", "--out", str(out),
                           "--label", "t1", "--save-baseline", str(base), "--no-regressions"])  # fmt: skip
    assert code == 0 and (out / "eval-t1.md").exists() and json.loads(base.read_text())["results"]
    code = await cli_main(["--id", "tr_eft_cutoff_policy", "--out", str(out), "--label", "t2",
                           "--baseline", str(base), "--min", "intent_accuracy=1.01"])  # fmt: skip
    assert code == 1 and "GATE FAILED" in capsys.readouterr().out
    assert "## Compared with t1" in (out / "eval-t2.md").read_text()


async def test_cli_refuses_claude_without_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert await cli_main(["--provider", "anthropic", "--out", str(tmp_path)]) == 2
