"""Metrics over eval runs. Definitions are part of the report, so numbers can be
read without opening this file:

* scenario pass rate     all checks of a scenario pass (steps, escalation, side effects)
* pass^k                 a scenario passes in every one of k repeated runs (consistency)
* intent accuracy        turns whose reported intent equals the label / labelled turns
* tool selection         turns whose tool names, in order, equal the label / labelled turns
* tool-call success      tool calls that ended as expected / all tool calls. Without a
                         status label, ok, needs_confirmation and pending_approval count as
                         success; error and blocked do not
* escalation             conversation-level: did it end with a human when it should have?
* latency                agent turn latency (API), and for voice: end of caller speech to
                         first audio byte. Scenarios with injected faults (timeouts, retries)
                         are reported separately so they don't distort the headline
* cost per conversation  sum of the estimated cost of every turn
"""

import math
from collections import Counter, defaultdict
from typing import Any

from voiceops.evals.runner import SUCCESS_STATUSES, ScenarioResult


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "avg": round(sum(values) / len(values), 4),
        "p50": round(percentile(values, 0.5) or 0, 4),
        "p95": round(percentile(values, 0.95) or 0, 4),
        "max": round(max(values), 4),
    }


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _call_outcomes(results: list[ScenarioResult]) -> tuple[int, int, Counter[str]]:
    total = ok = 0
    by_status: Counter[str] = Counter()
    for r in results:
        for s in r.steps:
            statuses = [t["status"] for t in s.tools]
            by_status.update(statuses)
            total += len(statuses)
            if s.status_expectation is not None:
                ok += sum(1 for i, st in enumerate(statuses) if i < len(s.status_expectation) and s.status_expectation[i] == st)
            else:
                ok += sum(1 for st in statuses if st in SUCCESS_STATUSES)
    return total, ok, by_status


def compute_metrics(runs: list[list[ScenarioResult]]) -> dict[str, Any]:
    results = [r for run in runs for r in run]
    steps = [s for r in results for s in r.steps]
    turns = [s for s in steps if s.kind != "supervisor" and s.latency_ms is not None]
    clean = [r for r in results if not r.faulted]
    faulted = [r for r in results if r.faulted]

    def turn_latencies(rs: list[ScenarioResult]) -> list[float]:
        return [s.latency_ms for r in rs for s in r.steps if s.kind != "supervisor" and s.latency_ms is not None]

    def voice_metric(rs: list[ScenarioResult], key: str) -> list[float]:
        return [s.voice[key] for r in rs for s in r.steps if s.voice and s.voice.get(key) is not None and not s.reprompted]

    # scenarios
    by_id: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_id[r.id].append(r)
    passed = sum(r.passed for r in results)
    pass_all = sum(all(x.passed for x in rs) for rs in by_id.values())

    def pass_rate_by(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[ScenarioResult]] = defaultdict(list)
        for r in results:
            groups[getattr(r, key)].append(r)
        return {k: {"n": len(v), "passed": sum(x.passed for x in v), "pass_rate": _ratio(sum(x.passed for x in v), len(v))} for k, v in sorted(groups.items())}

    # intents
    labelled = [s for s in steps if s.expected_intent is not None and s.intent is not None]
    per_intent: dict[str, dict[str, Any]] = {}
    for s in labelled:
        row = per_intent.setdefault(s.expected_intent or "", {"support": 0, "correct": 0})
        row["support"] += 1
        row["correct"] += s.intent == s.expected_intent
    for row in per_intent.values():
        row["recall"] = _ratio(row["correct"], row["support"])
    confusion = Counter((s.expected_intent, s.intent) for s in labelled if s.intent != s.expected_intent)

    # tools
    tool_labelled = [s for s in steps if s.tool_expectation is not None and s.latency_ms is not None]
    tool_correct = sum([t["name"] for t in s.tools] == s.tool_expectation for s in tool_labelled)
    calls, calls_ok, by_status = _call_outcomes(results)

    # escalation (conversation level)
    tp = sum(r.expect_escalation and r.escalated for r in results)
    tn = sum(not r.expect_escalation and not r.escalated for r in results)
    fp = sum(not r.expect_escalation and r.escalated for r in results)
    fn = sum(r.expect_escalation and not r.escalated for r in results)

    # latency (clean scenarios), cost
    clean_turns = summarize(turn_latencies(clean)) or {}
    voice_first = voice_metric(clean, "first_audio_ms")
    costs = [r.cost_usd for r in results]
    tokens_in = [sum(s.input_tokens for s in r.steps) for r in results]
    tokens_out = [sum(s.output_tokens for s in r.steps) for r in results]
    llm_turns = len(turns)

    failures = Counter(c.category for r in results for _, c in r.failures)
    known = {
        sid: {"reason": rs[0].known_limit, "passed_runs": sum(x.passed for x in rs), "runs": len(rs)}
        for sid, rs in by_id.items()
        if rs[0].known_limit
    }
    regressions = sorted({r.id for r in results if not r.passed and not r.known_limit})

    headline = {
        "scenario_pass_rate": _ratio(passed, len(results)),
        "intent_accuracy": _ratio(sum(s.intent == s.expected_intent for s in labelled), len(labelled)),
        "tool_selection_accuracy": _ratio(tool_correct, len(tool_labelled)),
        "tool_call_success_rate": _ratio(calls_ok, calls),
        "escalation_accuracy": _ratio(tp + tn, len(results)),
        "escalation_precision": _ratio(tp, tp + fp),
        "escalation_recall": _ratio(tp, tp + fn),
        "avg_latency_ms": clean_turns.get("avg"),
        "p95_latency_ms": clean_turns.get("p95"),
        "voice_p95_first_audio_ms": (summarize(voice_first) or {}).get("p95"),
        "avg_cost_per_conversation_usd": (summarize(costs) or {}).get("avg"),
        "p95_cost_per_conversation_usd": (summarize(costs) or {}).get("p95"),
        "regressions": len(regressions),
    }
    return {
        "headline": headline,
        "scenarios": {
            "runs": len(runs),
            "unique": len(by_id),
            "executed": len(results),
            "passed": passed,
            "pass_rate": _ratio(passed, len(results)),
            "pass_all_runs": pass_all,
            "pass_hat_k": _ratio(pass_all, len(by_id)),
            "by_language": pass_rate_by("language"),
            "by_channel": pass_rate_by("channel"),
        },
        "intent": {
            "labelled_turns": len(labelled),
            "accuracy": headline["intent_accuracy"],
            "per_intent": dict(sorted(per_intent.items())),
            "confusions": [{"expected": e, "actual": a, "count": n} for (e, a), n in confusion.most_common()],
        },
        "tools": {
            "labelled_turns": len(tool_labelled),
            "selection_accuracy": headline["tool_selection_accuracy"],
            "calls": calls,
            "call_success_rate": headline["tool_call_success_rate"],
            "by_status": dict(by_status.most_common()),
            "http_calls": sum(r.http["calls"] for r in results),
            "http_retried": sum(r.http["retried"] for r in results),
            "http_replayed": sum(r.http["replayed"] for r in results),
        },
        "escalation": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": headline["escalation_accuracy"],
            "precision": headline["escalation_precision"],
            "recall": headline["escalation_recall"],
        },  # fmt: skip
        "latency_ms": {
            "turn": summarize(turn_latencies(clean)),
            "voice_first_audio": summarize(voice_first),
            "voice_total": summarize(voice_metric(clean, "total_ms")),
            "turn_with_faults": summarize(turn_latencies(faulted)),
            "voice_first_audio_with_faults": summarize(voice_metric(faulted, "first_audio_ms")),
        },
        "llm": {
            "calls": sum(r.llm_calls for r in results),
            "calls_per_turn": _ratio(sum(r.llm_calls for r in results), llm_turns),
        },
        "cost": {
            "estimated": any(s.estimated for s in steps),
            "per_conversation_usd": summarize(costs),
            "total_usd": round(sum(costs), 6),
            "input_tokens_per_conversation": summarize([float(t) for t in tokens_in]),
            "output_tokens_per_conversation": summarize([float(t) for t in tokens_out]),
        },
        "failures": {"by_category": dict(failures.most_common()), "regressions": regressions, "known_limits": known},
    }


def check_thresholds(metrics: dict[str, Any], minimums: dict[str, float], maximums: dict[str, float]) -> list[str]:
    """Return human-readable violations for a CI gate."""
    head = metrics["headline"]
    problems = []
    for key, bound in minimums.items():
        value = head.get(key)
        if value is None or value < bound:
            problems.append(f"{key} = {value} < {bound}")
    for key, bound in maximums.items():
        value = head.get(key)
        if value is not None and value > bound:
            problems.append(f"{key} = {value} > {bound}")
    return problems
