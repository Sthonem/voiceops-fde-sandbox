"""Eval report: JSON for machines (diffing, dashboards), Markdown for people."""

import json
from dataclasses import asdict
from typing import Any

from voiceops.evals.metrics import compute_metrics
from voiceops.evals.runner import ScenarioResult

HEADLINE_LABELS = [
    ("scenario_pass_rate", "Scenario pass rate", "pct"),
    ("intent_accuracy", "Intent accuracy", "pct"),
    ("tool_selection_accuracy", "Tool selection accuracy", "pct"),
    ("tool_call_success_rate", "Tool-call success rate", "pct"),
    ("escalation_accuracy", "Escalation accuracy", "pct"),
    ("escalation_precision", "Escalation precision", "pct"),
    ("escalation_recall", "Escalation recall", "pct"),
    ("avg_latency_ms", "Avg turn latency (no injected faults)", "ms"),
    ("p95_latency_ms", "p95 turn latency (no injected faults)", "ms"),
    ("voice_p95_first_audio_ms", "Voice p95 time to first audio", "ms"),
    ("avg_cost_per_conversation_usd", "Avg cost per conversation", "usd"),
    ("p95_cost_per_conversation_usd", "p95 cost per conversation", "usd"),
    ("regressions", "Failing scenarios (excl. known limits)", "int"),
]


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "usd":
        return f"${value:.4f}"
    return str(value)


def build_report(runs: list[list[ScenarioResult]], meta: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = compute_metrics(runs)
    report = {"meta": meta, "metrics": metrics, "results": [asdict(r) | {"passed": r.passed} for run in runs for r in run]}
    if baseline is not None:
        report["comparison"] = compare(report, baseline)
    return report


def slim(report: dict[str, Any]) -> dict[str, Any]:
    """What a committed baseline needs: metrics and pass/fail per scenario run."""
    results = [{"id": r["id"], "run": r["run"], "passed": r["passed"], "known_limit": r["known_limit"]} for r in report["results"]]
    return {"meta": report["meta"], "metrics": report["metrics"], "results": results}


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def passing(rep: dict[str, Any]) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for r in rep["results"]:
            out[r["id"]] = out.get(r["id"], True) and r["passed"]
        return out

    now, before = passing(current), passing(baseline)
    deltas = {}
    for key, _, _ in HEADLINE_LABELS:
        a, b = current["metrics"]["headline"].get(key), baseline["metrics"]["headline"].get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas[key] = {"baseline": b, "current": a, "delta": round(a - b, 6)}
    return {
        "baseline": baseline["meta"].get("label", "baseline"),
        "regressed": sorted(k for k in now if not now[k] and before.get(k)),
        "fixed": sorted(k for k in now if now[k] and before.get(k) is False),
        "new": sorted(k for k in now if k not in before),
        "deltas": deltas,
    }


def _short(value: Any, limit: int = 90) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(report: dict[str, Any]) -> str:
    meta, m = report["meta"], report["metrics"]
    results = report["results"]
    lines: list[str] = []
    add = lines.append

    add(f"# Eval report: {meta['label']}")
    add("")
    add(f"Generated {meta['generated_at']} · LLM `{meta['llm_provider']}` (`{meta['llm_model']}`) · "
        f"STT `{meta['stt_provider']}` · TTS `{meta['tts_provider']}` · "
        f"{m['scenarios']['unique']} scenarios × {m['scenarios']['runs']} run(s)")  # fmt: skip
    add("")
    if meta.get("notes"):
        for note in meta["notes"]:
            add(f"> {note}")
        add("")

    add("## Headline")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    for key, label, kind in HEADLINE_LABELS:
        add(f"| {label} | {fmt(m['headline'].get(key), kind)} |")
    add("")
    sc = m["scenarios"]
    add(f"Consistency: {sc['pass_all_runs']} of {sc['unique']} scenarios passed in every run (pass^k = {fmt(sc['pass_hat_k'], 'pct')}).")
    add("")

    if comp := report.get("comparison"):
        add(f"## Compared with {comp['baseline']}")
        add("")
        add(f"Regressed: {', '.join(comp['regressed']) or 'none'} · Fixed: {', '.join(comp['fixed']) or 'none'} · New: {', '.join(comp['new']) or 'none'}")
        add("")
        add("| Metric | Baseline | Current | Δ |")
        add("|---|---|---|---|")
        kinds = {k: kind for k, _, kind in HEADLINE_LABELS}
        labels = {k: label for k, label, _ in HEADLINE_LABELS}
        for key, d in comp["deltas"].items():
            add(f"| {labels[key]} | {fmt(d['baseline'], kinds[key])} | {fmt(d['current'], kinds[key])} | {d['delta']:+.4g} |")
        add("")

    add("## Breakdown")
    add("")
    add("| Slice | Scenarios | Passed | Pass rate |")
    add("|---|---|---|---|")
    for group in ("by_language", "by_channel"):
        for name, row in sc[group].items():
            add(f"| {group.removeprefix('by_')}: {name} | {row['n']} | {row['passed']} | {fmt(row['pass_rate'], 'pct')} |")
    add("")
    add("| Intent | Labelled turns | Correct | Recall |")
    add("|---|---|---|---|")
    for intent, row in m["intent"]["per_intent"].items():
        add(f"| {intent} | {row['support']} | {row['correct']} | {fmt(row['recall'], 'pct')} |")
    add("")
    if m["intent"]["confusions"]:
        add("Intent confusions: " + ", ".join(f"{c['expected']} → {c['actual']} ×{c['count']}" for c in m["intent"]["confusions"]))
        add("")
    t, e = m["tools"], m["escalation"]
    add(f"Tool calls: {t['calls']} ({', '.join(f'{k} {v}' for k, v in t['by_status'].items())}). "
        f"Integration HTTP calls: {t['http_calls']}, retried {t['http_retried']}, replayed from an idempotency key {t['http_replayed']}.")  # fmt: skip
    add("")
    add(f"Escalation confusion matrix: TP {e['tp']} · FP {e['fp']} · FN {e['fn']} · TN {e['tn']}.")
    add("")
    lat, cost = m["latency_ms"], m["cost"]
    add("| Distribution | n | avg | p50 | p95 | max |")
    add("|---|---|---|---|---|---|")
    for name, row, kind in (
        ("Turn latency", lat["turn"], "ms"),
        ("Voice: time to first audio", lat["voice_first_audio"], "ms"),
        ("Voice: end of speech → playback done", lat["voice_total"], "ms"),
        ("Turn latency, injected faults", lat["turn_with_faults"], "ms"),
        ("Voice: time to first audio, injected faults", lat["voice_first_audio_with_faults"], "ms"),
        ("Cost per conversation", cost["per_conversation_usd"], "usd"),
        ("Input tokens per conversation", cost["input_tokens_per_conversation"], "int"),
        ("Output tokens per conversation", cost["output_tokens_per_conversation"], "int"),
    ):
        if row:
            vals = [fmt(row[k], kind) if kind != "int" else f"{row[k]:.0f}" for k in ("avg", "p50", "p95", "max")]
            add(f"| {name} | {row['n']} | " + " | ".join(vals) + " |")
    add("")
    add(f"LLM calls: {m['llm']['calls']} ({m['llm']['calls_per_turn'] or 0:.2f} per turn; turns handled by guards or templates make none). "
        f"Total estimated cost: ${cost['total_usd']:.4f}{' (token counts estimated)' if cost['estimated'] else ''}.")  # fmt: skip
    add("")

    add("## Failure-mode report")
    add("")
    f = m["failures"]
    if f["by_category"]:
        add("| Failure category | Failed checks |")
        add("|---|---|")
        for cat, n in f["by_category"].items():
            add(f"| {cat} | {n} |")
        add("")
    failing = [r for r in results if not r["passed"]]
    if failing:
        add("### Failed checks")
        add("")
        add("| Scenario | Step | Category | Check | Expected | Actual |")
        add("|---|---|---|---|---|---|")
        for r in failing:
            tag = " (known limit)" if r["known_limit"] else ""
            for step in r["steps"]:
                for c in step["checks"]:
                    if not c["passed"]:
                        add(f"| {r['id']}{tag} | {step['index'] + 1}: {_short(step['input'], 40)} | {c['category']} | "
                            f"{_short(c['name'], 40)} | {_short(c['expected'], 60)} | {_short(c['actual'])} |")  # fmt: skip
            for c in r["checks"]:
                if not c["passed"]:
                    add(f"| {r['id']}{tag} | conversation | {c['category']} | {_short(c['name'], 40)} | {_short(c['expected'], 60)} | {_short(c['actual'])} |")
        add("")
    else:
        add("No failed checks.")
        add("")
    if f["known_limits"]:
        add("### Known limits")
        add("")
        add("| Scenario | Why it is expected to fail | Passed runs |")
        add("|---|---|---|")
        for sid, k in f["known_limits"].items():
            note = " — **now passing: remove the known_limit flag**" if k["passed_runs"] == k["runs"] else ""
            add(f"| {sid} | {k['reason']} | {k['passed_runs']}/{k['runs']}{note} |")
        add("")

    coverage: dict[str, list[tuple[str, bool]]] = {}
    for r in results:
        if r["run"] == 0:
            for mode in r["failure_modes"]:
                coverage.setdefault(mode, []).append((r["id"], r["passed"]))
    if coverage:
        add("### Failure modes exercised")
        add("")
        add("| Failure mode | Scenarios | Handled |")
        add("|---|---|---|")
        for mode, items in sorted(coverage.items()):
            ok = sum(p for _, p in items)
            add(f"| {mode} | {', '.join(i for i, _ in items)} | {ok}/{len(items)} |")
        add("")

    add("## Scenarios")
    add("")
    add("| Scenario | Lang | Channel | Steps | Result | Escalated | Turn latency (avg) | Cost |")
    add("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r["run"] != 0:
            continue
        lats = [s["latency_ms"] for s in r["steps"] if s["latency_ms"] is not None]
        avg = sum(lats) / len(lats) if lats else None
        verdict = "pass" if r["passed"] else ("known limit" if r["known_limit"] else "**FAIL**")
        cost_v = sum(s["cost_usd"] for s in r["steps"])
        add(f"| {r['id']} | {r['language']} | {r['channel']} | {len(r['steps'])} | {verdict} | "
            f"{'yes' if r['escalated'] else 'no'} | {fmt(avg, 'ms')} | {fmt(cost_v, 'usd')} |")  # fmt: skip
    add("")
    add("## Definitions")
    add("")
    add("- **Scenario pass**: every step check, the escalation label and every side-effect check pass.")
    add("- **Intent accuracy**: the turn's reported intent equals the label (labelled turns only).")
    add("- **Tool selection accuracy**: tool names, in order, equal the label (labelled turns only).")
    add("- **Tool-call success**: the call ended as the scenario expects; without a status label, "
        "`ok`, `needs_confirmation` and `pending_approval` count as success.")  # fmt: skip
    add("- **Escalation**: conversation-level label vs. whether the conversation ended with a human.")
    add("- **Turn latency**: server-side agent time per turn (router, LLM, tools, persistence). "
        "**Voice time to first audio**: end of caller speech to first audio byte sent. Scenarios with "
        "injected faults (timeouts, retries, slow backends) are reported in their own rows.")  # fmt: skip
    add("- **Cost**: estimated from token usage and list prices (`voiceops/pricing.py`).")
    add("")
    return "\n".join(lines)
