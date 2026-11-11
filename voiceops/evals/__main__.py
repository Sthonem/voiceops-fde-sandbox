"""Run the golden eval set.

    python -m voiceops.evals                                  # mock baseline, report in reports/
    python -m voiceops.evals --repeat 3                       # consistency (pass^k) + more latency samples
    python -m voiceops.evals --provider anthropic             # Claude (needs ANTHROPIC_API_KEY; costs money)
    python -m voiceops.evals --baseline evals/baselines/mock.json      # regressions vs. the accepted baseline
    python -m voiceops.evals --save-baseline evals/baselines/mock.json # accept the current run
    python -m voiceops.evals --tag voice --id tr_card_block_full
    python -m voiceops.evals --min intent_accuracy=0.9 --max p95_latency_ms=3000 --no-regressions   # CI gate

Exit code 1 when a gate fails, 2 on a usage problem.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from voiceops.config import Settings
from voiceops.evals.metrics import check_thresholds
from voiceops.evals.report import build_report, render_markdown, slim
from voiceops.evals.runner import EvalRunner, ScenarioResult
from voiceops.evals.scenarios import DEFAULT_DIR, load_scenarios


def _bounds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key.strip()] = float(value)
    return out


def _portable_path(path: Path) -> str:
    """Path relative to the working directory, so reports don't record local home paths."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the VoiceOps golden eval set")
    p.add_argument("--scenarios", type=Path, default=DEFAULT_DIR, help="YAML file or directory")
    p.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--tag", action="append", default=[], help="only scenarios with this tag (repeatable)")
    p.add_argument("--id", action="append", default=[], help="only these scenario ids (repeatable)")
    p.add_argument("--out", type=Path, default=Path("reports"), help="directory for the JSON and Markdown report")
    p.add_argument("--label", help="report label (default: provider + timestamp)")
    p.add_argument("--md", type=Path, help="also write the Markdown report to this path")
    p.add_argument("--baseline", type=Path, help="previous JSON report (or saved baseline) to compare with")
    p.add_argument("--save-baseline", type=Path, help="write this run as the new baseline (metrics + pass/fail)")
    p.add_argument("--min", action="append", default=[], metavar="METRIC=VALUE")
    p.add_argument("--max", action="append", default=[], metavar="METRIC=VALUE")
    p.add_argument("--no-regressions", action="store_true", help="fail if any scenario fails that is not a known limit")
    args = p.parse_args(argv)

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("--provider anthropic needs ANTHROPIC_API_KEY in the environment (each run makes real, billed requests).")
        return 2

    scenarios = load_scenarios(args.scenarios)
    if args.tag:
        scenarios = [s for s in scenarios if set(args.tag) & set(s.tags)]
    if args.id:
        scenarios = [s for s in scenarios if s.id in args.id]
    if not scenarios:
        print("no scenarios selected")
        return 2

    settings = Settings(llm_provider=args.provider, log_level="WARNING")
    now = datetime.now(UTC)
    label = args.label or f"{args.provider}-{now:%Y%m%d-%H%M%S}"

    def progress(r: ScenarioResult) -> None:
        mark = "ok  " if r.passed else ("lim " if r.known_limit else "FAIL")
        print(f"  {mark} {r.id}  ({r.duration_ms:.0f} ms)", flush=True)

    print(f"running {len(scenarios)} scenarios × {args.repeat} with {args.provider} …")
    runs = await EvalRunner(settings).run(scenarios, repeat=args.repeat, on_result=progress)

    from voiceops.agent.llm import build_llm_provider

    llm = build_llm_provider(settings)
    notes = []
    if args.provider == "mock":
        notes.append(
            "Mock LLM, mock STT/TTS and in-process mock systems: latencies measure the pipeline "
            "(routing, guards, tools, persistence), not model or speech inference. Costs price the "
            "mock's estimated tokens as Claude Opus 5.5 for an order of magnitude only."
        )
    meta = {
        "label": label,
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "llm_provider": settings.llm_provider,
        "llm_model": llm.model,
        "effort": settings.anthropic_effort if args.provider == "anthropic" else None,
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "scenario_source": _portable_path(args.scenarios),
        "notes": notes,
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report = build_report(runs, meta, baseline)
    markdown = render_markdown(report)

    args.out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.out / f"eval-{label}.json", args.out / f"eval-{label}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    md_path.write_text(markdown)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown)
    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(json.dumps(slim(report), ensure_ascii=False, indent=1, default=str))

    head = report["metrics"]["headline"]
    print(
        f"\npass {head['scenario_pass_rate']:.1%} · intent {head['intent_accuracy']:.1%} · "
        f"tools {head['tool_selection_accuracy']:.1%} · escalation {head['escalation_accuracy']:.1%} · "
        f"p95 {head['p95_latency_ms']} ms · cost/conv ${head['avg_cost_per_conversation_usd']:.4f}"
    )
    print(f"report: {md_path}  (json: {json_path})")

    problems = check_thresholds(report["metrics"], _bounds(args.min), _bounds(args.max))
    if args.no_regressions and (reg := report["metrics"]["failures"]["regressions"]):
        problems.append(f"failing scenarios: {', '.join(reg)}")
    if args.no_regressions and (regressed := report.get("comparison", {}).get("regressed")):
        problems.append(f"passed in the baseline, failing now: {', '.join(regressed)}")
    for problem in problems:
        print(f"GATE FAILED: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
