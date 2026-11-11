"""Golden-set evaluation: scenarios (evals/golden/*.yaml) → runner → metrics → report."""

from voiceops.evals.metrics import check_thresholds, compute_metrics
from voiceops.evals.runner import EvalRunner, ScenarioResult
from voiceops.evals.scenarios import Scenario, load_scenarios

__all__ = ["EvalRunner", "Scenario", "ScenarioResult", "check_thresholds", "compute_metrics", "load_scenarios"]
