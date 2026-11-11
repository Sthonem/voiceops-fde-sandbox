"""Golden scenario format.

A scenario is a scripted conversation with expectations per step and for the
whole conversation. Files live in `evals/golden/*.yaml`; each file holds a list
of scenarios. The schema is strict (unknown keys are errors) so a typo in an
expectation cannot silently turn a check off.

    - id: tr_card_block_full
      title: Stolen card, verified, choose card, confirm
      language: tr
      tags: [card_block, sensitive_action]
      failure_modes: [several_cards_match]
      steps:
        - say: "Kartım çalındı"
          expect: {intent: card_block, tools: [], reply_contains: ["telefon numaranızı"]}
        - say: "Evet"
          expect: {tools: [block_card], tool_statuses: [ok]}
      expect_escalation: false
      systems:
        cards: {C-1001: {"7710": blocked, "4821": active}}
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voiceops.mock_systems.middleware import FaultRuleIn

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "evals" / "golden"


class TurnExpect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str | None = None
    tools: list[str] | None = Field(default=None, description="exact ordered tool names; [] means no tool call")
    tool_statuses: list[str] | None = Field(default=None, description="statuses aligned with tools")
    reply_contains: list[str] = Field(default_factory=list)
    reply_not_contains: list[str] = Field(default_factory=list)
    escalated: bool | None = Field(default=None, description="conversation status after this step")
    awaiting_confirmation: bool | None = None
    verified: bool | None = None
    customer_id: str | None = None
    citations: bool | None = None
    approvals_raised: int | None = None
    language: Literal["tr", "en"] | None = None
    reprompt: bool | None = Field(default=None, description="voice: re-prompted instead of running a turn")
    filler: bool | None = Field(default=None, description="voice: a filler was spoken before the reply")


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    say: str | None = None
    audio: Literal["silence"] | None = None  # voice only
    supervisor: Literal["approve", "reject"] | None = None  # decide the latest approval request
    faults: list[FaultRuleIn] = Field(default_factory=list)  # injected before this step
    expect: TurnExpect = Field(default_factory=TurnExpect)

    @model_validator(mode="after")
    def _one_action(self) -> "Step":
        if sum(x is not None for x in (self.say, self.audio, self.supervisor)) != 1:
            raise ValueError("each step needs exactly one of: say, audio, supervisor")
        return self

    @property
    def kind(self) -> str:
        return "say" if self.say is not None else "audio" if self.audio is not None else "supervisor"


class SystemsExpect(BaseModel):
    """State of the mock systems after the conversation (side effects)."""

    model_config = ConfigDict(extra="forbid")

    cards: dict[str, dict[str, str]] = Field(default_factory=dict)  # customer -> last4 -> status
    tickets: dict[str, int] = Field(default_factory=dict)  # customer -> number of tickets
    disputes: dict[str, int] = Field(default_factory=dict)
    callbacks: dict[str, int] = Field(default_factory=dict)
    pending_approvals: int | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    language: Literal["tr", "en"]
    channel: Literal["text", "voice"] = "text"
    tags: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    known_limit: str | None = Field(default=None, description="why this is expected to fail today")
    faults: list[FaultRuleIn] = Field(default_factory=list)  # active for the whole scenario
    steps: list[Step] = Field(min_length=1)
    expect_escalation: bool = False
    systems: SystemsExpect = Field(default_factory=SystemsExpect)

    @model_validator(mode="after")
    def _channel_rules(self) -> "Scenario":
        if self.channel == "text" and any(s.audio for s in self.steps):
            raise ValueError(f"{self.id}: audio steps need channel: voice")
        return self


def load_scenarios(path: Path | str | None = None) -> list[Scenario]:
    root = Path(path) if path else DEFAULT_DIR
    files = sorted(root.glob("*.yaml")) if root.is_dir() else [root]
    scenarios: list[Scenario] = []
    for f in files:
        try:
            items = yaml.safe_load(f.read_text(encoding="utf-8")) or []
            scenarios += [Scenario.model_validate(raw) for raw in items]
        except (yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"{f.name}: {exc}") from exc
    ids = [s.id for s in scenarios]
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate scenario ids: {dupes}")
    return scenarios
