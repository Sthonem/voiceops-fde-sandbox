"""Token pricing (USD per million tokens) for cost-per-conversation estimates.

Anthropic first-party list prices as of 2026-10. The mock provider is priced as
the configured reference model so offline runs still produce a realistic
"what would this cost on Claude" number (flagged as estimated).
"""

from voiceops.agent.llm.base import LLMUsage

PRICES: dict[str, dict[str, float]] = {
    # model: input, output, cache read, cache write (5-minute TTL = 1.25x input)
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


def estimate_cost_usd(model: str, usage: LLMUsage, reference_model: str = "claude-opus-5-5") -> float:
    price = PRICES.get(model) or PRICES.get(reference_model) or PRICES["claude-opus-5-5"]
    total = (
        usage.input_tokens * price["input"]
        + usage.output_tokens * price["output"]
        + usage.cache_read_tokens * price["cache_read"]
        + usage.cache_write_tokens * price["cache_write"]
    ) / 1_000_000
    return round(total, 6)
