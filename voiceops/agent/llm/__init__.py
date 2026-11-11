from voiceops.agent.llm.base import LLMProvider, LLMResult, LLMUsage, ToolSpec
from voiceops.config import Settings


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        from voiceops.agent.llm.mock import MockLLMProvider

        return MockLLMProvider(tz=settings.tz)
    if settings.llm_provider == "anthropic":
        from voiceops.agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            max_tokens=settings.anthropic_max_tokens,
            timeout_s=settings.anthropic_timeout_s,
            eager_tool_streaming=settings.anthropic_eager_tool_streaming,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


__all__ = ["LLMProvider", "LLMResult", "LLMUsage", "ToolSpec", "build_llm_provider"]
