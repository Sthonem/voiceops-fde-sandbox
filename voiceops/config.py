from functools import lru_cache
from zoneinfo import ZoneInfo

from typing import Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./voiceops.db"
    # Create the agent's tables at startup (dev, tests). Docker runs Alembic instead.
    db_auto_create: bool = True
    mock_api_base_url: str = "inprocess"
    # Outbound calls to the customer's systems: per-attempt timeout, total budget
    # per call (retries included) and full-jitter exponential backoff.
    integration_timeout_s: float = 3.0
    integration_deadline_s: float = 6.0
    integration_max_attempts: int = 3
    integration_backoff_base_s: float = 0.2
    integration_backoff_max_s: float = 2.0

    llm_provider: str = "mock"
    anthropic_model: str = "claude-opus-5-5"
    # Voice turns are latency-sensitive and the tasks are routine: low effort first,
    # raise only if the eval shows quality headroom.
    anthropic_effort: str = "low"
    anthropic_max_tokens: int = 16000
    anthropic_timeout_s: float = 30.0
    # Streaming requests (voice) send tools with eager_input_streaming; the
    # registry validates every tool input against its schema before running it.
    anthropic_eager_tool_streaming: bool = True

    # Speech. "mock" providers need no keys and no models (CI, demos).
    stt_provider: str = "mock"  # mock | whisper_local | whisper_http
    tts_provider: str = "mock"  # mock | azure
    stt_timeout_s: float = 10.0
    tts_timeout_s: float = 8.0  # max wait for the next audio chunk
    speech_max_attempts: int = 2
    mock_stt_latency_ms: int = 0
    mock_tts_latency_ms: int = 0
    # whisper_local: faster-whisper in-process; audio never leaves the host
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # whisper_http: any OpenAI-compatible /audio/transcriptions endpoint (self-hosted or cloud)
    whisper_http_url: str = "http://localhost:9000/v1"
    whisper_http_model: str = "Systran/faster-whisper-small"
    whisper_api_key: SecretStr | None = None
    # azure: Azure AI Speech neural TTS (REST, streamed)
    azure_speech_key: SecretStr | None = None
    azure_speech_region: str = "westeurope"
    azure_tts_voice_tr: str = "tr-TR-EmelNeural"
    azure_tts_voice_en: str = "en-US-JennyNeural"
    azure_tts_format: str = "audio-24khz-48kbitrate-mono-mp3"

    # Voice session behaviour
    voice_filler_after_ms: int = 900  # say "one moment" if the agent is still busy
    voice_min_confidence: float = 0.35
    voice_max_utterance_s: float = 30.0
    voice_vad_threshold: float = 500.0  # RMS of 16-bit PCM
    voice_vad_end_silence_ms: int = 700
    voice_greeting: bool = True

    # Required on /v1/approvals and /v1/demo/faults when set (X-API-Key header).
    # Unset = open, for local development only; /health reports which.
    supervisor_api_key: SecretStr | None = None
    demo_enabled: bool = True  # serve /demo and the fault-injection proxy

    policy_dir: str | None = None
    log_json: bool = True
    log_level: str = "INFO"

    agent_max_steps: int = 6
    timezone: str = "Europe/Istanbul"
    seed_on_startup: bool = True

    @field_validator("supervisor_api_key", "azure_speech_key", "whisper_api_key", mode="before")
    @classmethod
    def _empty_secret_is_unset(cls, value: Any) -> Any:
        # docker compose passes unset variables as empty strings
        return None if value == "" else value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache
def get_settings() -> Settings:
    return Settings()
