"""Speech-to-text and text-to-speech providers (see base.py for the interfaces)."""

from voiceops.config import Settings
from voiceops.retry import RetryPolicy
from voiceops.speech.base import AudioInput, SpeechError, STTProvider, Transcript, TTSProvider


def build_stt(settings: Settings) -> STTProvider:
    if settings.stt_provider == "mock":
        from voiceops.speech.mock import MockSTT

        return MockSTT(latency_ms=settings.mock_stt_latency_ms)
    if settings.stt_provider == "whisper_local":
        from voiceops.speech.whisper import FasterWhisperSTT

        return FasterWhisperSTT(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    if settings.stt_provider == "whisper_http":
        from voiceops.speech.whisper import WhisperHTTPSTT

        key = settings.whisper_api_key.get_secret_value() if settings.whisper_api_key else None
        policy = RetryPolicy(max_attempts=settings.speech_max_attempts, attempt_timeout_s=settings.stt_timeout_s,
                             deadline_s=settings.stt_timeout_s * 1.5)  # fmt: skip
        return WhisperHTTPSTT(settings.whisper_http_url, settings.whisper_http_model, key, policy)
    raise ValueError(f"Unknown STT_PROVIDER: {settings.stt_provider!r}")


def build_tts(settings: Settings) -> TTSProvider:
    if settings.tts_provider == "mock":
        from voiceops.speech.mock import MockTTS

        return MockTTS(latency_ms=settings.mock_tts_latency_ms)
    if settings.tts_provider == "azure":
        from voiceops.speech.azure import AzureTTS

        if settings.azure_speech_key is None:
            raise ValueError("TTS_PROVIDER=azure needs AZURE_SPEECH_KEY in the environment")
        policy = RetryPolicy(max_attempts=settings.speech_max_attempts, attempt_timeout_s=settings.tts_timeout_s,
                             deadline_s=settings.tts_timeout_s * 1.5)  # fmt: skip
        return AzureTTS(
            settings.azure_speech_key.get_secret_value(),
            settings.azure_speech_region,
            {"tr": settings.azure_tts_voice_tr, "en": settings.azure_tts_voice_en},
            settings.azure_tts_format,
            policy,
        )
    raise ValueError(f"Unknown TTS_PROVIDER: {settings.tts_provider!r}")


__all__ = ["AudioInput", "STTProvider", "SpeechError", "TTSProvider", "Transcript", "build_stt", "build_tts"]
