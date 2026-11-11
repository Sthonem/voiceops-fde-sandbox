"""Provider-neutral speech interfaces.

The voice session only depends on these, so mock providers (offline, CI, evals)
and real ones (Whisper, Azure neural TTS) are interchangeable, exactly like the
LLM providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

AudioFormat = Literal["wav", "pcm16"]


class SpeechError(Exception):
    def __init__(self, provider: str, code: str, message: str, retryable: bool = False):
        super().__init__(f"{provider}: {code} {message}")
        self.provider = provider
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass
class AudioInput:
    data: bytes
    format: AudioFormat = "wav"  # pcm16 = raw 16-bit little-endian mono
    sample_rate: int = 16000


@dataclass
class Transcript:
    text: str
    language: str | None
    confidence: float  # 0..1; 0 means nothing intelligible
    provider: str
    audio_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class STTProvider(ABC):
    name: str

    @abstractmethod
    async def transcribe(self, audio: AudioInput, *, language_hint: str | None = None) -> Transcript: ...

    async def aclose(self) -> None:
        return None


class TTSProvider(ABC):
    name: str
    media_type: str  # what the client receives, e.g. audio/wav, audio/mpeg

    @abstractmethod
    def stream(self, text: str, *, language: str) -> AsyncIterator[bytes]:
        """Audio for `text`, yielded in chunks as soon as they are available."""

    async def aclose(self) -> None:
        return None
