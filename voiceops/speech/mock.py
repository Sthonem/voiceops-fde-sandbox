"""Offline speech providers.

MockTTS renders real, playable 16 kHz WAV audio (one tone per word, speech-like
timing) and embeds the text in the file's INFO/ICMT chunk. MockSTT reads that
chunk back. Together they give a deterministic speech round trip for tests,
evals and the voice CLI, with realistic audio durations, and no models or keys.

Audio that doesn't carry an embedded transcript (a real microphone) is
"unintelligible" to MockSTT unless a script of expected transcripts is given;
use a Whisper provider for real speech.
"""

import asyncio
import json
import re
import zlib
from collections.abc import AsyncIterator

from voiceops.agent.text import detect_language
from voiceops.speech.audio import AudioFormatError, build_wav, duration_ms, parse_wav, rms, silence, tone
from voiceops.speech.base import AudioInput, SpeechError, STTProvider, Transcript, TTSProvider

MARKER = "voiceops-mock:"
SPEECH_RMS = 200.0


def synthesize(text: str, language: str, sample_rate: int = 16000) -> bytes:
    pcm = bytearray(silence(80, sample_rate))
    for word in re.findall(r"\S+", text):
        freq = 170 + zlib.crc32(word.lower().encode()) % 170
        pcm += tone(freq, max(120, min(450, 55 * len(word))), sample_rate)
        pcm += silence(260 if word[-1] in ".!?…" else 70, sample_rate)
    comment = MARKER + json.dumps({"text": text, "lang": language}, ensure_ascii=False)
    return build_wav(bytes(pcm), sample_rate, comment=comment)


class MockTTS(TTSProvider):
    name = "mock"
    media_type = "audio/wav"

    def __init__(self, sample_rate: int = 16000, latency_ms: int = 0, chunk_bytes: int = 8000):
        self.sample_rate = sample_rate
        self.latency_ms = latency_ms
        self.chunk_bytes = chunk_bytes  # 250 ms of audio per chunk

    async def stream(self, text: str, *, language: str) -> AsyncIterator[bytes]:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        wav = synthesize(text, language, self.sample_rate)
        for i in range(0, len(wav), self.chunk_bytes):
            yield wav[i : i + self.chunk_bytes]


class MockSTT(STTProvider):
    name = "mock"

    def __init__(self, latency_ms: int = 0, script: list[str] | None = None):
        self.latency_ms = latency_ms
        self.script = list(script or [])

    async def transcribe(self, audio: AudioInput, *, language_hint: str | None = None) -> Transcript:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        comment = None
        if audio.format == "wav":
            try:
                wav = parse_wav(audio.data)
            except AudioFormatError as exc:
                raise SpeechError(self.name, "unsupported_audio", str(exc)) from exc
            pcm, rate, comment = wav.pcm, wav.sample_rate, wav.comment
        else:
            pcm, rate = audio.data, audio.sample_rate
        audio_ms = duration_ms(len(pcm), rate)
        if comment and comment.startswith(MARKER):
            payload = json.loads(comment[len(MARKER) :])
            return Transcript(payload["text"], payload.get("lang"), 0.95, self.name, audio_ms, {"source": "embedded"})
        if self.script and rms(pcm) >= SPEECH_RMS:
            text = self.script.pop(0)
            return Transcript(text, detect_language(text, default=language_hint or "tr"), 0.9, self.name, audio_ms, {"source": "script"})
        return Transcript("", language_hint, 0.0, self.name, audio_ms, {"source": "none"})
