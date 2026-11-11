"""Whisper speech-to-text, two deployment options.

* `FasterWhisperSTT` (STT_PROVIDER=whisper_local): faster-whisper in-process on
  CPU/GPU. Audio never leaves the host, which matters for banks under KVKK/GDPR.
  `pip install -e ".[voice-local]"`; the model downloads on first use.
* `WhisperHTTPSTT` (STT_PROVIDER=whisper_http): any OpenAI-compatible
  `POST /audio/transcriptions` endpoint, e.g. a self-hosted faster-whisper server
  in the same VPC, or a cloud API.

Both auto-detect the language (callers switch between Turkish and English) and
fall back to the session language when detection lands outside TR/EN.
"""

import asyncio
import io
import math
import random
from typing import Any

import httpx

from voiceops.retry import RetryPolicy
from voiceops.speech.audio import build_wav, duration_ms, parse_wav
from voiceops.speech.base import AudioInput, SpeechError, STTProvider, Transcript

SUPPORTED = {"tr", "en"}
LANGUAGE_NAMES = {"turkish": "tr", "english": "en"}
NO_SPEECH_PROB = 0.6


def _wav_bytes(audio: AudioInput) -> bytes:
    return audio.data if audio.format == "wav" else build_wav(audio.data, audio.sample_rate)


def _audio_ms(audio: AudioInput) -> float:
    if audio.format == "pcm16":
        return duration_ms(len(audio.data), audio.sample_rate)
    try:
        wav = parse_wav(audio.data)
        return duration_ms(len(wav.pcm), wav.sample_rate, wav.channels, wav.sample_width)
    except ValueError:
        return 0.0


def _confidence(segments: list[dict[str, Any]]) -> float:
    """Mean per-segment probability from avg_logprob; 0 if Whisper heard no speech."""
    speech = [s for s in segments if (s.get("no_speech_prob") or 0.0) < NO_SPEECH_PROB and s.get("text", "").strip()]
    if not speech:
        return 0.0
    probs = [math.exp(min(0.0, float(s.get("avg_logprob", -1.0)))) for s in speech]
    return round(sum(probs) / len(probs), 3)


def _language(code: str | None, fallback: str | None) -> str | None:
    code = LANGUAGE_NAMES.get((code or "").lower(), (code or "").lower())
    return code if code in SUPPORTED else fallback


class FasterWhisperSTT(STTProvider):
    name = "whisper_local"

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8", beam_size: int = 1, model: Any = None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = model
        self._lock = asyncio.Lock()  # one decode at a time per model instance

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise SpeechError(self.name, "not_installed", 'pip install -e ".[voice-local]"') from exc
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def _decode(self, wav: bytes, language: str | None) -> tuple[list[dict[str, Any]], str | None]:
        segments, info = self._load().transcribe(
            io.BytesIO(wav), language=language, beam_size=self.beam_size, vad_filter=True, condition_on_previous_text=False
        )
        segs = [{"text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob} for s in segments]
        return segs, getattr(info, "language", None)

    async def transcribe(self, audio: AudioInput, *, language_hint: str | None = None) -> Transcript:
        wav = _wav_bytes(audio)
        async with self._lock:
            try:
                segs, detected = await asyncio.to_thread(self._decode, wav, None)
                if detected not in SUPPORTED and language_hint:
                    segs, detected = await asyncio.to_thread(self._decode, wav, language_hint)
            except SpeechError:
                raise
            except Exception as exc:  # decoder errors (corrupt audio, OOM) are not retryable here
                raise SpeechError(self.name, "decode_failed", repr(exc)[:200]) from exc
        confidence = _confidence(segs)
        text = "".join(s["text"] for s in segs).strip() if confidence else ""
        return Transcript(text, _language(detected, language_hint), confidence, self.name, _audio_ms(audio), {"segments": len(segs)})


class WhisperHTTPSTT(STTProvider):
    name = "whisper_http"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        policy: RetryPolicy | None = None,
        http: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.model = model
        self.policy = policy or RetryPolicy(max_attempts=2, attempt_timeout_s=10.0, deadline_s=12.0)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = http or httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=self.policy.attempt_timeout_s)
        self._sleep = sleep
        self._rng = rng

    async def aclose(self) -> None:
        await self._http.aclose()

    async def transcribe(self, audio: AudioInput, *, language_hint: str | None = None) -> Transcript:
        wav = _wav_bytes(audio)
        loop = asyncio.get_running_loop()
        start = loop.time()
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await self._http.post(
                    "/audio/transcriptions",
                    files={"file": ("utterance.wav", wav, "audio/wav")},
                    data={"model": self.model, "response_format": "verbose_json"},
                )
                if resp.status_code == 200:
                    break
                err = SpeechError(self.name, f"http_{resp.status_code}", resp.text[:200], resp.status_code in self.policy.retry_statuses)
            except httpx.TimeoutException:
                err = SpeechError(self.name, "timeout", "transcription timed out", retryable=True)
            except httpx.TransportError as exc:
                err = SpeechError(self.name, "unreachable", repr(exc)[:200], retryable=True)
            delay = self.policy.next_delay(attempt, loop.time() - start, None, self._rng) if err.retryable else None
            if delay is None:
                raise err
            await self._sleep(delay)
        body = resp.json()
        segments = body.get("segments") or [{"text": body.get("text", ""), "avg_logprob": -0.3, "no_speech_prob": 0.0}]
        confidence = _confidence(segments)
        text = (body.get("text") or "").strip() if confidence else ""
        meta = {"segments": len(segments), "attempts": attempt}
        return Transcript(text, _language(body.get("language"), language_hint), confidence, self.name, _audio_ms(audio), meta)
