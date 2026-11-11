"""Azure AI Speech neural text-to-speech over REST (TTS_PROVIDER=azure).

Turkish and English neural voices, SSML input, audio streamed back in chunks as
it is synthesised. A request is retried only before its first audio byte: once
the caller has heard part of a sentence, replaying it from the start would be
worse than a clean failure (the session then falls back to text for that segment).

Credentials: AZURE_SPEECH_KEY (+ AZURE_SPEECH_REGION) from the environment.
"""

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import httpx

from voiceops.retry import RetryPolicy
from voiceops.speech.base import SpeechError, TTSProvider

LOCALES = {"tr": "tr-TR", "en": "en-US"}


def media_type_for(output_format: str) -> str:
    if "mp3" in output_format:
        return "audio/mpeg"
    if output_format.startswith("riff"):
        return "audio/wav"
    if "opus" in output_format or output_format.startswith("ogg"):
        return "audio/ogg"
    return "audio/L16"


def build_ssml(text: str, language: str, voice: str) -> str:
    locale = LOCALES.get(language, "tr-TR")
    return (
        f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang={quoteattr(locale)}>"
        f"<voice name={quoteattr(voice)}>{escape(text)}</voice></speak>"
    )


class AzureTTS(TTSProvider):
    name = "azure"

    def __init__(
        self,
        key: str,
        region: str,
        voices: dict[str, str],
        output_format: str = "audio-24khz-48kbitrate-mono-mp3",
        policy: RetryPolicy | None = None,
        http: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.voices = voices
        self.output_format = output_format
        self.media_type = media_type_for(output_format)
        self.policy = policy or RetryPolicy(max_attempts=2, attempt_timeout_s=8.0, deadline_s=10.0)
        self.url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self._headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
            "User-Agent": "voiceops-fde-sandbox",
        }
        self._http = http or httpx.AsyncClient(timeout=self.policy.attempt_timeout_s)
        self._sleep = sleep
        self._rng = rng

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream(self, text: str, *, language: str) -> AsyncIterator[bytes]:
        voice = self.voices.get(language) or self.voices["tr"]
        ssml = build_ssml(text, language, voice).encode()
        loop = asyncio.get_running_loop()
        start = loop.time()
        attempt = 0
        while True:
            attempt += 1
            started = False
            try:
                async with self._http.stream("POST", self.url, content=ssml, headers=self._headers) as resp:
                    if resp.status_code != 200:
                        detail = (await resp.aread())[:200].decode("utf-8", "replace")
                        retryable = resp.status_code in self.policy.retry_statuses
                        raise SpeechError(self.name, f"http_{resp.status_code}", detail, retryable)
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            started = True
                            yield chunk
                    return
            except SpeechError as exc:
                err = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                code = "timeout" if isinstance(exc, httpx.TimeoutException) else "unreachable"
                err = SpeechError(self.name, "stream_interrupted" if started else code, repr(exc)[:200], retryable=not started)
            delay = self.policy.next_delay(attempt, loop.time() - start, None, self._rng) if err.retryable and not started else None
            if delay is None:
                raise err
            await self._sleep(delay)
