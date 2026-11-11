"""Speech providers and helpers. Real providers are tested offline: Whisper with a
fake model / fake HTTP transport, Azure TTS with a fake HTTP transport."""

import importlib.util
import json
import random
from types import SimpleNamespace

import httpx
import pytest

from voiceops.retry import RetryPolicy
from voiceops.speech.audio import build_wav, duration_ms, parse_wav, silence, tone
from voiceops.speech.azure import AzureTTS, build_ssml, media_type_for
from voiceops.speech.base import AudioInput, SpeechError
from voiceops.speech.mock import MockSTT, MockTTS, synthesize
from voiceops.speech.text import SentenceChunker, to_speech_text
from voiceops.speech.vad import EnergyEndpointer
from voiceops.speech.whisper import FasterWhisperSTT, WhisperHTTPSTT


async def _no_sleep(_: float) -> None:
    return None


# --------------------------------------------------------------------------- audio + mock providers
def test_wav_roundtrip_with_comment():
    pcm = tone(440, 250) + silence(250)
    wav = parse_wav(build_wav(pcm, 16000, comment="merhaba dünya"))
    assert wav.pcm == pcm and wav.sample_rate == 16000 and wav.channels == 1
    assert wav.comment == "merhaba dünya"
    assert duration_ms(len(pcm), 16000) == 500.0


@pytest.mark.parametrize(("text", "lang"), [("Kartım çalındı", "tr"), ("My card was stolen", "en")])
async def test_mock_tts_to_mock_stt_roundtrip(text, lang):
    chunks = [c async for c in MockTTS().stream(text, language=lang)]
    assert len(chunks) > 1  # streamed, not one blob
    tr = await MockSTT().transcribe(AudioInput(b"".join(chunks)), language_hint="tr")
    assert (tr.text, tr.language) == (text, lang)
    assert tr.confidence > 0.9 and 500 < tr.audio_ms < 5000  # speech-like duration


async def test_mock_stt_hears_nothing_in_silence_and_uses_a_script_for_real_audio():
    quiet = AudioInput(silence(800), "pcm16")
    assert (await MockSTT().transcribe(quiet, language_hint="en")).confidence == 0.0
    loud = AudioInput(tone(300, 600), "pcm16")
    tr = await MockSTT(script=["C-1001 lütfen"]).transcribe(loud)
    assert tr.text == "C-1001 lütfen" and tr.language == "tr"


async def test_mock_stt_rejects_non_wav():
    with pytest.raises(SpeechError) as exc:
        await MockSTT().transcribe(AudioInput(b"ID3 not a wav"))
    assert exc.value.code == "unsupported_audio"


def test_energy_endpointer_finds_one_utterance_and_ignores_clicks():
    vad = EnergyEndpointer(end_silence_ms=500)
    stream = silence(400) + tone(220, 40) + silence(300) + tone(250, 900) + silence(700)
    events = []
    for i in range(0, len(stream), 999):  # odd chunk size: frames span chunks
        events += vad.feed(stream[i : i + 999])
    assert events == ["speech_start", "speech_end"]


# --------------------------------------------------------------------------- text shaping
def test_sentence_chunker_streams_sentences_without_splitting_numbers():
    c = SentenceChunker()
    out = []
    for delta in ["Teşekkürler Ay", "şe. 06.11.2026 tarihli 12.450,00 ", "TL tutarındaki işlem bulundu! Onaylı", "yor musunuz?"]:
        out += c.feed(delta)
    assert out == ["Teşekkürler Ayşe.", "06.11.2026 tarihli 12.450,00 TL tutarındaki işlem bulundu!"]
    assert c.flush() == ["Onaylıyor musunuz?"] and c.flush() == []


def test_sentence_chunker_cuts_very_long_runs():
    c = SentenceChunker(max_chars=40)
    parts = c.feed("bir iki üç dört beş altı yedi sekiz dokuz on on bir on iki on üç on dört")
    assert parts and all(len(p) <= 40 for p in parts)


def test_speech_text_drops_citations_and_spells_references():
    text = "**EFT** 17:00'de kesilir. (Kaynak: Para Transferleri, Kesim saatleri) Referansınız APR-C1ED4C08, kaydınız TCK-000003."
    assert to_speech_text(text) == "EFT 17:00'de kesilir. Referansınız A P R, C 1 E D 4 C 0 8, kaydınız T C K, 0 0 0 0 0 3."


# --------------------------------------------------------------------------- whisper (local)
class FakeWhisperModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def transcribe(self, audio, **kw):
        self.calls.append({"bytes": len(audio.getvalue()), **kw})
        segments, language = self.results.pop(0)
        segs = [SimpleNamespace(text=t, avg_logprob=lp, no_speech_prob=ns) for t, lp, ns in segments]
        return iter(segs), SimpleNamespace(language=language)


async def test_faster_whisper_transcribes_and_scores_confidence():
    model = FakeWhisperModel([([(" Kartım", -0.1, 0.01), (" çalındı.", -0.3, 0.02)], "tr")])
    tr = await FasterWhisperSTT(model=model).transcribe(AudioInput(silence(500), "pcm16"))
    assert tr.text == "Kartım çalındı." and tr.language == "tr"
    assert 0.8 < tr.confidence < 0.9
    assert model.calls[0]["language"] is None and model.calls[0]["vad_filter"] is True  # auto-detect first


async def test_faster_whisper_retries_with_the_session_language_when_detection_is_off():
    model = FakeWhisperModel([([(" merhaba", -0.9, 0.1)], "az"), ([(" Merhaba", -0.2, 0.05)], "tr")])
    tr = await FasterWhisperSTT(model=model).transcribe(AudioInput(build_wav(silence(300))), language_hint="tr")
    assert tr.text == "Merhaba" and [c["language"] for c in model.calls] == [None, "tr"]


async def test_faster_whisper_reports_no_speech():
    model = FakeWhisperModel([([(" you", -1.2, 0.9)], "en")])
    tr = await FasterWhisperSTT(model=model).transcribe(AudioInput(silence(300), "pcm16"))
    assert tr.text == "" and tr.confidence == 0.0


@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is not None, reason="faster-whisper is installed")
async def test_faster_whisper_missing_dependency_is_a_clear_error():
    with pytest.raises(SpeechError) as exc:
        await FasterWhisperSTT().transcribe(AudioInput(silence(100), "pcm16"))
    assert exc.value.code == "not_installed"


# --------------------------------------------------------------------------- whisper (http)
def _whisper_http(handler, **kw) -> WhisperHTTPSTT:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://stt/v1", headers={"Authorization": "Bearer t"})
    return WhisperHTTPSTT("http://stt/v1", "whisper-small", http=http, sleep=_no_sleep, rng=random.Random(1), **kw)


async def test_whisper_http_request_and_language_mapping():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"text": " My card was stolen.", "language": "english",
                "segments": [{"text": " My card was stolen.", "avg_logprob": -0.2, "no_speech_prob": 0.01}]}  # fmt: skip
        return httpx.Response(200, json=body)

    tr = await _whisper_http(handler).transcribe(AudioInput(synthesize("x", "en")), language_hint="tr")
    assert tr.text == "My card was stolen." and tr.language == "en" and tr.confidence > 0.8
    req = seen[0]
    assert req.url.path == "/v1/audio/transcriptions" and req.headers["authorization"] == "Bearer t"
    body = req.content.decode("latin-1")
    assert 'name="model"' in body and "whisper-small" in body and "verbose_json" in body and "RIFF" in body


async def test_whisper_http_retries_transient_errors_only():
    statuses = iter([503, 200])

    def flaky(request):
        status = next(statuses)
        return httpx.Response(status, json={"text": "Merhaba", "language": "tr"} if status == 200 else {"error": "busy"})

    tr = await _whisper_http(flaky).transcribe(AudioInput(silence(200), "pcm16"))
    assert tr.text == "Merhaba" and tr.meta["attempts"] == 2

    with pytest.raises(SpeechError) as exc:
        await _whisper_http(lambda r: httpx.Response(401, text="bad key")).transcribe(AudioInput(silence(200), "pcm16"))
    assert exc.value.code == "http_401" and not exc.value.retryable


# --------------------------------------------------------------------------- azure tts
def _azure(handler) -> AzureTTS:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    voices = {"tr": "tr-TR-EmelNeural", "en": "en-US-JennyNeural"}
    return AzureTTS("test-key-not-real", "westeurope", voices, http=http, sleep=_no_sleep,
                    policy=RetryPolicy(max_attempts=2, attempt_timeout_s=1, deadline_s=2))  # fmt: skip


def test_ssml_is_escaped_and_media_types_follow_the_format():
    ssml = build_ssml("Ali & Veli <test>", "en", "en-US-JennyNeural")
    assert "Ali &amp; Veli &lt;test&gt;" in ssml and 'xml:lang="en-US"' in ssml and 'name="en-US-JennyNeural"' in ssml
    assert media_type_for("audio-24khz-48kbitrate-mono-mp3") == "audio/mpeg"
    assert media_type_for("riff-16khz-16bit-mono-pcm") == "audio/wav"


async def test_azure_tts_streams_audio_with_the_right_voice_and_headers():
    seen = []

    async def body():
        for part in (b"ID3-part1", b"part2", b"part3"):
            yield part

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body())

    tts = _azure(handler)
    chunks = [c async for c in tts.stream("Kartınızı bloke ettim.", language="tr")]
    assert b"".join(chunks) == b"ID3-part1part2part3" and len(chunks) == 3
    req = seen[0]
    assert req.url.host == "westeurope.tts.speech.microsoft.com"
    assert req.headers["ocp-apim-subscription-key"] == "test-key-not-real"
    assert req.headers["x-microsoft-outputformat"] == "audio-24khz-48kbitrate-mono-mp3"
    assert "tr-TR-EmelNeural" in req.content.decode() and tts.media_type == "audio/mpeg"


async def test_azure_tts_retries_before_the_first_byte_only():
    statuses = iter([503, 200])
    tts = _azure(lambda r: httpx.Response(next(statuses), content=b"audio"))
    assert [c async for c in tts.stream("hi", language="en")] == [b"audio"]

    with pytest.raises(SpeechError) as exc:
        [c async for c in _azure(lambda r: httpx.Response(401, text="denied")).stream("hi", language="en")]
    assert exc.value.code == "http_401" and not exc.value.retryable


def test_settings_build_providers_without_hardcoded_secrets(monkeypatch):
    from voiceops.config import Settings
    from voiceops.speech import build_stt, build_tts

    assert build_stt(Settings()).name == "mock" and build_tts(Settings()).name == "mock"
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        build_tts(Settings(tts_provider="azure"))
    monkeypatch.setenv("AZURE_SPEECH_KEY", "from-env")
    tts = build_tts(Settings(tts_provider="azure"))
    assert tts.name == "azure" and "from-env" not in json.dumps(Settings().model_dump(mode="json"))
