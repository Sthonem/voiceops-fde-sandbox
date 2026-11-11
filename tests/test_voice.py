"""Voice channel end to end: VoiceSession driven directly (fake socket), plus one
real WebSocket round trip through the ASGI app."""

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator

import pytest

from voiceops.mock_systems.app import fault_injector
from voiceops.mock_systems.middleware import FaultRuleIn
from voiceops.speech.audio import parse_wav, silence, tone
from voiceops.speech.base import SpeechError, STTProvider, TTSProvider
from voiceops.speech.mock import MockSTT, MockTTS, synthesize
from voiceops.voice import VoiceServices, VoiceSession


class FakeSocket:
    def __init__(self):
        self.messages: list[dict | bytes] = []
        self._changed = asyncio.Condition()

    async def send_json(self, data: dict) -> None:
        await self._record(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._record(data)

    async def _record(self, item) -> None:
        async with self._changed:
            self.messages.append(item)
            self._changed.notify_all()

    def events(self, kind: str | None = None) -> list[dict]:
        return [m for m in self.messages if isinstance(m, dict) and (kind is None or m["type"] == kind)]

    async def wait_for(self, predicate, timeout: float = 5.0) -> dict:
        async def _wait():
            async with self._changed:
                while True:
                    for m in self.messages:
                        if isinstance(m, dict) and predicate(m):
                            return m
                    await self._changed.wait()

        return await asyncio.wait_for(_wait(), timeout)

    def segments(self) -> list[dict]:
        """Audio per tts segment: {kind, text, audio}."""
        out, current = [], None
        for m in self.messages:
            if isinstance(m, dict) and m["type"] == "tts.start":
                current = {"kind": m["kind"], "text": m["text"], "audio": b""}
                out.append(current)
            elif isinstance(m, bytes) and current is not None:
                current["audio"] += m
            elif isinstance(m, dict) and m["type"] in ("tts.end", "playback.interrupted"):
                current = None
        return out


def services(app, *, stt: STTProvider | None = None, tts: TTSProvider | None = None, **config) -> VoiceServices:
    base: VoiceServices = app.state.voice
    cfg = dataclasses.replace(base.config, **{"greeting": False, "filler_after_ms": 0, **config})
    return VoiceServices(base.orchestrator, stt or MockSTT(), tts or MockTTS(), cfg)


async def start(app, sock: FakeSocket, **kw) -> VoiceSession:
    session = VoiceSession(services(app, **{k: v for k, v in kw.items() if k not in ("start",)}), sock)
    await session.handle_message(json.dumps({"type": "session.start", "language": "tr", **kw.get("start", {})}))
    return session


async def speak(session: VoiceSession, text: str, lang: str = "tr") -> None:
    wav = synthesize(text, lang)
    for i in range(0, len(wav), 4000):  # the client streams its recording in chunks
        await session.handle_audio(wav[i : i + 4000])
    await session.handle_message('{"type": "audio.end"}')
    await session.drain()


def last_reply(sock: FakeSocket) -> dict:
    return sock.events("agent.reply")[-1]


# --------------------------------------------------------------------------- happy path
async def test_stolen_card_call_end_to_end(app, client, mock_client):
    sock = FakeSocket()
    session = await start(app, sock, greeting=True)
    await session.drain()
    started = sock.events("session.started")[0]
    assert started["stt"] == "mock" and started["tts_media_type"] == "audio/wav"
    greeting = sock.segments()[0]
    assert greeting["kind"] == "greeting" and "hoş geldiniz" in greeting["text"]

    for line in ["Kartım çalındı", "0555 010 1001", "12.04.1988", "kredi kartım", "Evet"]:
        await speak(session, line)
    assert [e["text"] for e in sock.events("transcript.final")] == ["Kartım çalındı", "0555 010 1001", "12.04.1988", "kredi kartım", "Evet"]
    final = last_reply(sock)
    assert "bloke ettim" in final["reply"] and final["verified"] is True

    # Every reply sentence arrived as its own playable WAV, in order.
    replies = [s for s in sock.segments() if s["kind"] == "reply"]
    assert all(parse_wav(s["audio"]).sample_rate == 16000 for s in replies)
    assert " ".join(s["text"] for s in replies).endswith(final["reply"])
    done = sock.events("turn.done")[-1]
    m = done["metrics"]
    assert done["kind"] == "agent" and m["first_audio_ms"] is not None and m["total_ms"] >= m["first_audio_ms"]
    assert m["segments"] >= 1 and m["tts_errors"] == []

    cards = {c["last4"]: c["status"] for c in (await mock_client.get("/payments/customers/C-1001/cards")).json()}
    assert cards["7710"] == "blocked"

    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    assert transcript["channel"] == "voice"
    users = [m for m in transcript["messages"] if m["role"] == "user"]
    assert users[0]["meta"]["stt"]["provider"] == "mock" and users[0]["meta"]["input"] == "speech"
    assistant = [m for m in transcript["messages"] if m["role"] == "assistant" and not m["tool_calls"]][-1]
    assert assistant["meta"]["voice"]["heard"] == final["reply"] and assistant["meta"]["voice"]["interrupted"] is False
    events = [m for m in transcript["messages"] if m["role"] == "event"]
    assert events[0]["meta"]["kind"] == "greeting"
    kinds = {s["kind"] for t in transcript["traces"] for s in t["spans"]}
    assert {"stt", "tts", "router", "tool"} <= kinds
    assert "12.04.1988" not in json.dumps(transcript["traces"])  # spans stay redacted


async def test_typed_input_runs_the_same_pipeline(app):
    sock = FakeSocket()
    session = await start(app, sock)
    await session.handle_message(json.dumps({"type": "text", "text": "EFT kesim saati kaç?"}))
    await session.drain()
    reply = last_reply(sock)
    assert reply["intent"] == "policy_question" and reply["citations"]
    spoken = " ".join(s["text"] for s in sock.segments())
    assert "EFT" in spoken and sock.events("transcript.final") == []


async def test_resume_an_existing_conversation(app, new_chat):
    chat = await new_chat()
    await chat.say("Müşteri numaram C-1001")
    sock = FakeSocket()
    session = await start(app, sock, greeting=True, start={"conversation_id": chat.id})
    await session.handle_message(json.dumps({"type": "text", "text": "Ödememin durumu ne?"}))
    await session.drain()
    assert sock.events("session.started")[0]["resumed"] is True
    assert all(s["kind"] != "greeting" for s in sock.segments())
    assert last_reply(sock)["customer_id"] == "C-1001" and last_reply(sock)["turn"] == 2


# --------------------------------------------------------------------------- latency masking
async def test_filler_covers_a_slow_backend(app, client):
    fault_injector.add(FaultRuleIn(path_prefix="/crm/customers", method="GET", mode="delay", delay_ms=400))
    sock = FakeSocket()
    session = await start(app, sock, filler_after_ms=50)
    await speak(session, "Müşteri numaram C-1001")
    kinds = [s["kind"] for s in sock.segments()]
    assert kinds[0] == "filler" and "reply" in kinds
    assert sock.segments()[0]["text"] == "Bir saniye, kontrol ediyorum."
    m = sock.events("turn.done")[-1]["metrics"]
    assert m["first_audio_ms"] < m["first_reply_audio_ms"]  # the caller heard something early
    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    assert transcript["messages"][-1]["meta"]["voice"]["filler"] == "Bir saniye, kontrol ediyorum."


async def test_no_filler_when_the_answer_is_fast(app):
    sock = FakeSocket()
    session = await start(app, sock, filler_after_ms=2000)
    await speak(session, "Müşteri numaram C-1001")
    assert {s["kind"] for s in sock.segments()} == {"reply"}


# --------------------------------------------------------------------------- barge-in
class GatedTTS(TTSProvider):
    """Sends one chunk per sentence, then waits: playback is 'still going'."""

    name, media_type = "gated", "audio/wav"

    def __init__(self):
        self.release = asyncio.Event()

    async def stream(self, text: str, *, language: str) -> AsyncIterator[bytes]:
        yield b"RIFF-first-chunk"
        await self.release.wait()
        yield b"rest"


async def test_barge_in_stops_playback_but_the_turn_completes(app, client):
    sock = FakeSocket()
    session = await start(app, sock, tts=GatedTTS())
    wav = synthesize("Müşteri numaram C-1001", "tr")
    await session.handle_audio(wav)
    await session.handle_message('{"type": "audio.end"}')
    await sock.wait_for(lambda m: m["type"] == "tts.start" and m["kind"] == "reply")
    await session.handle_message('{"type": "interrupt"}')
    await session.drain()

    interrupted = sock.events("playback.interrupted")[0]
    assert interrupted["source"] == "client" and interrupted["segment"] == 1
    done = sock.events("turn.done")[-1]
    assert done["interrupted"] is True
    assert last_reply(sock)["customer_id"] == "C-1001"  # the turn itself was not cancelled
    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    voice = transcript["messages"][-1]["meta"]["voice"]
    assert voice["interrupted"] is True and voice["heard"] == "" and voice["cut_off"].startswith("Teşekkürler Ayşe")
    tts_span = next(s for t in transcript["traces"] for s in t["spans"] if s["kind"] == "tts")
    assert tts_span["status"] == "interrupted"
    assert transcript["state"]["customer_id"] == "C-1001"


async def test_interrupt_with_nothing_playing_is_a_no_op(app):
    sock = FakeSocket()
    session = await start(app, sock)
    await session.handle_message('{"type": "interrupt"}')
    assert sock.events("playback.interrupted") == []


# --------------------------------------------------------------------------- failures
async def test_silence_reprompts_without_spending_a_turn(app, client):
    sock = FakeSocket()
    session = await start(app, sock)
    from voiceops.speech.audio import build_wav

    for _ in range(2):
        await session.handle_audio(build_wav(silence(600)))
        await session.handle_message('{"type": "audio.end"}')
        await session.drain()
    prompts = [e for e in sock.events("turn.done") if e["kind"] == "prompt"]
    assert [p["reason"] for p in prompts] == ["no_input", "no_input"]
    assert prompts[0]["text"] == "Sizi duyamadım. Tekrar söyler misiniz?"
    assert "yazarak" in prompts[1]["text"]  # second time: offer typing
    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    assert transcript["state"]["turn"] == 0 and sock.events("agent.reply") == []
    assert [m["meta"]["reason"] for m in transcript["messages"]] == ["no_input", "no_input"]


class BrokenSTT(STTProvider):
    name = "broken"

    async def transcribe(self, audio, *, language_hint=None):
        raise SpeechError(self.name, "http_503", "unavailable", retryable=True)


class SlowSTT(STTProvider):
    name = "slow"

    async def transcribe(self, audio, *, language_hint=None):
        await asyncio.sleep(5)


@pytest.mark.parametrize(("stt", "detail"), [(BrokenSTT(), "http_503"), (SlowSTT(), "timeout")])
async def test_stt_failure_asks_the_caller_to_repeat(app, stt, detail):
    sock = FakeSocket()
    session = await start(app, sock, stt=stt, stt_timeout_s=0.05)
    await speak(session, "Kartım çalındı")
    err = sock.events("error")[0]
    assert err["code"] == "stt_failed" and err["detail"] == detail
    assert sock.events("turn.done")[-1]["reason"] == "stt_failed"
    assert "tekrar eder misiniz" in sock.segments()[-1]["text"]


class BrokenTTS(TTSProvider):
    name, media_type = "broken", "audio/wav"

    def __init__(self, fail_times: int = 99, retryable: bool = False):
        self.fail_times, self.retryable, self.calls = fail_times, retryable, 0

    async def stream(self, text: str, *, language: str) -> AsyncIterator[bytes]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise SpeechError(self.name, "http_401", "bad key", retryable=self.retryable)
        yield b"audio"


async def test_tts_failure_falls_back_to_text(app, client):
    sock = FakeSocket()
    session = await start(app, sock, tts=BrokenTTS())
    await speak(session, "Müşteri numaram C-1001")
    assert "Ayşe" in last_reply(sock)["reply"]  # the client can still show the answer
    assert {e["code"] for e in sock.events("error")} == {"tts_failed"}
    segments = len(sock.events("tts.start"))
    assert segments == 2  # "Teşekkürler Ayşe, ..." + "Size nasıl yardımcı olabilirim?"
    assert sock.events("turn.done")[-1]["metrics"]["tts_errors"] == ["http_401"] * segments  # not retried: 401
    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    assert transcript["messages"][-1]["meta"]["voice"]["heard"] == ""
    assert next(s for t in transcript["traces"] for s in t["spans"] if s["kind"] == "tts")["status"] == "error"


async def test_tts_is_retried_once_before_the_first_byte(app):
    sock = FakeSocket()
    tts = BrokenTTS(fail_times=1, retryable=True)
    session = await start(app, sock, tts=tts)
    await speak(session, "Müşteri numaram C-1001")
    assert sock.events("error") == []
    assert tts.calls == len(sock.events("tts.start")) + 1  # first sentence retried once, the rest fine


async def test_audio_too_long_is_rejected(app):
    sock = FakeSocket()
    session = await start(app, sock, max_utterance_s=0.5)
    await session.handle_audio(synthesize("bu çok uzun bir mesaj olacak ve sınırı aşacak kadar kelime içeriyor", "tr"))
    await session.handle_message('{"type": "audio.end"}')
    await session.drain()
    assert sock.events("error")[0]["code"] == "audio_too_long"
    assert sock.events("transcript.final") == []


async def test_protocol_errors(app):
    sock = FakeSocket()
    session = VoiceSession(services(app), sock)
    await session.handle_message("not json")
    await session.handle_audio(b"\x00\x00")
    await session.handle_message('{"type": "audio.end"}')
    await session.handle_message('{"type": "session.start", "audio": {"format": "mp3"}}')
    await session.handle_message('{"type": "session.start"}')
    await session.handle_message('{"type": "dance"}')
    assert [e["code"] for e in sock.events("error")] == [
        "bad_message", "session_not_started", "session_not_started", "unsupported_audio", "unknown_message",
    ]  # fmt: skip
    assert await session.handle_message('{"type": "session.end"}') is False
    await session.close(graceful=True)
    assert sock.events("session.ended")


async def test_hang_up_mid_turn_still_persists_the_turn(app, client):
    fault_injector.add(FaultRuleIn(path_prefix="/crm/customers", method="GET", mode="delay", delay_ms=200))
    sock = FakeSocket()
    session = await start(app, sock)
    await session.handle_audio(synthesize("Müşteri numaram C-1001", "tr"))
    await session.handle_message('{"type": "audio.end"}')
    await sock.wait_for(lambda m: m["type"] == "transcript.final")
    await session.close()  # caller hung up while the CRM was still answering
    transcript = (await client.get(f"/v1/conversations/{session.conversation_id}")).json()
    assert transcript["state"]["customer_id"] == "C-1001"
    assert sock.events("agent.reply") == []  # nothing was sent after the hang-up


# --------------------------------------------------------------------------- server-side VAD
def _pcm_call(*parts: bytes, chunk: int = 640) -> list[bytes]:
    stream = b"".join(parts)
    return [stream[i : i + chunk] for i in range(0, len(stream), chunk)]  # 20 ms frames, like a phone line


async def test_server_vad_finds_the_end_of_the_utterance(app):
    sock = FakeSocket()
    session = await start(app, sock, stt=MockSTT(script=["Müşteri numaram C-1001"]), start={"audio": {"format": "pcm16"}})
    assert sock.events("session.started")[0]["audio"]["vad"] == "server"
    for frame in _pcm_call(silence(300), tone(260, 900), silence(900)):
        await session.handle_audio(frame)
    await session.drain()
    assert [e["type"] for e in sock.events() if e["type"].startswith("vad.")] == ["vad.speech_started", "vad.speech_ended"]
    assert sock.events("transcript.final")[0]["text"] == "Müşteri numaram C-1001"
    assert "Ayşe" in last_reply(sock)["reply"]
    assert len(session.buffer) <= 16000 * 2 * 0.3 + 640  # only a short pre-roll is kept between utterances


async def test_caller_speaking_over_the_bot_is_a_barge_in(app):
    sock = FakeSocket()
    session = await start(
        app, sock, tts=GatedTTS(), stt=MockSTT(script=["Müşteri numaram C-1001", "Teşekkürler"]),
        start={"audio": {"format": "pcm16"}},
    )  # fmt: skip
    for frame in _pcm_call(tone(260, 600), silence(800)):
        await session.handle_audio(frame)
    await sock.wait_for(lambda m: m["type"] == "tts.start" and m["kind"] == "reply")
    for frame in _pcm_call(tone(300, 300)):
        await session.handle_audio(frame)
    interrupted = await sock.wait_for(lambda m: m["type"] == "playback.interrupted")
    assert interrupted["source"] == "vad"


# --------------------------------------------------------------------------- real websocket
def test_websocket_round_trip(tmp_path):
    from starlette.testclient import TestClient

    from voiceops.db import configure_engine
    from voiceops.main import create_app

    configure_engine(f"sqlite+aiosqlite:///{tmp_path / 'ws.db'}")
    with TestClient(create_app()) as tc, tc.websocket_connect("/v1/voice") as ws:
        ws.send_text(json.dumps({"type": "session.start", "language": "en", "greeting": False}))
        assert ws.receive_json()["type"] == "session.started"
        ws.send_bytes(synthesize("What is the EFT cut-off time?", "en"))
        ws.send_text('{"type": "audio.end"}')
        seen, audio = [], 0
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                audio += len(msg["bytes"])
                continue
            event = json.loads(msg["text"])
            seen.append(event["type"])
            if event["type"] == "agent.reply":
                assert event["language"] == "en" and "EFT" in event["reply"]
            if event["type"] == "turn.done":
                break
        assert seen[0] == "transcript.final" and "agent.reply" in seen and audio > 1000
        ws.send_text('{"type": "session.end"}')
        assert ws.receive_json()["type"] == "session.ended"
