"""Real-time voice channel: one WebSocket connection = one phone call.

    client                                   server
    session.start {audio, language} ───────► create/resume conversation ─► greeting
    binary audio frames ───────────────────► buffer (+ energy VAD for pcm16)
    audio.end (push-to-talk) ──────────────► STT ─► transcript.final
                                             same turn pipeline as text
                                             (router, guards, tools, traces)
                         ◄─────────────────  agent.tool {start|end}
                         ◄─────────────────  tts.start · binary audio · tts.end
                                             (per sentence, while the model writes)
                         ◄─────────────────  agent.reply (full TurnOut)
                         ◄─────────────────  turn.done {metrics}
    interrupt (barge-in) ──────────────────► stop playback, record what was heard
    text {text} ───────────────────────────► typed input (noisy line, accessibility)
    session.end ───────────────────────────► session.ended

Design choices:

* Speech is pipelined per sentence: TTS starts on the first sentence while the
  model is still generating, and a short filler ("one moment") covers slow tool
  calls, so the caller never sits in silence.
* Barge-in stops the audio, never the turn: a tool call may already have changed
  a system of record, so the turn always completes and is persisted. The
  transcript records which sentences the caller actually heard.
* STT failure or silence re-prompts the caller without spending an agent turn.
* Audio is never stored (a voice is biometric data under KVKK/GDPR); transcripts
  are, with the same redaction rules as text.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from voiceops.agent import templates as tpl
from voiceops.agent.orchestrator import Orchestrator, TraceRecord, TurnEvent
from voiceops.agent.state import ChatMessage
from voiceops.agent.text import detect_language
from voiceops.api.persistence import add_message, add_trace, next_seq
from voiceops.api.turns import TurnExecution, execute_turn, new_conversation
from voiceops.config import Settings
from voiceops.db import session_factory
from voiceops.models import Conversation, Message
from voiceops.speech.base import AudioInput, SpeechError, STTProvider, TTSProvider
from voiceops.speech.text import SentenceChunker, to_speech_text
from voiceops.speech.vad import EnergyEndpointer

log = logging.getLogger("voiceops.voice")


@dataclass(frozen=True)
class VoiceConfig:
    filler_after_ms: int = 900
    min_confidence: float = 0.35
    max_utterance_s: float = 30.0
    stt_timeout_s: float = 10.0
    tts_timeout_s: float = 8.0
    vad_threshold: float = 500.0
    vad_end_silence_ms: int = 700
    preroll_ms: int = 300
    greeting: bool = True

    @classmethod
    def from_settings(cls, s: Settings) -> "VoiceConfig":
        return cls(
            filler_after_ms=s.voice_filler_after_ms,
            min_confidence=s.voice_min_confidence,
            max_utterance_s=s.voice_max_utterance_s,
            stt_timeout_s=s.stt_timeout_s,
            tts_timeout_s=s.tts_timeout_s,
            vad_threshold=s.voice_vad_threshold,
            vad_end_silence_ms=s.voice_vad_end_silence_ms,
            greeting=s.voice_greeting,
        )


@dataclass
class VoiceServices:
    orchestrator: Orchestrator
    stt: STTProvider
    tts: TTSProvider
    config: VoiceConfig = field(default_factory=VoiceConfig)
    sessions: Callable[[], async_sessionmaker[AsyncSession]] = session_factory

    async def aclose(self) -> None:
        await self.stt.aclose()
        await self.tts.aclose()


class Outbound(Protocol):
    async def send_json(self, data: dict[str, Any]) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...


def _ms(start: float, end: float | None = None) -> float:
    return round(((end if end is not None else time.perf_counter()) - start) * 1000, 1)


class Speaker:
    """Plays one turn's speech: sentences are synthesised and streamed in order."""

    def __init__(self, session: "VoiceSession", language: str):
        self.session = session
        self.language = language
        self.queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self.heard: list[str] = []  # reply sentences fully delivered
        self.cut_off: str | None = None  # sentence playing when the caller barged in
        self.current: dict[str, Any] | None = None
        self.first_audio_at: float | None = None
        self.first_reply_audio_at: float | None = None
        self.tts_ms = 0.0
        self.audio_bytes = 0
        self.segments = 0
        self.errors: list[str] = []
        self.interrupted = False
        self.task = asyncio.create_task(self._run())

    def say(self, text: str, kind: str = "reply") -> None:
        if not self.interrupted and text.strip():
            self.queue.put_nowait((kind, text))

    async def finish(self) -> None:
        """Wait until everything queued has been played (or playback was interrupted)."""
        self.queue.put_nowait(None)
        await asyncio.wait({self.task})

    async def interrupt(self) -> dict[str, Any] | None:
        if self.task.done():
            return None
        self.interrupted = True
        current = self.current
        self.task.cancel()
        await asyncio.wait({self.task})
        if current and current["kind"] == "reply":
            self.cut_off = current["text"]
        return current

    async def _run(self) -> None:
        while (item := await self.queue.get()) is not None:
            await self._speak(*item)

    async def _speak(self, kind: str, text: str) -> None:
        s = self.session
        speech = to_speech_text(text)
        if not speech or not s.connected:
            return
        seg = s.next_segment()
        self.current = {"segment": seg, "kind": kind, "text": text, "bytes": 0}
        await s.send({"type": "tts.start", "segment": seg, "kind": kind, "text": text, "media_type": s.tts.media_type})
        t0 = time.perf_counter()
        error: str | None = None
        for attempt in (1, 2):
            error = None
            agen = s.tts.stream(speech, language=self.language)
            try:
                while True:
                    async with asyncio.timeout(s.cfg.tts_timeout_s):
                        chunk = await anext(agen, None)
                    if chunk is None:
                        break
                    now = time.perf_counter()
                    self.first_audio_at = self.first_audio_at or now
                    if kind == "reply":
                        self.first_reply_audio_at = self.first_reply_audio_at or now
                    self.current["bytes"] += len(chunk)
                    await s.send_bytes(chunk)
            except (SpeechError, TimeoutError) as exc:
                error = getattr(exc, "code", "timeout")
                retryable = isinstance(exc, TimeoutError) or bool(getattr(exc, "retryable", False))
                if attempt == 1 and retryable and self.current["bytes"] == 0:
                    continue
            finally:
                await agen.aclose()
            break
        sent = self.current["bytes"]
        elapsed = _ms(t0)
        self.tts_ms += elapsed
        self.audio_bytes += sent
        self.segments += 1
        if error:
            self.errors.append(error)
            await s.send(
                {"type": "error", "code": "tts_failed", "detail": error, "segment": seg, "retryable": False,
                 "message": "No audio for this sentence; its text is in agent.reply."}  # fmt: skip
            )
        await s.send({"type": "tts.end", "segment": seg, "bytes": sent, "tts_ms": elapsed})
        if kind == "reply" and sent and not error:
            self.heard.append(text)
        self.current = None


class VoiceSession:
    def __init__(self, services: VoiceServices, out: Outbound, *, clock: Callable[[], float] = time.perf_counter):
        self.services = services
        self.cfg = services.config
        self.tts = services.tts
        self.out = out
        self.clock = clock
        self.conversation_id: str | None = None
        self.language = "tr"
        self.input_format = "wav"
        self.sample_rate = 16000
        self.endpointer: EnergyEndpointer | None = None
        self.buffer = bytearray()
        self.inputs: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.speaker: Speaker | None = None
        self.connected = True
        self.closing = False
        self.last_turn = 0
        self.no_input_streak = 0
        self._too_long = False
        self._segment = 0
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------ transport
    async def send(self, event: dict[str, Any]) -> None:
        if not self.connected:
            return
        try:
            async with self._send_lock:
                await self.out.send_json(event)
        except Exception:  # caller hung up: keep going so the turn is still persisted
            self.connected = False

    async def send_bytes(self, data: bytes) -> None:
        if not self.connected:
            return
        try:
            async with self._send_lock:
                await self.out.send_bytes(data)
        except Exception:
            self.connected = False

    def next_segment(self) -> int:
        self._segment += 1
        return self._segment

    # ------------------------------------------------------------------ inbound
    async def handle_message(self, raw: str) -> bool:
        """Handle one JSON control message. Returns False when the call should end."""
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
        except (ValueError, AttributeError):
            await self.send({"type": "error", "code": "bad_message", "message": "Expected a JSON object"})
            return True
        if kind == "session.start":
            await self._start(msg)
        elif kind == "ping":
            await self.send({"type": "pong"})
        elif kind == "session.end":
            return False
        elif self.conversation_id is None:
            await self.send({"type": "error", "code": "session_not_started", "message": "Send session.start first"})
        elif kind == "audio.end":
            self._finalize_utterance()
        elif kind == "text":
            if text := str(msg.get("text", "")).strip()[:2000]:
                self.inputs.put_nowait(("text", text, self.clock()))
        elif kind == "interrupt":
            await self.barge_in("client")
        else:
            await self.send({"type": "error", "code": "unknown_message", "message": f"Unknown type {kind!r}"})
        return True

    async def handle_audio(self, chunk: bytes) -> None:
        if self.conversation_id is None:
            await self.send({"type": "error", "code": "session_not_started", "message": "Send session.start first"})
            return
        self.buffer += chunk
        if self.endpointer is not None:
            for event in self.endpointer.feed(chunk):
                if event == "speech_start":
                    await self.send({"type": "vad.speech_started"})
                    await self.barge_in("vad")
                else:
                    await self.send({"type": "vad.speech_ended"})
                    self._finalize_utterance()
            if not self.endpointer.in_speech:
                keep = self.sample_rate * 2 * self.cfg.preroll_ms // 1000
                del self.buffer[: max(0, len(self.buffer) - keep)]
            elif len(self.buffer) > self._max_bytes():
                self._finalize_utterance()
        elif len(self.buffer) > self._max_bytes():
            self.buffer.clear()
            if not self._too_long:
                self._too_long = True
                await self.send(
                    {"type": "error", "code": "audio_too_long", "message": f"Utterances are limited to {self.cfg.max_utterance_s:.0f}s"}
                )

    async def barge_in(self, source: str) -> None:
        sp = self.speaker
        if sp is None or sp.task.done():
            return
        current = await sp.interrupt()
        await self.send(
            {"type": "playback.interrupted", "source": source, "segment": current["segment"] if current else None}
        )

    async def drain(self) -> None:
        """Wait until every queued input has been processed (tests, graceful end)."""
        await self.inputs.join()

    async def close(self, *, graceful: bool = False) -> None:
        if graceful:
            await self.drain()
            await self.send({"type": "session.ended", "conversation_id": self.conversation_id})
        else:
            self.connected = False  # hang-up: finish the turn in flight, skip queued input
            self.closing = True
            if self.speaker is not None:
                await self.speaker.interrupt()
            try:
                async with asyncio.timeout(60):
                    await self.drain()
            except TimeoutError:
                log.warning("voice_close_timeout", extra={"fields": {"conversation_id": self.conversation_id}})
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.wait({self.worker})

    # ------------------------------------------------------------------ session
    def _max_bytes(self) -> int:
        return int(self.cfg.max_utterance_s * self.sample_rate * 2) + 4096

    def _finalize_utterance(self) -> None:
        data, too_long = bytes(self.buffer), self._too_long
        self.buffer.clear()
        self._too_long = False
        if data and not too_long:
            self.inputs.put_nowait(("audio", AudioInput(data, self.input_format, self.sample_rate), self.clock()))  # type: ignore[arg-type]

    async def _start(self, msg: dict[str, Any]) -> None:
        if self.conversation_id is not None:
            await self.send({"type": "error", "code": "already_started", "message": "Session already started"})
            return
        audio = msg.get("audio") or {}
        fmt = audio.get("format", "wav")
        rate = int(audio.get("sample_rate", 16000))
        if fmt not in ("wav", "pcm16") or not 8000 <= rate <= 48000:
            await self.send({"type": "error", "code": "unsupported_audio", "message": "audio.format must be wav or pcm16 (8-48 kHz)"})
            return
        self.input_format, self.sample_rate = fmt, rate
        vad = msg.get("vad") or ("server" if fmt == "pcm16" else "client")
        if vad == "server" and fmt == "pcm16":
            self.endpointer = EnergyEndpointer(rate, threshold=self.cfg.vad_threshold, end_silence_ms=self.cfg.vad_end_silence_ms)
        language = msg.get("language") if msg.get("language") in ("tr", "en") else "tr"
        orch = self.services.orchestrator
        async with self.services.sessions()() as db:
            if cid := msg.get("conversation_id"):
                conv = await db.get(Conversation, cid)
                if conv is None:
                    await self.send({"type": "error", "code": "conversation_not_found", "message": "Unknown conversation"})
                    return
                resumed = True
            else:
                conv = new_conversation("voice", language, orch.llm.name)
                db.add(conv)
                await db.commit()
                resumed = False
            self.conversation_id, self.language = conv.id, conv.language
            self.last_turn = int((conv.state or {}).get("turn", 0))
        await self.send(
            {
                "type": "session.started",
                "conversation_id": self.conversation_id,
                "language": self.language,
                "resumed": resumed,
                "audio": {"format": fmt, "sample_rate": rate, "vad": "server" if self.endpointer else "client"},
                "stt": self.services.stt.name,
                "tts": self.tts.name,
                "tts_media_type": self.tts.media_type,
            }
        )
        self.worker = asyncio.create_task(self._work())
        if self.cfg.greeting and not resumed and msg.get("greeting", True):
            self.inputs.put_nowait(("greeting", self.clock()))

    async def _work(self) -> None:
        while True:
            item = await self.inputs.get()
            try:
                if self.closing:
                    continue
                if item[0] == "greeting":
                    await self._say_and_log("greeting", "greeting", item[1])
                elif item[0] == "audio":
                    await self._utterance(item[1], item[2])
                elif item[0] == "text":
                    await self._agent_turn(item[1], item[2], {"channel": "voice", "input": "text"}, [], None)
            except Exception as exc:
                log.exception("voice_input_failed", exc_info=exc)
                await self.send({"type": "error", "code": "internal_error", "message": "Something went wrong; please repeat."})
            finally:
                self.inputs.task_done()

    # ------------------------------------------------------------------ pipeline
    async def _utterance(self, audio: AudioInput, t_end: float) -> None:
        stt = self.services.stt
        t0 = self.clock()
        try:
            async with asyncio.timeout(self.cfg.stt_timeout_s):
                tr = await stt.transcribe(audio, language_hint=self.language)
        except (SpeechError, TimeoutError) as exc:
            code = getattr(exc, "code", "timeout")
            await self.send({"type": "error", "code": "stt_failed", "detail": code, "retryable": True})
            await self._say_and_log("voice_stt_error", "prompt", t_end, reason="stt_failed")
            return
        stt_ms = _ms(t0, self.clock())
        await self.send(
            {"type": "transcript.final", "text": tr.text, "language": tr.language, "confidence": tr.confidence,
             "audio_ms": tr.audio_ms, "stt_ms": stt_ms}  # fmt: skip
        )
        if not tr.text.strip() or tr.confidence < self.cfg.min_confidence:
            self.no_input_streak += 1
            key = "voice_no_input" if self.no_input_streak == 1 else "voice_no_input_again"
            await self._say_and_log(key, "prompt", t_end, reason="no_input", stt_ms=stt_ms)
            return
        self.no_input_streak = 0
        meta = {"provider": stt.name, "language": tr.language, "confidence": tr.confidence, "audio_ms": tr.audio_ms, "latency_ms": stt_ms}
        span = TraceRecord(
            "stt", stt.name, "ok",
            input={"format": audio.format, "audio_ms": tr.audio_ms},
            output={"language": tr.language, "confidence": tr.confidence, "chars": len(tr.text)},
            latency_ms=stt_ms,
        )  # fmt: skip
        await self._agent_turn(tr.text, t_end, {"channel": "voice", "input": "speech", "stt": meta}, [span], stt_ms)

    async def _agent_turn(
        self, text: str, t_end: float, user_meta: dict[str, Any], spans: list[TraceRecord], stt_ms: float | None
    ) -> None:
        language = detect_language(text, default=self.language)
        sp = self.speaker = Speaker(self, language)
        chunker = SentenceChunker()
        progress: dict[str, Any] = {"spoken": False, "filler": None}

        def speak(sentences: list[str]) -> None:
            for sentence in sentences:
                progress["spoken"] = True
                sp.say(sentence)

        async def on_event(ev: TurnEvent) -> None:
            if ev["type"] == "text":
                speak(chunker.feed(ev["text"]))
            elif ev["type"] == "text_break":
                speak(chunker.flush())
            elif ev["type"] == "tool":
                await self.send({"type": "agent.tool", **{k: v for k, v in ev.items() if k != "type"}})

        async def filler() -> None:
            await asyncio.sleep(self.cfg.filler_after_ms / 1000)
            if not progress["spoken"]:
                progress["filler"] = tpl.render("voice_filler", language)  # type: ignore[arg-type]
                sp.say(progress["filler"], "filler")

        filler_task = asyncio.create_task(filler()) if self.cfg.filler_after_ms > 0 else None
        t_agent = self.clock()
        try:
            async with self.services.sessions()() as db:
                turn = await execute_turn(
                    db, self.services.orchestrator, self.conversation_id or "", text,
                    user_meta=user_meta, spans=spans, on_event=on_event,
                )  # fmt: skip
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
            await self.send({"type": "error", **detail})
            await sp.finish()
            return
        except Exception as exc:
            log.exception("voice_turn_failed", exc_info=exc)
            await self.send({"type": "error", "code": "turn_failed", "message": "The agent could not complete this turn."})
            sp.say(tpl.render("system_error", language))  # type: ignore[arg-type]
            await sp.finish()
            return
        finally:
            if filler_task is not None:
                filler_task.cancel()
        agent_ms = _ms(t_agent, self.clock())
        speak(chunker.flush())
        self.language, self.last_turn = turn.out.language, turn.out.turn
        await self.send({"type": "agent.reply", **turn.out.model_dump(mode="json")})
        await sp.finish()
        self.speaker = None
        metrics = self._metrics(sp, t_end, stt_ms=stt_ms, agent_ms=agent_ms)
        await self.send(
            {"type": "turn.done", "kind": "agent", "turn": turn.out.turn, "trace_id": turn.out.trace_id,
             "interrupted": sp.interrupted, "metrics": metrics}  # fmt: skip
        )
        await self._annotate(turn, sp, metrics, progress["filler"])

    async def _say_and_log(self, key: str, kind: str, t_end: float, *, reason: str | None = None, stt_ms: float | None = None) -> None:
        """Speak a fixed prompt (greeting, re-prompt) and record it as a transcript event."""
        text = tpl.render(key, self.language)  # type: ignore[arg-type]
        sp = self.speaker = Speaker(self, self.language)
        sp.say(text, kind)
        await sp.finish()
        self.speaker = None
        async with self.services.sessions()() as db:
            cid = self.conversation_id or ""
            meta = {"kind": kind, "channel": "voice", **({"reason": reason} if reason else {})}
            add_message(db, cid, await next_seq(db, cid), self.last_turn, ChatMessage(role="event", content=text, meta=meta))
            await db.commit()
        await self.send(
            {"type": "turn.done", "kind": kind, "reason": reason, "text": text,
             "metrics": self._metrics(sp, t_end, stt_ms=stt_ms, agent_ms=None)}  # fmt: skip
        )

    def _metrics(self, sp: Speaker, t_end: float, *, stt_ms: float | None, agent_ms: float | None) -> dict[str, Any]:
        """Latency from the end of the caller's speech (or typed input) to audio out."""
        return {
            "stt_ms": stt_ms,
            "agent_ms": agent_ms,
            "first_audio_ms": _ms(t_end, sp.first_audio_at) if sp.first_audio_at else None,
            "first_reply_audio_ms": _ms(t_end, sp.first_reply_audio_at) if sp.first_reply_audio_at else None,
            "tts_ms": round(sp.tts_ms, 1),
            "total_ms": _ms(t_end, self.clock()),
            "segments": sp.segments,
            "audio_bytes": sp.audio_bytes,
            "tts_errors": sp.errors,
        }

    async def _annotate(self, turn: TurnExecution, sp: Speaker, metrics: dict[str, Any], filler: str | None) -> None:
        """Attach what the caller actually heard, and the TTS span, to the persisted turn."""
        if turn.result is None or turn.final_seq is None:
            return
        cid = self.conversation_id or ""
        voice = {"heard": " ".join(sp.heard), "interrupted": sp.interrupted, "cut_off": sp.cut_off, "filler": filler, "metrics": metrics}
        status = "error" if sp.errors else "interrupted" if sp.interrupted else "ok"
        root = next(t for t in turn.result.traces if t.kind == "turn")
        async with self.services.sessions()() as db:
            msg = await db.scalar(select(Message).where(Message.conversation_id == cid, Message.seq == turn.final_seq))
            if msg is not None:
                msg.meta = {**msg.meta, "voice": voice}
            span = TraceRecord(
                "tts", self.tts.name, status,
                input={"segments": sp.segments, "media_type": self.tts.media_type},
                output={"first_audio_ms": metrics["first_audio_ms"], "audio_bytes": sp.audio_bytes,
                        "heard_segments": len(sp.heard), "errors": sp.errors},
                latency_ms=round(sp.tts_ms, 1), parent_id=root.span_id,
            )  # fmt: skip
            add_trace(db, cid, turn.out.turn, turn.result.trace_id, span)
            await db.commit()
