"""Voice call simulator over the real WebSocket protocol.

    python -m voiceops.voice_cli                       # starts the API in-process (mock STT/TTS)
    python -m voiceops.voice_cli --url http://localhost:8000
    python -m voiceops.voice_cli --save-audio calls/   # keep the bot's audio, one file per sentence

Each line you type becomes caller audio (mock TTS, which the mock STT reads back),
streamed to the server in 100 ms chunks followed by `audio.end`.

    /text <message>   send typed input instead of audio
    @path/to/file.wav send a real recording (needs STT_PROVIDER=whisper_* on the server)
    /interrupt        barge in while the bot is talking
    Ctrl+D            hang up
"""

import argparse
import asyncio
import contextlib
import json
import os
import socket
from pathlib import Path
from typing import Any

EXTENSIONS = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/ogg": "ogg"}


class CallState:
    def __init__(self, save_dir: Path | None):
        self.save_dir = save_dir
        self.started = asyncio.Event()
        self.done = asyncio.Event()
        self.ended = asyncio.Event()
        self.segment: dict[str, Any] | None = None
        self.media_type = "audio/wav"


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}ms"


def _on_event(ev: dict[str, Any], st: CallState) -> None:
    t = ev["type"]
    if t == "session.started":
        print(f"call {ev['conversation_id']}  stt={ev['stt']} tts={ev['tts']} ({ev['tts_media_type']})  Ctrl+D to hang up\n")
        st.media_type = ev["tts_media_type"]
        st.started.set()
    elif t == "transcript.final":
        print(f"       heard \"{ev['text']}\" ({ev['language']}, confidence {ev['confidence']:.2f}, stt {_fmt_ms(ev['stt_ms'])})")
    elif t == "agent.tool" and ev.get("phase") == "end":
        print(f"       ↳ {ev['name']} {ev['status']} {_fmt_ms(ev['latency_ms'])}")
    elif t == "tts.start":
        tag = "" if ev["kind"] == "reply" else f"  [{ev['kind']}]"
        print(f"bot  ♪ {ev['text']}{tag}")
        st.segment = {"n": ev["segment"], "kind": ev["kind"], "audio": bytearray()}
    elif t == "tts.end" and st.segment is not None:
        if st.save_dir is not None and st.segment["audio"]:
            ext = EXTENSIONS.get(st.media_type, "bin")
            path = st.save_dir / f"{st.segment['n']:03d}-{st.segment['kind']}.{ext}"
            path.write_bytes(bytes(st.segment["audio"]))
        st.segment = None
    elif t == "agent.reply":
        print(f"       intent={ev['intent']} status={ev['status']} customer={ev['customer_id']} verified={ev['verified']}")
    elif t == "playback.interrupted":
        print(f"       ✂ playback interrupted ({ev['source']})")
        st.segment = None
    elif t == "error":
        print(f"       ! {ev.get('code')}: {ev.get('message') or ev.get('detail', '')}")
    elif t == "turn.done":
        m = ev["metrics"]
        print(
            f"       ⏱ first audio {_fmt_ms(m['first_audio_ms'])} · agent {_fmt_ms(m['agent_ms'])} · "
            f"tts {_fmt_ms(m['tts_ms'])} · total {_fmt_ms(m['total_ms'])}\n"
        )
        st.done.set()
    elif t == "session.ended":
        st.ended.set()


async def _receive(ws: Any, st: CallState) -> None:
    async for msg in ws:
        if isinstance(msg, bytes):
            if st.segment is not None:
                st.segment["audio"] += msg
            continue
        _on_event(json.loads(msg), st)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve_in_process() -> tuple[Any, asyncio.Task[None], str]:
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    import uvicorn

    from voiceops.main import create_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.05)
    return server, task, f"ws://127.0.0.1:{port}"


async def _send_audio(ws: Any, audio: bytes, realtime: bool) -> None:
    for i in range(0, len(audio), 3200):  # ~100 ms of 16 kHz PCM per frame
        await ws.send(audio[i : i + 3200])
        if realtime:
            await asyncio.sleep(0.1)
    await ws.send(json.dumps({"type": "audio.end"}))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a phone call with the VoiceOps agent")
    parser.add_argument("--url", help="Base URL of a running API (http://... or ws://...); omit to run in-process")
    parser.add_argument("--lang", choices=["tr", "en"], default="tr")
    parser.add_argument("--save-audio", type=Path, help="Directory to write the bot's audio segments to")
    parser.add_argument("--no-greeting", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace caller audio at real time instead of sending it at once")
    args = parser.parse_args()

    from websockets.asyncio.client import connect

    from voiceops.agent.text import detect_language
    from voiceops.speech.mock import synthesize

    if args.save_audio:
        args.save_audio.mkdir(parents=True, exist_ok=True)
    st = CallState(args.save_audio)

    async with contextlib.AsyncExitStack() as stack:
        if args.url:
            base = args.url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        else:
            server, task, base = await _serve_in_process()

            async def _stop() -> None:
                server.should_exit = True
                await task

            stack.push_async_callback(_stop)
        ws = await stack.enter_async_context(connect(f"{base}/v1/voice", max_size=None))
        receiver = asyncio.create_task(_receive(ws, st))
        await ws.send(json.dumps({"type": "session.start", "language": args.lang, "greeting": not args.no_greeting}))
        await asyncio.wait_for(st.started.wait(), 10)
        if not args.no_greeting:
            await asyncio.wait_for(st.done.wait(), 30)
        while True:
            try:
                line = (await asyncio.to_thread(input, "you  > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line == "/interrupt":
                await ws.send(json.dumps({"type": "interrupt"}))
                continue
            st.done.clear()
            if line.startswith("/text "):
                await ws.send(json.dumps({"type": "text", "text": line[6:]}))
            elif line.startswith("@"):
                await _send_audio(ws, Path(line[1:]).expanduser().read_bytes(), args.realtime)
            else:
                await _send_audio(ws, synthesize(line, detect_language(line, default=args.lang)), args.realtime)
            await asyncio.wait_for(st.done.wait(), 60)
        await ws.send(json.dumps({"type": "session.end"}))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(st.ended.wait(), 10)
        receiver.cancel()


if __name__ == "__main__":
    asyncio.run(main())
