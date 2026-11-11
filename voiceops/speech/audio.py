"""Minimal WAV / PCM helpers (stdlib only)."""

import math
import struct
from array import array
from dataclasses import dataclass


class AudioFormatError(ValueError):
    pass


@dataclass
class WavData:
    pcm: bytes
    sample_rate: int
    channels: int
    sample_width: int
    comment: str | None = None  # RIFF LIST/INFO/ICMT


def build_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2, comment: str | None = None) -> bytes:
    fmt = struct.pack(
        "<HHIIHH", 1, channels, sample_rate, sample_rate * channels * sample_width, channels * sample_width, sample_width * 8
    )
    chunks = [b"fmt " + struct.pack("<I", len(fmt)) + fmt]
    if comment is not None:
        text = comment.encode() + b"\x00"
        if len(text) % 2:
            text += b"\x00"
        info = b"INFO" + b"ICMT" + struct.pack("<I", len(text)) + text
        chunks.append(b"LIST" + struct.pack("<I", len(info)) + info)
    data = pcm + (b"\x00" if len(pcm) % 2 else b"")
    chunks.append(b"data" + struct.pack("<I", len(pcm)) + data)
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def parse_wav(data: bytes) -> WavData:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioFormatError("not a RIFF/WAVE file")
    pos, fmt, pcm, comment = 12, None, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise AudioFormatError("short fmt chunk")
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        elif cid == b"LIST" and body[:4] == b"INFO":
            sub = 4
            while sub + 8 <= len(body):
                sid, ssize = body[sub : sub + 4], struct.unpack("<I", body[sub + 4 : sub + 8])[0]
                if sid == b"ICMT":
                    comment = body[sub + 8 : sub + 8 + ssize].rstrip(b"\x00").decode("utf-8", "replace")
                sub += 8 + ssize + (ssize % 2)
        pos += 8 + size + (size % 2)
    if fmt is None or pcm is None:
        raise AudioFormatError("missing fmt or data chunk")
    audio_format, channels, rate, _, _, bits = fmt
    if audio_format != 1:
        raise AudioFormatError("only uncompressed PCM WAV is supported")
    return WavData(pcm, rate, channels, bits // 8, comment)


def duration_ms(n_bytes: int, sample_rate: int, channels: int = 1, sample_width: int = 2) -> float:
    return round(n_bytes / (sample_rate * channels * sample_width) * 1000, 1)


def rms(pcm16: bytes) -> float:
    n = len(pcm16) // 2
    if n == 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm16[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


def tone(freq: float, ms: int, sample_rate: int = 16000, amplitude: int = 3000) -> bytes:
    n = sample_rate * ms // 1000
    fade = max(1, min(n // 4, sample_rate // 200))  # 5 ms fade to avoid clicks
    out = array("h")
    for i in range(n):
        gain = min(1.0, i / fade, (n - i) / fade)
        out.append(int(amplitude * gain * math.sin(2 * math.pi * freq * i / sample_rate)))
    return out.tobytes()


def silence(ms: int, sample_rate: int = 16000) -> bytes:
    return b"\x00\x00" * (sample_rate * ms // 1000)
