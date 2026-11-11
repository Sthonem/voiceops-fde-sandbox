"""Text shaping for speech.

* `SentenceChunker` turns streamed text deltas into sentences, so TTS can start
  on the first sentence while the model is still writing the rest.
* `to_speech_text` turns a written reply into what should be *said*: no markdown,
  no "(Source: ...)" citations (kept in the transcript and the API response), and
  reference numbers spelled out so a caller can write them down.
"""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_CITATION = re.compile(r"\s*\((?:Kaynak|Kaynaklar|Source|Sources):[^)]*\)", re.IGNORECASE)
_MARKDOWN = re.compile(r"[*_`#>|]+")
_REFERENCE = re.compile(r"\b([A-Z]{1,3})-([A-Z0-9]{4,8})\b")


class SentenceChunker:
    def __init__(self, max_chars: int = 240):
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        last = 0
        for m in _SENTENCE_END.finditer(self._buf):
            out.append(self._buf[last : m.end()].strip())
            last = m.end()
        self._buf = self._buf[last:]
        # A very long run without punctuation: cut at the last comma or space.
        while len(self._buf) > self.max_chars:
            cut = max(self._buf.rfind(", ", 0, self.max_chars), self._buf.rfind(" ", 0, self.max_chars))
            cut = cut + 1 if cut > 0 else self.max_chars
            out.append(self._buf[:cut].strip())
            self._buf = self._buf[cut:]
        return [s for s in out if s]

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def spell_reference(match: re.Match[str]) -> str:
    prefix, body = match.group(1), match.group(2)
    return f"{' '.join(prefix)}, {' '.join(body)}"


def to_speech_text(text: str) -> str:
    t = _CITATION.sub("", text)
    t = _MARKDOWN.sub("", t)
    t = _REFERENCE.sub(spell_reference, t)
    return re.sub(r"\s+", " ", t).strip()
