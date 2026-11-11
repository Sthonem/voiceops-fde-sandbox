"""Server-side endpointing for continuous PCM streams.

Callers don't press a button at the end of a sentence. With `pcm16` input the
session runs this energy-based detector: speech starts after `min_speech_ms` of
loud frames and ends after `end_silence_ms` of quiet ones, which finalises the
utterance (and, while the bot is talking, counts as a barge-in).

Energy VAD is crude next to a neural VAD (e.g. Silero), but it is dependency
free, deterministic, and the interface is the same.
"""

from voiceops.speech.audio import rms


class EnergyEndpointer:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        threshold: float = 500.0,
        min_speech_ms: int = 120,
        end_silence_ms: int = 700,
    ):
        self.frame_ms = frame_ms
        self.frame_bytes = sample_rate * frame_ms // 1000 * 2
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.in_speech = False
        self._loud_ms = 0
        self._quiet_ms = 0
        self._rest = b""

    def feed(self, pcm: bytes) -> list[str]:
        """Returns the events ("speech_start" / "speech_end") found in this chunk."""
        data = self._rest + pcm
        usable = len(data) // self.frame_bytes * self.frame_bytes
        self._rest = data[usable:]
        events: list[str] = []
        for i in range(0, usable, self.frame_bytes):
            loud = rms(data[i : i + self.frame_bytes]) >= self.threshold
            if not self.in_speech:
                self._loud_ms = self._loud_ms + self.frame_ms if loud else 0
                if self._loud_ms >= self.min_speech_ms:
                    self.in_speech, self._quiet_ms = True, 0
                    events.append("speech_start")
            else:
                self._quiet_ms = 0 if loud else self._quiet_ms + self.frame_ms
                if self._quiet_ms >= self.end_silence_ms:
                    self.in_speech, self._loud_ms = False, 0
                    events.append("speech_end")
        return events
