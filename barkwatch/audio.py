"""Audio input: live capture (sounddevice), WAV replay, resampling and level helpers."""
from __future__ import annotations

import logging
import queue
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)

TARGET_RATE = 16000


@dataclass
class Block:
    t_end: float          # wall-clock time (epoch seconds) of the last sample
    samples: np.ndarray   # float32 mono, 16 kHz, range [-1, 1]


def dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (a full-scale sine is about -3 dBFS)."""
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


class Decimator:
    """Streaming integer-factor decimator with a windowed-sinc anti-alias filter."""

    def __init__(self, factor: int, taps: int = 63):
        self.factor = factor
        if factor == 1:
            return
        n = np.arange(taps) - (taps - 1) / 2
        cutoff = 0.45 / factor  # normalised to the input rate, a little below the new Nyquist
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)
        self.tail = np.zeros(taps - 1, dtype=np.float32)
        self.phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1:
            return x
        buf = np.concatenate([self.tail, x])
        y = np.convolve(buf, self.h, mode="valid")
        self.tail = buf[-(len(self.h) - 1):]
        out = y[self.phase::self.factor]
        self.phase = (self.phase - len(y)) % self.factor
        return out.astype(np.float32)


def _gain(cfg: AudioConfig) -> float:
    return float(10 ** (cfg.gain_db / 20.0))


class LiveSource:
    """Captures from an ALSA/PortAudio device and yields hop-sized 16 kHz blocks."""

    def __init__(self, cfg: AudioConfig):
        if cfg.capture_rate % TARGET_RATE:
            raise ValueError("audio.capture_rate must be a multiple of 16000 (16000, 32000, 48000)")
        self.cfg = cfg
        self.decimator = Decimator(cfg.capture_rate // TARGET_RATE)
        self.q: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=200)
        self.hop = int(round(cfg.hop_seconds * TARGET_RATE))

    def _callback(self, indata, frames, time_info, status):  # runs on the PortAudio thread
        if status:
            log.warning("audio status: %s", status)
        try:
            self.q.put_nowait((time.time(), indata[:, self.cfg.channel_index].copy()))
        except queue.Full:
            log.error("audio queue full; analysis is falling behind, dropping audio")

    def blocks(self) -> Iterator[Block]:
        import sounddevice as sd  # imported lazily so tests/replay work without PortAudio

        blocksize = int(self.cfg.capture_rate * self.cfg.hop_seconds)
        pending = np.zeros(0, dtype=np.float32)
        gain = _gain(self.cfg)
        with sd.InputStream(device=self.cfg.device, channels=self.cfg.channels,
                            samplerate=self.cfg.capture_rate, blocksize=blocksize,
                            dtype="float32", callback=self._callback):
            log.info("capturing from %s at %d Hz", self.cfg.device or "default device",
                     self.cfg.capture_rate)
            while True:
                t, raw = self.q.get()
                pending = np.concatenate([pending, self.decimator.process(raw) * gain])
                while len(pending) >= self.hop:
                    yield Block(t, pending[:self.hop])
                    pending = pending[self.hop:]


class WavSource:
    """Replays a WAV file as if it were live (for tuning and regression tests)."""

    def __init__(self, path: str, cfg: AudioConfig, start_time: float | None = None):
        self.path = path
        self.cfg = cfg
        self.start_time = time.time() if start_time is None else start_time

    def blocks(self) -> Iterator[Block]:
        with wave.open(self.path, "rb") as wf:
            rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if width != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        x = np.frombuffer(raw, dtype="<i2").reshape(-1, ch)[:, min(self.cfg.channel_index, ch - 1)]
        x = x.astype(np.float32) / 32768.0 * _gain(self.cfg)
        if rate != TARGET_RATE:
            if rate % TARGET_RATE == 0:
                x = Decimator(rate // TARGET_RATE).process(x)
            else:  # crude but adequate for offline analysis
                n = int(len(x) * TARGET_RATE / rate)
                x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        hop = int(round(self.cfg.hop_seconds * TARGET_RATE))
        for i in range(0, len(x) - hop + 1, hop):
            yield Block(self.start_time + (i + hop) / TARGET_RATE, x[i:i + hop])
