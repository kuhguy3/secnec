"""Bark detectors.

Both detectors analyse a 0.975 s window of 16 kHz mono audio and return a
bark score in [0, 1]. The YAMNet detector also returns a speech score used
by the privacy filter.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_RATE, dbfs
from .config import DetectorConfig

log = logging.getLogger(__name__)

WINDOW_SAMPLES = 15600  # YAMNet's fixed input: 0.975 s at 16 kHz
WINDOW_SECONDS = WINDOW_SAMPLES / TARGET_RATE


@dataclass
class Detection:
    bark: float
    speech: float
    level_dbfs: float
    label: str = ""       # top class (YAMNet) or short reason (heuristic), for debugging


class NoiseFloor:
    """Tracks the ambient level: follows drops quickly, rises slowly, and is frozen during barking."""

    def __init__(self, initial_db: float = -60.0, rise_db_per_s: float = 0.05, fall_alpha: float = 0.3):
        self.db = initial_db
        self.rise = rise_db_per_s
        self.fall_alpha = fall_alpha
        self._initialised = False

    MIN_DB = -90.0  # real microphones never reach digital silence; avoids -200 dB floors on muted input

    def update(self, level_db: float, dt: float, frozen: bool) -> float:
        level_db = max(level_db, self.MIN_DB)
        if not self._initialised:
            self.db, self._initialised = level_db, True
        elif level_db < self.db:
            self.db += self.fall_alpha * (level_db - self.db)
        elif not frozen:
            self.db = min(level_db, self.db + self.rise * dt)
        return self.db


def _load_interpreter_class():
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            return __import__(mod, fromlist=["Interpreter"]).Interpreter
        except ImportError:
            continue
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError as exc:
        raise RuntimeError("No TFLite runtime found: pip install ai-edge-litert") from exc


def load_class_map(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [row["display_name"] for row in csv.DictReader(fh)]


class YamnetDetector:
    name = "yamnet"

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.names = load_class_map(cfg.class_map_path)
        missing = [c for c in cfg.bark_classes + cfg.speech_classes if c not in self.names]
        if missing:
            raise ValueError(f"unknown YAMNet class name(s): {missing}")
        self.bark_idx = [self.names.index(c) for c in cfg.bark_classes]
        self.speech_idx = [self.names.index(c) for c in cfg.speech_classes]
        interpreter_cls = _load_interpreter_class()
        self.interp = interpreter_cls(model_path=cfg.model_path, num_threads=cfg.num_threads)
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.in_index, self.in_shape = inp["index"], tuple(inp["shape"])
        if int(np.prod(self.in_shape)) != WINDOW_SAMPLES:
            raise ValueError(f"unexpected YAMNet input shape {self.in_shape}")
        self.out_index = self.interp.get_output_details()[0]["index"]

    def analyse(self, window: np.ndarray, level_db: float) -> Detection:
        self.interp.set_tensor(self.in_index, window.astype(np.float32).reshape(self.in_shape))
        self.interp.invoke()
        scores = self.interp.get_tensor(self.out_index).reshape(-1, len(self.names)).max(axis=0)
        return Detection(
            bark=float(scores[self.bark_idx].max()),
            speech=float(scores[self.speech_idx].max()),
            level_dbfs=level_db,
            label=self.names[int(scores.argmax())],
        )


class HeuristicDetector:
    """Non-ML detector: short, loud, harmonic bursts concentrated in the bark band.

    Cheap enough for a Pi Zero, needs no model, but is fooled more easily by
    other impulsive harmonic sounds (some birds, squeaky gates, shouting).
    """

    name = "heuristic"
    frame = 320   # 20 ms
    hop = 160     # 10 ms

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        freqs = np.fft.rfftfreq(self.frame, 1 / TARGET_RATE)
        self.band = (freqs >= cfg.band_low_hz) & (freqs <= cfg.band_high_hz)
        self.win = np.hanning(self.frame).astype(np.float32)
        self.floor_db = -60.0

    def analyse(self, window: np.ndarray, level_db: float) -> Detection:
        n = 1 + (len(window) - self.frame) // self.hop
        idx = np.arange(self.frame)[None, :] + self.hop * np.arange(n)[:, None]
        frames = window[idx] * self.win
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
        # frame level in dBFS, consistent with dbfs() for a Hann window
        frame_db = 10 * np.log10(power.sum(axis=1) * 2 / (self.frame * np.sum(self.win ** 2)))
        band_ratio = power[:, self.band].sum(axis=1) / power.sum(axis=1)
        bp = power[:, self.band]
        flatness = np.exp(np.mean(np.log(bp), axis=1)) / np.mean(bp, axis=1)

        active = frame_db > self.floor_db + self.cfg.onset_db
        best, reason = 0.0, "quiet"
        for start, end in _runs(active):
            dur_ms = (end - start) * self.hop / TARGET_RATE * 1000 + 10
            seg = slice(start, end)
            if not self.cfg.min_burst_ms <= dur_ms <= self.cfg.max_burst_ms:
                reason = f"burst {dur_ms:.0f}ms"
                continue
            if np.median(band_ratio[seg]) < self.cfg.min_band_ratio:
                reason = "off-band"
                continue
            if np.median(flatness[seg]) > self.cfg.max_flatness:
                reason = "noise-like"
                continue
            prominence = float(frame_db[seg].max() - self.floor_db - self.cfg.onset_db)
            score = 0.5 + 0.5 * min(max(prominence / 15.0, 0.0), 1.0)
            best = max(best, score)
        return Detection(bark=best, speech=0.0, level_dbfs=level_db, label="bark-like" if best else reason)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True values; end is exclusive."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def make_detector(cfg: DetectorConfig):
    if cfg.type == "yamnet":
        return YamnetDetector(cfg)
    if cfg.type == "heuristic":
        return HeuristicDetector(cfg)
    raise ValueError(f"unknown detector type {cfg.type!r}")


__all__ = ["Detection", "HeuristicDetector", "NoiseFloor", "WINDOW_SAMPLES", "WINDOW_SECONDS",
           "YamnetDetector", "dbfs", "make_detector"]
