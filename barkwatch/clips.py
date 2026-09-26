"""Audio clip capture with pre-roll, post-roll and a length cap."""
from __future__ import annotations

import logging
import os
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import TARGET_RATE
from .config import ClipConfig

log = logging.getLogger(__name__)


def write_audio(path: Path, samples: np.ndarray, fmt: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    if fmt == "flac":
        try:
            import soundfile as sf
            sf.write(str(path), pcm, TARGET_RATE, format="FLAC", subtype="PCM_16")
            return path
        except ImportError:
            log.warning("soundfile not installed; writing WAV instead of FLAC")
            path = path.with_suffix(".wav")
    tmp = path.with_suffix(path.suffix + ".part")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(pcm.tobytes())
    os.replace(tmp, path)
    return path


@dataclass
class _Recording:
    key: object
    path: Path
    chunks: list[np.ndarray] = field(default_factory=list)
    samples: int = 0
    stop_at: float | None = None


class ClipRecorder:
    """Keeps a rolling buffer of recent audio so a clip can include the moments before detection.

    `on_done(key, path_or_None)` is called once the post-roll has been captured.
    """

    def __init__(self, cfg: ClipConfig, node_id: str, lookback_s: float,
                 on_done: Callable[[object, Path | None], None]):
        self.cfg = cfg
        self.node_id = node_id
        self.on_done = on_done
        self.ring: deque[tuple[float, np.ndarray]] = deque()
        self.lookback_s = lookback_s + cfg.pre_roll_s
        self.active: list[_Recording] = []
        self.max_samples = int(cfg.max_clip_s * TARGET_RATE)

    def path_for(self, start: float) -> Path:
        dt = datetime.fromtimestamp(start)
        ext = "flac" if self.cfg.format == "flac" else "wav"
        return Path(self.cfg.directory) / dt.strftime("%Y/%m/%d") / \
            f"bark_{self.node_id}_{dt:%Y%m%d_%H%M%S}.{ext}"

    def push(self, t_end: float, samples: np.ndarray) -> None:
        self.ring.append((t_end, samples))
        while self.ring and self.ring[0][0] < t_end - self.lookback_s:
            self.ring.popleft()
        for rec in list(self.active):
            self._append(rec, samples)
            if rec.stop_at is not None and t_end >= rec.stop_at:
                self._finish(rec)

    def start(self, key: object, episode_start: float) -> Path:
        path = self.path_for(episode_start)
        rec = _Recording(key, path)
        for t_end, chunk in self.ring:
            if t_end > episode_start - self.cfg.pre_roll_s:
                self._append(rec, chunk)
        self.active.append(rec)
        return path

    def stop(self, key: object, t_now: float) -> None:
        for rec in self.active:
            if rec.key == key:
                rec.stop_at = t_now + self.cfg.post_roll_s

    def flush(self) -> None:
        for rec in list(self.active):
            self._finish(rec)

    def _append(self, rec: _Recording, chunk: np.ndarray) -> None:
        room = self.max_samples - rec.samples
        if room > 0:
            rec.chunks.append(chunk[:room])
            rec.samples += min(room, len(chunk))

    def _finish(self, rec: _Recording) -> None:
        self.active.remove(rec)
        path: Path | None = None
        if rec.chunks:
            try:
                path = write_audio(rec.path, np.concatenate(rec.chunks), self.cfg.format)
            except OSError:
                log.exception("failed to write clip %s", rec.path)
        self.on_done(rec.key, path)
