"""Turns per-window detections into bark episodes.

An *episode* is a burst of barking: it is confirmed after `min_hits` positive
windows within `confirm_window_s`, and it ends after `episode_gap_s` without
a hit. Isolated single hits (a door slam classified as a bark) are discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import EventConfig
from .detectors import Detection


@dataclass
class Episode:
    first_hit: float
    start: float                  # first_hit minus the analysis window length
    last_hit: float
    hop_s: float
    hits: int = 0
    peak_dbfs: float = -120.0
    max_score: float = 0.0
    score_sum: float = 0.0
    max_speech: float = 0.0
    doa: list[int] = field(default_factory=list)
    confirmed: bool = False
    db_id: int | None = None      # set by the pipeline once stored

    @property
    def end(self) -> float:
        return self.last_hit

    @property
    def duration_s(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def bark_seconds(self) -> float:
        """Approximate time spent barking: positive windows x hop."""
        return self.hits * self.hop_s

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.hits if self.hits else 0.0

    @property
    def doa_deg(self) -> int | None:
        if not self.doa:
            return None
        return sorted(self.doa)[len(self.doa) // 2]

    def add(self, t: float, det: Detection, doa: int | None) -> None:
        self.hits += 1
        self.last_hit = t
        self.peak_dbfs = max(self.peak_dbfs, det.level_dbfs)
        self.max_score = max(self.max_score, det.bark)
        self.score_sum += det.bark
        if doa is not None:
            self.doa.append(doa)


@dataclass
class TrackerEvent:
    kind: str       # "started" | "ended"
    episode: Episode


class EventTracker:
    def __init__(self, cfg: EventConfig, threshold: float, min_level_dbfs: float,
                 hop_s: float, window_s: float):
        self.cfg = cfg
        self.threshold = threshold
        self.min_level = min_level_dbfs
        self.hop_s = hop_s
        self.window_s = window_s
        self.current: Episode | None = None

    @property
    def active(self) -> bool:
        return self.current is not None and self.current.confirmed

    def is_hit(self, det: Detection) -> bool:
        return det.bark >= self.threshold and det.level_dbfs >= self.min_level

    def update(self, t: float, det: Detection, doa: int | None = None,
               doa_ok: bool = True) -> list[TrackerEvent]:
        out: list[TrackerEvent] = []
        ep = self.current
        if ep is not None:
            ep.max_speech = max(ep.max_speech, det.speech)
            expired = (not ep.confirmed and t - ep.first_hit > self.cfg.confirm_window_s) or \
                      (t - ep.last_hit > self.cfg.episode_gap_s) or \
                      (ep.confirmed and t - ep.start > self.cfg.max_episode_s)
            if expired:
                if ep.confirmed:
                    out.append(TrackerEvent("ended", ep))
                self.current = ep = None

        if self.is_hit(det) and doa_ok:
            if ep is None:
                ep = self.current = Episode(first_hit=t, start=t - self.window_s, last_hit=t,
                                            hop_s=self.hop_s, max_speech=det.speech)
            ep.add(t, det, doa)
            if not ep.confirmed and ep.hits >= self.cfg.min_hits:
                ep.confirmed = True
                out.append(TrackerEvent("started", ep))
        return out

    def flush(self) -> list[TrackerEvent]:
        """Close any open episode (shutdown / end of file)."""
        ep, self.current = self.current, None
        return [TrackerEvent("ended", ep)] if ep is not None and ep.confirmed else []
