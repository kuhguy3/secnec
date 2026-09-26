"""Glues audio, detector, episode tracker, clips, storage and MQTT together."""
from __future__ import annotations

import logging
import queue
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import TARGET_RATE, Block, dbfs
from .clips import ClipRecorder
from .config import Config
from .detectors import WINDOW_SAMPLES, WINDOW_SECONDS, Detection, NoiseFloor, make_detector
from .doa import in_sectors
from .events import Episode, EventTracker, TrackerEvent
from .storage import Store

log = logging.getLogger(__name__)

LABELS = {"bark", "not_bark", "other", None}


class Pipeline:
    def __init__(self, cfg: Config, store: Store | None, mqtt=None, detector=None, doa=None):
        self.cfg = cfg
        self.store = store
        self.mqtt = mqtt
        self.detector = detector or make_detector(cfg.detector)
        self.doa = doa
        self.floor = NoiseFloor()
        self.window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self.tracker = EventTracker(cfg.events, cfg.detector.threshold, cfg.detector.min_level_dbfs,
                                    cfg.audio.hop_seconds, WINDOW_SECONDS)
        self.clips = ClipRecorder(cfg.clips, cfg.node_id,
                                  lookback_s=cfg.events.confirm_window_s + WINDOW_SECONDS,
                                  on_done=self._clip_done) if cfg.clips.enabled else None
        self.clip_speech: dict[int, float] = {}
        self.commands: queue.Queue[dict] = queue.Queue()
        self.episodes: list[Episode] = []          # finished episodes (used by `analyze`)
        self._last_live = self._last_stats = 0.0
        self._last_bark_level: float | None = None
        self._minute: int | None = None
        self._minute_acc: list[tuple[float, float]] = []   # (level, score)
        self._minute_hits = 0
        self._samples_seen = 0

    # -- main loop ----------------------------------------------------------
    def run(self, blocks: Iterable[Block], live: bool = True) -> None:
        if self.store:
            self.store.close_dangling()
        try:
            for block in blocks:
                self.process(block, live)
        finally:
            self.shutdown()

    def process(self, block: Block, live: bool = True) -> Detection:
        t = block.t_end
        x = block.samples
        self.window = np.concatenate([self.window[len(x):], x])
        self._samples_seen += len(x)
        if self.clips:
            self.clips.push(t, x)

        level = dbfs(self.window)
        floor_db = self.floor.update(level, self.cfg.audio.hop_seconds, frozen=self.tracker.current is not None)
        if self._samples_seen < WINDOW_SAMPLES:
            det = Detection(0.0, 0.0, level, "warmup")
        elif level < floor_db + self.cfg.detector.energy_gate_db and self.tracker.current is None:
            det = Detection(0.0, 0.0, level, "gated")
        else:
            if hasattr(self.detector, "floor_db"):
                self.detector.floor_db = floor_db
            det = self.detector.analyse(self.window, level)

        doa = self.doa.angle() if (self.doa and self.tracker.is_hit(det)) else None
        doa_ok = in_sectors(doa, self.cfg.doa.accept_sectors)
        for ev in self.tracker.update(t, det, doa, doa_ok):
            self._on_event(ev, t)
        if self.tracker.is_hit(det) and doa_ok:
            self._last_bark_level = level
        for key in self.clip_speech:
            self.clip_speech[key] = max(self.clip_speech[key], det.speech)

        log.debug("t=%.2f level=%.1f floor=%.1f bark=%.2f speech=%.2f %s",
                  t, level, floor_db, det.bark, det.speech, det.label)
        self._accumulate_minute(t, det, floor_db)
        if live:
            self._periodic(t, det, floor_db)
            self._drain_commands()
        return det

    def shutdown(self) -> None:
        for ev in self.tracker.flush():
            self._on_event(ev, ev.episode.last_hit)
        if self.clips:
            self.clips.flush()
        self._flush_minute()

    # -- episodes -----------------------------------------------------------
    def _on_event(self, ev: TrackerEvent, t: float) -> None:
        ep = ev.episode
        if ev.kind == "started":
            clip_path, status = None, "disabled"
            if self.clips:
                status = "recording"
            if self.store:
                ep.db_id = self.store.insert_episode(ep.start, self.detector.name, None, status)
            else:
                ep.db_id = id(ep)
            if self.clips:
                clip_path = self.clips.start(ep.db_id, ep.start)
                self.clip_speech[ep.db_id] = ep.max_speech
                if self.store:
                    self.store.set_clip(ep.db_id, str(clip_path), status)
            log.info("bark episode #%s started", ep.db_id)
            if self.mqtt:
                self.mqtt.active(True)
                self.mqtt.event({"event_type": "started", "id": ep.db_id,
                                 "start": _iso(ep.start), "clip": str(clip_path) if clip_path else None})
        else:
            if self.store:
                self.store.finish_episode(
                    ep.db_id, end_ts=ep.end, duration_s=ep.duration_s, hits=ep.hits,
                    bark_seconds=ep.bark_seconds, peak_dbfs=ep.peak_dbfs, max_score=ep.max_score,
                    mean_score=ep.mean_score, max_speech=ep.max_speech, doa_deg=ep.doa_deg)
            if self.clips:
                self.clips.stop(ep.db_id, t)
            self.episodes.append(ep)
            log.info("bark episode #%s ended: %.0fs, %d hits, peak %.1f dBFS",
                     ep.db_id, ep.duration_s, ep.hits, ep.peak_dbfs)
            if self.mqtt:
                self.mqtt.active(False)
                self.mqtt.event({"event_type": "ended", "id": ep.db_id, "start": _iso(ep.start),
                                 "end": _iso(ep.end), "duration_s": round(ep.duration_s, 1),
                                 "bark_seconds": round(ep.bark_seconds, 1),
                                 "peak_dbfs": round(ep.peak_dbfs, 1), "max_score": round(ep.max_score, 2),
                                 "doa_deg": ep.doa_deg})
                self._publish_stats()

    def _clip_done(self, key: int, path: Path | None) -> None:
        speech = self.clip_speech.pop(key, 0.0)
        status = "saved" if path else "failed"
        if path and self.cfg.privacy.drop_clips_with_speech and speech >= self.cfg.privacy.speech_threshold:
            path.unlink(missing_ok=True)
            path, status = None, "suppressed_speech"
            log.info("clip for episode #%s discarded: speech detected (%.2f)", key, speech)
        if self.store:
            self.store.set_clip(key, str(path) if path else None, status)

    # -- periodic publishing --------------------------------------------------
    def _periodic(self, t: float, det: Detection, floor_db: float) -> None:
        if not self.mqtt:
            return
        if t - self._last_live >= self.cfg.mqtt.live_interval_s:
            self._last_live = t
            self.mqtt.live({"level": round(det.level_dbfs, 1), "floor": round(floor_db, 1),
                            "score": round(det.bark, 2),
                            "bark_level": round(self._last_bark_level, 1) if self._last_bark_level else None})
        if t - self._last_stats >= self.cfg.mqtt.stats_interval_s:
            self._last_stats = t
            self._publish_stats()

    def _publish_stats(self) -> None:
        if self.mqtt and self.store:
            self.mqtt.stats(self.store.live_stats())

    def _accumulate_minute(self, t: float, det: Detection, floor_db: float) -> None:
        minute = int(t // 60 * 60)
        if self._minute is not None and minute != self._minute:
            self._flush_minute()
        self._minute = minute
        self._minute_floor = floor_db
        self._minute_acc.append((det.level_dbfs, det.bark))
        self._minute_hits += int(self.tracker.is_hit(det))

    def _flush_minute(self) -> None:
        if not self._minute_acc or not self.store:
            return
        levels = np.array([a[0] for a in self._minute_acc])
        mean_db = 10 * np.log10(np.mean(10 ** (levels / 10)))
        self.store.upsert_minute(self._minute, round(float(mean_db), 1), round(float(levels.max()), 1),
                                 round(self._minute_floor, 1),
                                 round(max(a[1] for a in self._minute_acc), 2), self._minute_hits)
        self._minute_acc, self._minute_hits = [], 0

    # -- commands from Home Assistant ----------------------------------------
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return
            self.handle_command(cmd)

    def handle_command(self, cmd: dict) -> None:
        action = cmd.get("action")
        if not self.store:
            return
        if action == "label" and cmd.get("label") in LABELS:
            ok = self.store.set_label(int(cmd["id"]), cmd.get("label"))
            log.info("label episode #%s -> %s (%s)", cmd.get("id"), cmd.get("label"), "ok" if ok else "not found")
        elif action == "keep":
            self.store.set_keep(int(cmd["id"]), bool(cmd.get("keep", True)))
        elif action != "refresh":
            log.warning("unknown command %r", cmd)
            return
        self._publish_stats()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


__all__ = ["Pipeline", "TARGET_RATE"]
