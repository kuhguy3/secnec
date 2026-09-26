"""SQLite event log, live statistics and retention."""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import StorageConfig

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY,
    node         TEXT NOT NULL,
    start_ts     REAL NOT NULL,          -- unix epoch seconds
    end_ts       REAL,                   -- NULL while the episode is ongoing
    duration_s   REAL,
    hits         INTEGER,
    bark_seconds REAL,
    peak_dbfs    REAL,
    max_score    REAL,
    mean_score   REAL,
    max_speech   REAL,
    doa_deg      INTEGER,
    detector     TEXT,
    clip_path    TEXT,
    clip_status  TEXT,                   -- recording | saved | suppressed_speech | disabled | failed | deleted
    label        TEXT,                   -- NULL (unreviewed) | bark | not_bark | other
    keep         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS episodes_start ON episodes(start_ts);

CREATE TABLE IF NOT EXISTS minute_stats (
    node       TEXT NOT NULL,
    minute_ts  INTEGER NOT NULL,         -- unix epoch, floored to the minute
    mean_dbfs  REAL,
    max_dbfs   REAL,
    floor_dbfs REAL,
    max_score  REAL,
    hits       INTEGER,
    PRIMARY KEY (node, minute_ts)
);
"""

# Episodes the user marked as false positives are excluded from all statistics.
COUNTED = "(label IS NULL OR label != 'not_bark')"


def local_midnight(ts: float) -> float:
    d = datetime.fromtimestamp(ts)
    return datetime(d.year, d.month, d.day).timestamp()


class Store:
    def __init__(self, cfg: StorageConfig, node: str):
        self.cfg = cfg
        self.node = node
        Path(cfg.database).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cfg.database, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # -- writes -------------------------------------------------------------
    def insert_episode(self, start_ts: float, detector: str, clip_path: str | None,
                       clip_status: str) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes (node, start_ts, detector, clip_path, clip_status) VALUES (?,?,?,?,?)",
            (self.node, start_ts, detector, clip_path, clip_status))
        self.db.commit()
        return int(cur.lastrowid)

    def finish_episode(self, ep_id: int, *, end_ts: float, duration_s: float, hits: int,
                       bark_seconds: float, peak_dbfs: float, max_score: float, mean_score: float,
                       max_speech: float, doa_deg: int | None) -> None:
        self.db.execute(
            """UPDATE episodes SET end_ts=?, duration_s=?, hits=?, bark_seconds=?, peak_dbfs=?,
               max_score=?, mean_score=?, max_speech=?, doa_deg=? WHERE id=?""",
            (end_ts, duration_s, hits, bark_seconds, peak_dbfs, max_score, mean_score,
             max_speech, doa_deg, ep_id))
        self.db.commit()

    def set_clip(self, ep_id: int, path: str | None, status: str) -> None:
        self.db.execute("UPDATE episodes SET clip_path=?, clip_status=? WHERE id=?", (path, status, ep_id))
        self.db.commit()

    def set_label(self, ep_id: int, label: str | None) -> bool:
        cur = self.db.execute("UPDATE episodes SET label=? WHERE id=?", (label, ep_id))
        self.db.commit()
        return cur.rowcount > 0

    def set_keep(self, ep_id: int, keep: bool) -> bool:
        cur = self.db.execute("UPDATE episodes SET keep=? WHERE id=?", (int(keep), ep_id))
        self.db.commit()
        return cur.rowcount > 0

    def upsert_minute(self, minute_ts: int, mean_dbfs: float, max_dbfs: float, floor_dbfs: float,
                      max_score: float, hits: int) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO minute_stats VALUES (?,?,?,?,?,?,?)",
            (self.node, minute_ts, mean_dbfs, max_dbfs, floor_dbfs, max_score, hits))
        self.db.commit()

    def close_dangling(self) -> None:
        """Episodes left open by a crash get their end set to their start."""
        self.db.execute("UPDATE episodes SET end_ts=start_ts, duration_s=0 WHERE end_ts IS NULL")
        self.db.execute("UPDATE episodes SET clip_status='failed' WHERE clip_status='recording'")
        self.db.commit()

    # -- reads --------------------------------------------------------------
    def episodes_between(self, start_ts: float, end_ts: float, counted_only: bool = True) -> list[sqlite3.Row]:
        where = f"AND {COUNTED}" if counted_only else ""
        return self.db.execute(
            f"SELECT * FROM episodes WHERE node=? AND start_ts>=? AND start_ts<? {where} ORDER BY start_ts",
            (self.node, start_ts, end_ts)).fetchall()

    def live_stats(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        today = local_midnight(now)

        def agg(since: float) -> tuple[int, float]:
            row = self.db.execute(
                f"""SELECT COUNT(*), COALESCE(SUM(bark_seconds),0) FROM episodes
                    WHERE node=? AND start_ts>=? AND {COUNTED}""", (self.node, since)).fetchone()
            return int(row[0]), float(row[1])

        n_today, s_today = agg(today)
        n_hour, _ = agg(now - 3600)
        n_7d, s_7d = agg(now - 7 * 86400)
        n_30d, _ = agg(now - 30 * 86400)

        hourly = [0] * 24
        weekday = [0] * 7
        for r in self.db.execute(
                f"SELECT start_ts FROM episodes WHERE node=? AND start_ts>=? AND {COUNTED}",
                (self.node, now - 30 * 86400)):
            dt = datetime.fromtimestamp(r[0])
            weekday[dt.weekday()] += 1
            if r[0] >= now - 7 * 86400:
                hourly[dt.hour] += 1

        last = self.db.execute(
            f"SELECT * FROM episodes WHERE node=? AND {COUNTED} ORDER BY start_ts DESC LIMIT 1",
            (self.node,)).fetchone()
        first_today = self.db.execute(
            f"SELECT MIN(start_ts) FROM episodes WHERE node=? AND start_ts>=? AND {COUNTED}",
            (self.node, today)).fetchone()[0]

        def iso(ts: float | None) -> str | None:
            return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds") if ts else None

        return {
            "episodes_today": n_today,
            "bark_minutes_today": round(s_today / 60, 1),
            "episodes_last_hour": n_hour,
            "episodes_7d": n_7d,
            "bark_minutes_7d": round(s_7d / 60, 1),
            "episodes_30d": n_30d,
            "first_bark_today": iso(first_today),
            "last_episode": iso(last["start_ts"]) if last else None,
            "last_episode_id": last["id"] if last else None,
            "last_episode_duration_s": round(last["duration_s"] or 0, 1) if last else None,
            "last_episode_peak_dbfs": round(last["peak_dbfs"], 1) if last and last["peak_dbfs"] else None,
            "last_episode_clip": last["clip_path"] if last else None,
            "hourly_7d": hourly,
            "weekday_30d": weekday,
        }

    # -- retention ----------------------------------------------------------
    def purge(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        cfg = self.cfg
        deleted_clips = 0

        def drop_clips(rows) -> None:
            nonlocal deleted_clips
            for r in rows:
                try:
                    Path(r["clip_path"]).unlink(missing_ok=True)
                except OSError:
                    log.exception("could not delete %s", r["clip_path"])
                    continue
                self.db.execute("UPDATE episodes SET clip_status='deleted' WHERE id=?", (r["id"],))
                deleted_clips += 1

        saved = "clip_status='saved' AND keep=0"
        drop_clips(self.db.execute(
            f"SELECT id, clip_path FROM episodes WHERE {saved} AND label='not_bark' AND start_ts<?",
            (now - cfg.false_positive_clip_retention_days * 86400,)).fetchall())
        drop_clips(self.db.execute(
            f"SELECT id, clip_path FROM episodes WHERE {saved} AND start_ts<?",
            (now - cfg.clip_retention_days * 86400,)).fetchall())

        # Enforce the storage cap, oldest first.
        rows = self.db.execute(
            f"SELECT id, clip_path FROM episodes WHERE {saved} ORDER BY start_ts DESC").fetchall()
        total, over = 0, []
        for r in rows:
            p = Path(r["clip_path"])
            total += p.stat().st_size if p.exists() else 0
            if total > cfg.max_clip_storage_mb * 1024 * 1024:
                over.append(r)
        drop_clips(over)

        ev = self.db.execute("DELETE FROM episodes WHERE keep=0 AND start_ts<?",
                             (now - cfg.event_retention_days * 86400,)).rowcount
        ms = self.db.execute("DELETE FROM minute_stats WHERE minute_ts<?",
                             (now - cfg.minute_stats_retention_days * 86400,)).rowcount
        self.db.commit()
        return {"clips_deleted": deleted_clips, "episodes_deleted": ev, "minute_rows_deleted": ms}


def day_bounds(day: datetime) -> tuple[float, float]:
    start = datetime(day.year, day.month, day.day)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()
