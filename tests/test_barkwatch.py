import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from barkwatch.audio import TARGET_RATE, Block, Decimator, dbfs
from barkwatch.config import Config, ReportConfig, load_config
from barkwatch.detectors import Detection, HeuristicDetector, NoiseFloor, WINDOW_SAMPLES
from barkwatch.doa import in_sectors
from barkwatch.events import EventTracker
from barkwatch.pipeline import Pipeline
from barkwatch.reports import build_report, is_quiet, period_bounds, render_markdown
from barkwatch.storage import Store

RNG = np.random.default_rng(0)


def noise(seconds, db=-60.0):
    x = RNG.standard_normal(int(seconds * TARGET_RATE)).astype(np.float32)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)


def bark(ms=180, f0=550.0, db=-20.0):
    n = int(ms / 1000 * TARGET_RATE)
    t = np.arange(n) / TARGET_RATE
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = np.minimum(1, np.minimum(t / 0.01, (t[-1] - t) / 0.05))
    x = (x * env).astype(np.float32)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)


def with_barks(seconds, bark_times, floor_db=-60.0):
    x = noise(seconds, floor_db)
    for bt in bark_times:
        b = bark()
        i = int(bt * TARGET_RATE)
        x[i:i + len(b)] += b[: len(x) - i]
    return x


def blocks(x, hop=0.48, t0=1_000_000.0):
    n = int(hop * TARGET_RATE)
    for i in range(0, len(x) - n + 1, n):
        yield Block(t0 + (i + n) / TARGET_RATE, x[i:i + n])


# -- audio ---------------------------------------------------------------------
def test_dbfs_full_scale_sine():
    t = np.arange(TARGET_RATE) / TARGET_RATE
    assert dbfs(np.sin(2 * np.pi * 1000 * t)) == pytest.approx(-3.01, abs=0.05)


def test_decimator_streaming_matches_batch_and_rejects_alias():
    t = np.arange(48000) / 48000
    x = (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 15000 * t)).astype(np.float32)
    batch = Decimator(3).process(x)
    d = Decimator(3)
    stream = np.concatenate([d.process(x[i:i + 1234]) for i in range(0, len(x), 1234)])
    assert len(batch) == len(stream) == 16000
    np.testing.assert_allclose(batch, stream, atol=1e-5)
    spec = np.abs(np.fft.rfft(batch[1000:]))
    freqs = np.fft.rfftfreq(len(batch) - 1000, 1 / 16000)
    # 15 kHz would alias to 1 kHz; the passband tone must dominate everything else by far
    assert spec[np.argmin(np.abs(freqs - 1000))] > 20 * np.max(spec[np.abs(freqs - 1000) > 50])


# -- detectors -----------------------------------------------------------------
@pytest.fixture
def heuristic():
    d = HeuristicDetector(Config().detector)
    d.floor_db = -60.0
    return d


def window_from(x):
    return x[:WINDOW_SAMPLES]


def test_heuristic_detects_synthetic_bark(heuristic):
    w = window_from(with_barks(1.0, [0.3]))
    assert heuristic.analyse(w, dbfs(w)).bark >= 0.5


@pytest.mark.parametrize("name,signal", [
    ("broadband noise burst", lambda: np.concatenate([noise(0.3), noise(0.2, -20), noise(0.5)])),
    ("long tone (alarm)", lambda: noise(1.0) + np.pad(bark(ms=900, f0=1000), (800, 800))),
    ("low rumble", lambda: noise(1.0) + (0.1 * np.sin(2 * np.pi * 80 * np.arange(16000) / 16000))),
])
def test_heuristic_rejects_non_barks(heuristic, name, signal):
    w = window_from(np.asarray(signal(), dtype=np.float32))
    assert heuristic.analyse(w, dbfs(w)).bark < 0.5, name


def test_noise_floor_tracks_down_fast_up_slow_and_freezes():
    nf = NoiseFloor()
    nf.update(-60, 0.5, False)
    assert nf.update(-80, 0.5, False) < -65
    before = nf.db
    nf.update(-20, 0.5, False)
    assert nf.db - before < 0.1
    frozen = nf.db
    nf.update(-20, 0.5, True)
    assert nf.db == frozen


# -- episode tracking ----------------------------------------------------------
def tracker(**kw):
    cfg = Config().events
    for k, v in kw.items():
        setattr(cfg, k, v)
    return EventTracker(cfg, threshold=0.5, min_level_dbfs=-65, hop_s=0.5, window_s=1.0)


HIT, MISS = Detection(0.9, 0.0, -30.0), Detection(0.0, 0.0, -60.0)


def test_single_hit_is_discarded():
    tr = tracker(min_hits=2, confirm_window_s=5)
    events = tr.update(0, HIT)
    for t in range(1, 60):
        events += tr.update(t, MISS)
    assert events == []


def test_episode_lifecycle_and_metrics():
    tr = tracker(min_hits=2, episode_gap_s=10)
    evs = []
    for t, d in [(0, HIT), (1, HIT), (2, MISS), (5, HIT)] + [(t, MISS) for t in range(6, 20)]:
        evs += [(e.kind, t) for e in tr.update(t, d)]
    assert evs == [("started", 1), ("ended", 16)]
    assert tr.flush() == []


def test_quiet_hit_below_min_level_is_ignored():
    tr = tracker()
    assert not tr.is_hit(Detection(0.9, 0.0, -80.0))


def test_long_episode_is_split():
    tr = tracker(min_hits=1, max_episode_s=30, episode_gap_s=10)
    kinds = []
    for t in range(0, 70, 2):
        kinds += [e.kind for e in tr.update(t, HIT)]
    assert kinds.count("started") >= 2 and kinds.count("ended") >= 1


def test_doa_sectors():
    assert in_sectors(250, [[200, 290]])
    assert not in_sectors(100, [[200, 290]])
    assert in_sectors(10, [[330, 30]]) and not in_sectors(200, [[330, 30]])
    assert in_sectors(None, [[0, 10]]) and in_sectors(5, [])


# -- end-to-end pipeline --------------------------------------------------------
@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.detector.type = "heuristic"
    c.detector.threshold = 0.5
    c.storage.database = str(tmp_path / "db.sqlite")
    c.clips.directory = str(tmp_path / "clips")
    c.clips.format = "wav"
    c.mqtt.enabled = False
    return c


def test_pipeline_records_episode_clip_and_stats(cfg):
    store = Store(cfg.storage, cfg.node_id)
    pipe = Pipeline(cfg, store)
    x = with_barks(40, [10 + 0.7 * i for i in range(8)])
    t0 = datetime.now().timestamp() - 60
    pipe.run(blocks(x, t0=t0), live=False)
    rows = store.db.execute("SELECT * FROM episodes").fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["clip_status"] == "saved" and os.path.exists(r["clip_path"])
    assert 3 < r["duration_s"] < 10 and r["hits"] >= 4
    stats = store.live_stats()
    assert stats["episodes_today"] + stats["episodes_last_hour"] >= 1
    assert sum(stats["hourly_7d"]) == 1


def test_speech_suppresses_clip(cfg):
    store = Store(cfg.storage, cfg.node_id)

    class FakeDetector:
        name = "fake"

        def analyse(self, window, level):
            loud = level > -40
            return Detection(0.9 if loud else 0.0, 0.8 if loud else 0.0, level)

    pipe = Pipeline(cfg, store, detector=FakeDetector())
    pipe.run(blocks(with_barks(30, [10, 10.7, 11.4])), live=False)
    r = store.db.execute("SELECT * FROM episodes").fetchone()
    assert r["clip_status"] == "suppressed_speech" and r["clip_path"] is None
    assert not list(Path(cfg.clips.directory).rglob("*.wav"))


def test_label_command_excludes_false_positive(cfg):
    store = Store(cfg.storage, cfg.node_id)
    pipe = Pipeline(cfg, store)
    pipe.run(blocks(with_barks(30, [10 + 0.7 * i for i in range(5)]), t0=datetime.now().timestamp() - 40),
             live=False)
    ep_id = store.db.execute("SELECT id FROM episodes").fetchone()[0]
    assert store.live_stats()["episodes_last_hour"] == 1
    pipe.handle_command({"action": "label", "id": ep_id, "label": "not_bark"})
    assert store.live_stats()["episodes_last_hour"] == 0


def test_retention_deletes_old_clips_but_keeps_flagged(cfg, tmp_path):
    store = Store(cfg.storage, cfg.node_id)
    now = datetime.now().timestamp()
    ids = []
    for age_days, keep in [(40, False), (40, True), (1, False)]:
        p = tmp_path / f"c{age_days}{keep}.wav"
        p.write_bytes(b"x")
        i = store.insert_episode(now - age_days * 86400, "t", str(p), "saved")
        store.set_keep(i, keep)
        ids.append((i, p))
    store.purge(now)
    assert not ids[0][1].exists() and ids[1][1].exists() and ids[2][1].exists()


# -- reports -------------------------------------------------------------------
def test_quiet_hours_weekday_vs_weekend():
    rc = ReportConfig()  # weekday 22:00-07:00, weekend 23:00-09:00
    assert is_quiet(datetime(2026, 9, 21, 6, 30), rc)       # Monday 06:30
    assert not is_quiet(datetime(2026, 9, 21, 7, 30), rc)   # Monday 07:30
    assert is_quiet(datetime(2026, 9, 26, 8, 30), rc)       # Saturday 08:30
    assert not is_quiet(datetime(2026, 9, 25, 22, 30), rc)  # Friday 22:30 -> weekend profile starts 23:00
    assert is_quiet(datetime(2026, 9, 24, 22, 30), rc)      # Thursday 22:30


def test_period_bounds():
    s, e, label = period_bounds("weekly", date(2026, 9, 24))
    assert s == datetime(2026, 9, 21) and e == datetime(2026, 9, 28) and label == "2026-W39"
    s, e, label = period_bounds("monthly", date(2026, 12, 5))
    assert e == datetime(2027, 1, 1) and label == "2026-12"


def test_weekly_report_finds_recurring_saturday_morning(cfg):
    store = Store(cfg.storage, cfg.node_id)
    monday = datetime(2026, 9, 21)
    for day in range(7):
        times = [monday + timedelta(days=day, hours=18)]
        if day in (1, 3, 5, 6):   # Tue, Thu, Sat, Sun at ~06:45
            times.append(monday + timedelta(days=day, hours=6, minutes=45 + day // 2))
        for t in times:
            i = store.insert_episode(t.timestamp(), "t", None, "disabled")
            store.finish_episode(i, end_ts=t.timestamp() + 30, duration_s=30, hits=10, bark_seconds=5,
                                 peak_dbfs=-25, max_score=0.9, mean_score=0.7, max_speech=0, doa_deg=None)
    r = build_report(store, cfg.reports, "weekly", date(2026, 9, 23))
    assert r["episodes"] == 11
    assert r["hourly"][6] == 4 and r["hourly"][18] == 7
    assert r["quiet_hour_episodes"] == 4
    slots = {p["slot"]: p for p in r["recurring"]}
    assert slots["18:00-18:14"]["days"] == 7
    assert slots["06:45-06:59"]["weekdays"] == "Tue, Thu, Sat, Sun"
    md = render_markdown(r)
    assert "Recurring patterns" in md and "18:00" in md


def test_example_config_loads():
    root = os.path.dirname(os.path.dirname(__file__))
    cfg = load_config(os.path.join(root, "config.example.yaml"))
    assert cfg.detector.type in ("yamnet", "heuristic")


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("detector:\n  treshold: 0.4\n")
    with pytest.raises(ValueError, match="treshold"):
        load_config(p)


@pytest.mark.skipif(not os.environ.get("YAMNET_MODEL"), reason="set YAMNET_MODEL and YAMNET_CLASS_MAP")
def test_yamnet_scores_synthetic_bark_above_noise():
    from barkwatch.detectors import YamnetDetector
    c = Config().detector
    c.model_path, c.class_map_path = os.environ["YAMNET_MODEL"], os.environ["YAMNET_CLASS_MAP"]
    det = YamnetDetector(c)
    quiet = noise(1.0)[:WINDOW_SAMPLES]
    assert det.analyse(quiet, dbfs(quiet)).bark < 0.2
