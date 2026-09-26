"""Command-line interface: `python -m barkwatch <command>`."""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from .config import load_config

log = logging.getLogger("barkwatch")


def cmd_run(cfg, args) -> None:
    from .audio import LiveSource
    from .pipeline import Pipeline
    from .storage import Store

    store = Store(cfg.storage, cfg.node_id)
    doa = None
    if cfg.doa.enabled:
        from .doa import RespeakerDoa
        doa = RespeakerDoa()
    mqtt = None
    if cfg.mqtt.enabled:
        from .mqtt import MqttLink
        mqtt = MqttLink(cfg.mqtt, cfg.node_id)
    pipe = Pipeline(cfg, store, mqtt=mqtt, doa=doa)
    if mqtt:
        mqtt.on_command = pipe.commands.put
        mqtt.start()
        mqtt.active(False)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        pipe.run(LiveSource(cfg.audio).blocks())
    except (KeyboardInterrupt, SystemExit):
        log.info("stopping")
    finally:
        if mqtt:
            mqtt.active(False)
            mqtt.stop()


def cmd_analyze(cfg, args) -> None:
    """Run the detector over a WAV file; prints per-window scores and the resulting episodes."""
    from .audio import WavSource
    from .pipeline import Pipeline
    from .storage import Store

    cfg.clips.enabled = bool(args.clips)
    store = Store(cfg.storage, cfg.node_id) if args.store else None
    pipe = Pipeline(cfg, store)
    t0 = time.time() if args.store else 0.0
    for block in WavSource(args.wav, cfg.audio, start_time=t0).blocks():
        det = pipe.process(block, live=False)
        if args.verbose:
            print(f"{block.t_end - t0:8.2f}s level={det.level_dbfs:6.1f} bark={det.bark:.2f} "
                  f"speech={det.speech:.2f} {'HIT ' if pipe.tracker.is_hit(det) else '    '}{det.label}")
    pipe.shutdown()
    for ep in pipe.episodes:
        print(f"episode {ep.start - t0:7.1f}s .. {ep.end - t0:7.1f}s  dur={ep.duration_s:5.1f}s "
              f"hits={ep.hits:3d} peak={ep.peak_dbfs:6.1f}dBFS max={ep.max_score:.2f} speech={ep.max_speech:.2f}")
    print(f"{len(pipe.episodes)} episode(s)")


def cmd_report(cfg, args) -> None:
    from .reports import build_report, render_markdown, summary, write_report
    from .storage import Store

    store = Store(cfg.storage, cfg.node_id)
    ref = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    rep = build_report(store, cfg.reports, args.period, ref)
    paths = write_report(rep, cfg.reports.directory)
    if args.print:
        print(render_markdown(rep))
    for p in paths:
        log.info("wrote %s", p)
    if args.publish and cfg.mqtt.enabled:
        from .mqtt import MqttLink
        link = MqttLink(cfg.mqtt, cfg.node_id)
        link.client.connect(cfg.mqtt.host, cfg.mqtt.port)
        link.client.loop_start()
        data = summary(rep)
        data.pop("days", None)  # keep the retained payload small
        link.client.publish(f"{link.base}/report/{args.period}", json.dumps(data), qos=1,
                            retain=True).wait_for_publish(5)
        link.client.loop_stop()
        link.client.disconnect()


def cmd_purge(cfg, args) -> None:
    from .storage import Store
    print(json.dumps(Store(cfg.storage, cfg.node_id).purge()))


def cmd_label(cfg, args) -> None:
    from .storage import Store
    store = Store(cfg.storage, cfg.node_id)
    ok = store.set_label(args.id, None if args.label == "clear" else args.label)
    if args.keep is not None:
        ok = store.set_keep(args.id, args.keep) and ok
    sys.exit(0 if ok else f"episode {args.id} not found")


def cmd_list(cfg, args) -> None:
    from .storage import Store
    store = Store(cfg.storage, cfg.node_id)
    rows = store.db.execute("SELECT * FROM episodes ORDER BY start_ts DESC LIMIT ?", (args.limit,)).fetchall()
    for r in rows:
        print(f"#{r['id']:<6} {time.strftime('%a %Y-%m-%d %H:%M:%S', time.localtime(r['start_ts']))} "
              f"dur={r['duration_s'] or 0:6.1f}s peak={r['peak_dbfs'] or 0:6.1f} "
              f"label={r['label'] or '-':9} clip={r['clip_status']} {r['clip_path'] or ''}")


def cmd_devices(cfg, args) -> None:
    import sounddevice as sd
    print(sd.query_devices())


def cmd_calibrate(cfg, args) -> None:
    """Print levels for a while so thresholds can be chosen for the actual location."""
    import numpy as np

    from .audio import LiveSource, dbfs
    levels = []
    end = time.time() + args.seconds
    for block in LiveSource(cfg.audio).blocks():
        lvl = dbfs(block.samples)
        levels.append(lvl)
        print(f"{lvl:6.1f} dBFS {'#' * max(0, int((lvl + 90) / 2))}")
        if time.time() > end:
            break
    a = np.array(levels)
    print(f"\nmin {a.min():.1f}  p10 {np.percentile(a, 10):.1f}  median {np.median(a):.1f}  "
          f"p90 {np.percentile(a, 90):.1f}  max {a.max():.1f} dBFS")
    print("Suggested detector.min_level_dbfs: about p10 + 10 dB, then refine with `analyze`.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="barkwatch", description=__doc__)
    p.add_argument("-c", "--config", help="YAML config (default: ./config.yaml if present)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="live detection")
    a = sub.add_parser("analyze", help="run detection over a WAV file")
    a.add_argument("wav")
    a.add_argument("--store", action="store_true", help="write episodes to the database")
    a.add_argument("--clips", action="store_true", help="also write clips")
    r = sub.add_parser("report", help="generate a report")
    r.add_argument("--period", choices=["daily", "weekly", "monthly"], default="daily")
    r.add_argument("--date", help="any date inside the period (default: yesterday)")
    r.add_argument("--publish", action="store_true", help="publish the summary to MQTT")
    r.add_argument("--print", action="store_true")
    sub.add_parser("purge", help="apply retention rules")
    lb = sub.add_parser("label", help="label an episode (feedback / false positives)")
    lb.add_argument("id", type=int)
    lb.add_argument("label", choices=["bark", "not_bark", "other", "clear"])
    lb.add_argument("--keep", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                    help="true = exempt the clip from retention")
    ls = sub.add_parser("list", help="list recent episodes")
    ls.add_argument("--limit", type=int, default=20)
    sub.add_parser("devices", help="list audio devices")
    cal = sub.add_parser("calibrate", help="measure ambient levels")
    cal.add_argument("--seconds", type=int, default=60)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose and args.cmd == "run" else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = args.config or ("config.yaml" if Path("config.yaml").exists() else None)
    cfg = load_config(path if args.cmd != "devices" else None)
    globals()[f"cmd_{args.cmd}"](cfg, args)


if __name__ == "__main__":
    main()
