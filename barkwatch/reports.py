"""Daily / weekly / monthly reports with pattern analysis."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .config import QuietHours, ReportConfig
from .storage import Store

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SLOT_MIN = 15


def period_bounds(period: str, ref: date) -> tuple[datetime, datetime, str]:
    if period == "daily":
        start = datetime.combine(ref, time())
        return start, start + timedelta(days=1), ref.isoformat()
    if period == "weekly":
        monday = ref - timedelta(days=ref.weekday())
        start = datetime.combine(monday, time())
        iso = monday.isocalendar()
        return start, start + timedelta(days=7), f"{iso[0]}-W{iso[1]:02d}"
    if period == "monthly":
        start = datetime(ref.year, ref.month, 1)
        nxt = datetime(ref.year + (ref.month == 12), ref.month % 12 + 1, 1)
        return start, nxt, f"{ref:%Y-%m}"
    raise ValueError(f"unknown period {period!r}")


def previous_ref(period: str, ref: date) -> date:
    start, _, _ = period_bounds(period, ref)
    return (start - timedelta(days=1)).date()


def _hm(s: str) -> time:
    h, m = s.split(":")[:2]
    return time(int(h), int(m))


def is_quiet(dt: datetime, cfg: ReportConfig) -> bool:
    """Quiet hours span midnight. Evening times use the profile of the following morning,
    so Friday and Saturday nights (and Saturday/Sunday mornings) use the weekend profile."""
    t = dt.time()
    if t >= time(12):
        prof: QuietHours = cfg.weekend_quiet_hours if (dt + timedelta(days=1)).weekday() >= 5 else cfg.quiet_hours
        return t >= _hm(prof.start)
    prof = cfg.weekend_quiet_hours if dt.weekday() >= 5 else cfg.quiet_hours
    return t < _hm(prof.end)


def build_report(store: Store, cfg: ReportConfig, period: str, ref: date) -> dict:
    start, end, label = period_bounds(period, ref)
    rows = store.episodes_between(start.timestamp(), end.timestamp())
    eps = [dict(r) for r in rows]
    for e in eps:
        e["dt"] = datetime.fromtimestamp(e["start_ts"])
        e["quiet"] = is_quiet(e["dt"], cfg)

    ndays = (end - start).days
    days = []
    for i in range(ndays):
        d = (start + timedelta(days=i)).date()
        de = [e for e in eps if e["dt"].date() == d]
        days.append({
            "date": d.isoformat(), "weekday": WEEKDAYS[d.weekday()], "episodes": len(de),
            "bark_minutes": round(sum(e["bark_seconds"] or 0 for e in de) / 60, 1),
            "quiet_hour_episodes": sum(e["quiet"] for e in de),
            "first": min((e["dt"] for e in de), default=None),
            "last": max((e["dt"] for e in de), default=None),
        })

    hourly, weekday = [0] * 24, [0] * 7
    slot_days: dict[int, set] = {}
    for e in eps:
        hourly[e["dt"].hour] += 1
        weekday[e["dt"].weekday()] += 1
        slot = (e["dt"].hour * 60 + e["dt"].minute) // SLOT_MIN
        slot_days.setdefault(slot, set()).add(e["dt"].date())

    recurring = []
    if ndays > 1:
        min_days = max(2, math.ceil(0.4 * ndays))
        for slot, ds in sorted(slot_days.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(ds) >= min_days:
                h, m = divmod(slot * SLOT_MIN, 60)
                recurring.append({"slot": f"{h:02d}:{m:02d}-{h:02d}:{m + SLOT_MIN - 1:02d}",
                                  "days": len(ds), "of_days": ndays,
                                  "weekdays": ", ".join(WEEKDAYS[i] for i in sorted({d.weekday() for d in ds}))})

    durations = [e["duration_s"] or 0 for e in eps]
    weekend = [d for d in days if d["weekday"] in ("Sat", "Sun")]
    weekdays_ = [d for d in days if d["weekday"] not in ("Sat", "Sun")]
    firsts = [d["first"] for d in days if d["first"]]
    prev_start, prev_end, _ = period_bounds(period, previous_ref(period, ref))
    prev_n = len(store.episodes_between(prev_start.timestamp(), prev_end.timestamp()))

    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    def fmt_t(dt):
        return dt.strftime("%H:%M") if dt else None

    earliest_morning = [d["first"] for d in days if d["first"] and d["first"].hour < 12]
    report = {
        "period": period, "label": label, "start": start.isoformat(), "end": end.isoformat(),
        "node": store.node, "generated": datetime.now().isoformat(timespec="seconds"),
        "episodes": len(eps),
        "previous_period_episodes": prev_n,
        "change_vs_previous": len(eps) - prev_n,
        "bark_minutes": round(sum(e["bark_seconds"] or 0 for e in eps) / 60, 1),
        "quiet_hour_episodes": sum(e["quiet"] for e in eps),
        "longest_episode_s": round(max(durations, default=0), 1),
        "median_episode_s": round(statistics.median(durations), 1) if durations else 0,
        "peak_dbfs": round(max((e["peak_dbfs"] or -120 for e in eps), default=-120), 1) if eps else None,
        "days_with_barking": sum(1 for d in days if d["episodes"]),
        "avg_episodes_weekday": avg([d["episodes"] for d in weekdays_]),
        "avg_episodes_weekend": avg([d["episodes"] for d in weekend]),
        "earliest_morning_bark": fmt_t(min(earliest_morning, key=lambda d: d.time())) if earliest_morning else None,
        "typical_first_bark": fmt_t(sorted(firsts, key=lambda d: d.time())[len(firsts) // 2]) if firsts else None,
        "peak_hours": [h for h, _ in Counter(dict(enumerate(hourly))).most_common(3) if hourly[h]],
        "hourly": hourly, "weekday": weekday, "recurring": recurring[:10],
        "days": [{**d, "first": fmt_t(d["first"]), "last": fmt_t(d["last"])} for d in days],
        "longest": [{"id": e["id"], "start": e["dt"].isoformat(timespec="seconds"),
                     "duration_s": round(e["duration_s"] or 0, 1), "peak_dbfs": e["peak_dbfs"],
                     "clip": e["clip_path"] if e["clip_status"] == "saved" else None}
                    for e in sorted(eps, key=lambda e: -(e["duration_s"] or 0))[:5]],
        "_episodes": eps,
    }
    return report


def _bar(n: int, mx: int, width: int = 30) -> str:
    return "█" * (math.ceil(n / mx * width) if mx else 0)


def render_markdown(r: dict) -> str:
    L = [f"# Bark report – {r['period']} {r['label']} ({r['node']})", "",
         f"_Generated {r['generated']}. Episodes labelled `not_bark` are excluded._", "",
         "| Metric | Value |", "|---|---|",
         f"| Episodes | {r['episodes']} (previous period: {r['previous_period_episodes']}, "
         f"change {r['change_vs_previous']:+d}) |",
         f"| Time spent barking | {r['bark_minutes']} min |",
         f"| Episodes during quiet hours | {r['quiet_hour_episodes']} |",
         f"| Days with barking | {r['days_with_barking']} / {len(r['days'])} |",
         f"| Longest / median episode | {r['longest_episode_s']} s / {r['median_episode_s']} s |",
         f"| Loudest episode | {r['peak_dbfs']} dBFS |",
         f"| Avg episodes per weekday / weekend day | {r['avg_episodes_weekday']} / {r['avg_episodes_weekend']} |",
         f"| Earliest morning bark | {r['earliest_morning_bark'] or '–'} |",
         f"| Typical first bark of the day | {r['typical_first_bark'] or '–'} |",
         f"| Peak hours | {', '.join(f'{h:02d}:00' for h in r['peak_hours']) or '–'} |", ""]
    if len(r["days"]) > 1:
        L += ["## Per day", "", "| Date | Day | Episodes | Bark min | Quiet-hour | First | Last |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {d['date']} | {d['weekday']} | {d['episodes']} | {d['bark_minutes']} | "
              f"{d['quiet_hour_episodes']} | {d['first'] or '–'} | {d['last'] or '–'} |" for d in r["days"]]
        L.append("")
    mx = max(r["hourly"]) if r["hourly"] else 0
    L += ["## Time of day", "", "```"]
    L += [f"{h:02d}:00 {_bar(n, mx)} {n or ''}".rstrip() for h, n in enumerate(r["hourly"])]
    L += ["```", ""]
    if len(r["days"]) >= 7:
        mx = max(r["weekday"])
        L += ["## Day of week", "", "```"]
        L += [f"{WEEKDAYS[i]} {_bar(n, mx)} {n or ''}".rstrip() for i, n in enumerate(r["weekday"])]
        L += ["```", ""]
    if r["recurring"]:
        L += ["## Recurring patterns", "", "Time slots with barking on many different days:", ""]
        L += [f"- **{p['slot']}** on {p['days']} of {p['of_days']} days ({p['weekdays']})" for p in r["recurring"]]
        L.append("")
    if r["longest"]:
        L += ["## Longest episodes", "", "| ID | Start | Duration | Peak | Clip |", "|---|---|---|---|---|"]
        L += [f"| {e['id']} | {e['start']} | {e['duration_s']} s | {e['peak_dbfs']} dBFS | "
              f"{e['clip'] or '–'} |" for e in r["longest"]]
        L.append("")
    return "\n".join(L)


def write_report(r: dict, directory: str) -> list[Path]:
    out = Path(directory) / r["period"]
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{r['node']}_{r['label']}"
    md, js, cs = out / f"{stem}.md", out / f"{stem}.json", out / f"{stem}.csv"
    md.write_text(render_markdown(r), encoding="utf-8")
    js.write_text(json.dumps(summary(r), indent=2), encoding="utf-8")
    cols = ["id", "start_ts", "end_ts", "duration_s", "hits", "bark_seconds", "peak_dbfs", "max_score",
            "mean_score", "doa_deg", "quiet", "label", "clip_status", "clip_path"]
    with cs.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start_local"] + cols)
        for e in r["_episodes"]:
            w.writerow([e["dt"].isoformat(timespec="seconds")] + [e.get(c) for c in cols])
    return [md, js, cs]


def summary(r: dict) -> dict:
    """JSON-safe report without the raw episode list (also published to MQTT)."""
    return {k: v for k, v in r.items() if not k.startswith("_")}
