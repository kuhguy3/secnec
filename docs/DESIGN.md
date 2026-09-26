# BarkWatch – Design Document

A local-first system that **measures** a neighbouring dog's barking, **documents** it, and **protects your
sleep**. It never emits anything toward the dog: no ultrasonic "deterrents", no sprays, no shocks, no
sounds played into the garden. The only outputs are data, notifications and masking noise *inside your own
bedroom*.

Code in this repository implements the MVP and most of the production design:

| Path | What it is |
|---|---|
| `barkwatch/` | Python service: capture → detect → episodes → SQLite + clips → MQTT → reports |
| `homeassistant/packages/barkwatch.yaml` | Helpers, templates, scripts and 12 automations (alerts, sleep protection) |
| `homeassistant/dashboards/barkwatch.yaml` | Two-view Lovelace dashboard, built-in cards only |
| `deploy/`, `scripts/`, `Dockerfile`, `docker-compose.yml` | systemd units + timers, installer, model download, noise generator |
| `tests/` | Unit and end-to-end tests (synthetic audio; real YAMNet test when the model is present) |

---

## 1. Goals, non-goals and constraints

**Goals:** detect barking with high precision, count and time it, keep verifiable clips, report on
patterns (daily/weekly/monthly, time of day, day of week, recurring slots), integrate with Home Assistant,
alert only past configurable thresholds, and optionally mask barking in the bedroom.

**Non-goals:** anything that affects the dog. Also not a certified noise-measurement instrument (see §14,
*Evidence*).

**Constraints:** local-first (no cloud, no accounts), open source, runs on a Raspberry Pi, survives
reboots/power cuts, respects the privacy of neighbours and passers-by.

---

## 2. Architecture

```
                        ┌───────────────────────── Raspberry Pi (garden-facing window) ─────────────────────────┐
                        │                                                                                         │
 ┌──────────────┐  USB  │  ┌────────────┐   16 kHz mono   ┌───────────────┐  0.975 s window / 0.48 s hop         │
 │ Microphone / │──────▶│  │ LiveSource │ ──────────────▶ │ Level + noise │──┐                                    │
 │  mic array   │  I2S  │  │ (PortAudio │  (decimated     │ floor tracker │  │ energy gate: skip inference        │
 └──────┬───────┘       │  │  + FIR)    │   from 48 kHz)  └───────────────┘  │ when ambient-quiet                 │
        │ DOA (opt.)    │  └─────┬──────┘                                     ▼                                    │
        └──────────────▶│        │ raw audio        ┌──────────────────────────────────────┐                      │
                        │        │                  │ Detector                              │                      │
                        │        │                  │  • YAMNet TFLite (521 AudioSet classes)│ bark score,         │
                        │        │                  │  • or heuristic (band/burst/flatness) │ speech score        │
                        │        ▼                  └───────────────────┬──────────────────┘                      │
                        │ ┌─────────────┐  pre-roll ring                ▼                                          │
                        │ │ ClipRecorder│◀──── start/stop ──── ┌──────────────────┐   DOA sector filter (opt.)      │
                        │ │ FLAC, 3 s   │                      │ EventTracker      │                                │
                        │ │ pre/post    │                      │ hits → episodes   │  min_hits in confirm window,   │
                        │ └──────┬──────┘                      │ (hysteresis)      │  episode_gap, max length       │
                        │        │ privacy: drop clip if       └────────┬─────────┘                                │
                        │        │ speech detected                      │                                          │
                        │        ▼                                      ▼                                          │
                        │ ┌────────────────────────────────────────────────────┐    ┌─────────────────────────┐   │
                        │ │ SQLite (WAL): episodes, minute_stats, labels, keep │───▶│ Reports (systemd timers) │   │
                        │ └─────────────┬──────────────────────────────────────┘    │ daily/weekly/monthly     │   │
                        │               │ live stats (60 s)                          │ .md .csv .json + MQTT   │   │
                        │               ▼                                            └─────────────────────────┘   │
                        │        ┌─────────────┐  retention purge (03:30)                                          │
                        │        │  MqttLink   │  ◀── cmd topic (labels, keep, refresh)                            │
                        │        └──────┬──────┘                                                                   │
                        └───────────────┼──────────────────────────────────────────────────────────────────────────┘
                                        │ MQTT (TLS/auth), HA discovery, LWT availability
                                        ▼
 ┌───────────────────────────────────────── Home Assistant ─────────────────────────────────────────────┐
 │ Mosquitto broker ─▶ MQTT integration ─▶ entities: binary_sensor barking, sensors, event entity       │
 │                                                                                                        │
 │ Package: helpers (thresholds, quiet hours, volumes) · template sensors (quiet hours, day off,         │
 │          target volume) · scripts (noise fade-in/out) · automations (alerts, logbook, sleep protection)│
 │                                                                                                        │
 │ Dashboard ◀─ recorder/long-term statistics        Media browser ◀─ clips via network storage (NFS/SMB) │
 │                                                                                                        │
 │ Outputs: mobile notifications (actionable) · bedroom media_player (white/pink/brown noise)             │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Why a separate service rather than an HA custom integration? Continuous audio DSP/ML is CPU-heavy and
latency-sensitive; it belongs next to the microphone. The service can run on a dedicated Pi at the window
while HA runs elsewhere, and a crash or update of one does not affect the other. MQTT discovery makes it
feel native in HA anyway.

---

## 3. Hardware recommendations

| Role | MVP (reuse what you have) | Production |
|---|---|---|
| Compute | Existing Pi 4 (2 GB+) or Pi 3B+ (heuristic or 1-thread YAMNet) | **Raspberry Pi 5 (4 GB)** dedicated to audio, active cooler, official 27 W PSU |
| Storage | A2 microSD, "high endurance" type | USB 3 SSD or NVMe HAT (clips + SQLite WAL writes wear SD cards) |
| Mic | Any USB mic that does 16/48 kHz | See §4 – ReSpeaker array or measurement mic |
| Placement | Inside, on the windowsill facing the garden | Same, or mic in a weatherproof box on the wall, Pi indoors |
| Power | Normal PSU | Small UPS HAT or UPS for Pi + router, `Persistent=true` timers catch up after outages |
| Bedroom output | Any HA-controllable speaker (Sonos, Chromecast/Nest, HomePod via AirPlay, Music Assistant, ESPHome speaker, Snapcast client on a Pi Zero 2 W) | Dedicated small speaker at the bed, not the room's main speaker |

**Placement matters more than hardware.** A mic behind closed double glazing loses 25–35 dB and most
high-frequency content; a slightly open window, a trickle vent, or a mic outside in an enclosure makes a
bigger difference than any model upgrade. Keep the mic away from fridge/boiler hum, the router's fan,
and your own TV. Face the neighbour's garden.

**Outdoor enclosure:** IP65 junction box, mic capsule behind an acoustic vent membrane (e.g. Gore-type
PTFE vent) facing down, a foam windscreen inside the box, USB cable through a cable gland into the house.
Wind noise is the #1 source of false *level* readings outdoors; the windscreen is not optional.

---

## 4. Microphones

| Mic | Type | Approx. price | Pros | Cons |
|---|---|---|---|---|
| Generic USB lavalier/desk mic | USB, mono | €10–25 | Plug and play, fine for the MVP | Unknown frequency response, AGC on some models (disable it: `alsamixer`) |
| INMP441 / SPH0645 breakout | I2S MEMS on the GPIO header | €3–8 | Cheap, good SNR (~61–65 dB), low noise | Needs a device-tree overlay, fiddly wiring, one mic only |
| **ReSpeaker USB Mic Array v2.0** (XMOS XVF-3000) | USB, 4-mic array, on-board DSP | ~€65–80 | Beamforming, noise suppression, AEC, **direction of arrival** (supported in `barkwatch/doa.py`), 16 kHz native | DSP is tuned for speech (may attenuate barks slightly, use raw channels 1–4 if so), older product |
| ReSpeaker XVF3800 USB 4-mic array | USB, 4-mic array | ~€50–70 | Newer XMOS DSP, better far-field | DOA parameters differ from v2.0 (the DOA module needs adapting) |
| Dayton Audio UMM-6 / miniDSP UMIK-1 | USB measurement mic, calibration file | ~€80–120 | Flat response, **calibrated level** → real dB SPL, credible for noise diaries | Omni only, no DOA, needs windscreen outdoors |

**Recommendation:** MVP with any USB mic. Production: **ReSpeaker array** if distinguishing *which* dog
(yours vs the neighbour's, left vs right garden) matters, **measurement mic** if calibrated levels
matter (evidence, comparing against local noise rules). Both can run side by side as two nodes
(`node_id: garden_array`, `node_id: garden_spl`).

### Does a mic array or a bigger model improve accuracy?

| Upgrade | Improves | Cost / complexity | When it's worth it |
|---|---|---|---|
| **Mic array + DOA** | Attribution (which direction), rejects your own dog/TV/street behind you. Beamforming adds a few dB SNR for distant barks. | €50–80, USB permissions (udev rule), sector calibration | Several dogs nearby, or your own dog in the house |
| **YAMNet (vs heuristic)** | Separates barks from birds, shouts, gates, cars, fireworks | ~4 MB model, a few % CPU on Pi 4/5 | Always, unless on a Pi Zero |
| **Bigger model (PANNs CNN14, EfficientAT, BEATs)** | Higher AudioSet mAP (roughly 0.43–0.48 vs YAMNet's ~0.31) → fewer misses in noise | 80–300 MB, 0.3–2 s per window on a Pi 5; ONNX Runtime | Only as a *second stage* on candidate windows |
| **Custom head on YAMNet embeddings** | Learns *this* dog, *this* garden, *your* false-positive sources | Needs 200–1000 labelled windows (the `label` command collects them) | After 2–4 weeks of data – the best accuracy/cost ratio |

**DOA implementation (included):** set `doa.enabled: true`. On every candidate hit the service reads the
array's `DOAANGLE` register over USB (vendor control transfer, as in Seeed's `tuning.py`), stores the
median angle per episode (`episodes.doa_deg`), and, if `accept_sectors` is set, ignores hits from other
directions. Calibrate by standing at the neighbour's fence and clapping while watching `doa_deg` in
`barkwatch list`. Needs `pip install pyusb` and a udev rule:
`SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0660", GROUP="plugdev"`.

---

## 5. Raspberry Pi software stack

| Layer | Choice | Notes |
|---|---|---|
| OS | Raspberry Pi OS Lite 64-bit (Bookworm/Trixie) | Headless; `unattended-upgrades` for security patches |
| Audio | ALSA + PortAudio (`sounddevice`) | No PulseAudio/PipeWire needed; disable mic AGC in `alsamixer` |
| Runtime | Python 3.11+ venv, NumPy | |
| Inference | **LiteRT** (`ai-edge-litert`, the successor of `tflite-runtime`), XNNPACK delegate | Falls back to `tflite_runtime` or full TensorFlow |
| Model | YAMNet float32 TFLite (Apache-2.0), 521 AudioSet classes | `scripts/download_model.sh` |
| Storage | SQLite in WAL mode; FLAC clips via `soundfile` | Single-file DB, trivial backups |
| Messaging | Mosquitto (HA add-on or container), `paho-mqtt` v2 | Auth required, TLS if crossing untrusted networks |
| Scheduling | systemd service + timers (reports, purge) | Or the `barkwatch-cron` container |
| Supervision | systemd `Restart=always`, MQTT LWT → HA "unavailable" → offline alert | |
| Optional | Grafana + InfluxDB (or HA's own long-term statistics), Syncthing/restic for backups, Tailscale/WireGuard for remote access | Never expose HA or the clip share directly to the internet |

---

## 6. Detection

### 6.1 Signal chain

1. Capture at 16 kHz (or 48 kHz decimated ×3 through a windowed-sinc FIR, streaming, no aliasing).
2. Analysis window 0.975 s (YAMNet's native input), hop 0.48 s → ~2 decisions/second.
3. RMS level in dBFS and an **adaptive noise floor** (falls quickly, rises at 0.05 dB/s, frozen during
   episodes so a long barking session doesn't become "ambient").
4. **Energy gate:** if the window is within `energy_gate_db` of the floor, skip inference (a quiet night
   costs almost no CPU).
5. Detector → bark score and (YAMNet) speech score.
6. Hit = `score ≥ threshold` **and** `level ≥ min_level_dbfs` (**and** DOA in sector, if enabled).
7. EventTracker turns hits into **episodes**.

### 6.2 Non-ML detector (`detector.type: heuristic`)

Barks are short (≈60–600 ms), loud relative to ambient, harmonic (fundamental ≈300–1000 Hz, strong
harmonics up to ~4 kHz), and have sharp onsets. Per 20 ms frame (10 ms hop) the detector computes level,
share of energy in 350–4000 Hz, and spectral flatness; it then looks for bursts above floor + `onset_db`
and accepts those with the right duration, band ratio ≥ `min_band_ratio`, and flatness ≤ `max_flatness`
(wind, rain and traffic are flat/noise-like; barks are peaky). Score = 0.5–1.0 by prominence.

*Pros:* no model, ~1 % CPU on a Pi Zero 2 W, fully explainable. *Cons:* birds (magpies, crows), shouts,
squeaky gates and door slams can pass. Use it for the MVP on weak hardware, or as a pre-filter.

### 6.3 ML detector (`detector.type: yamnet`, default)

YAMNet (MobileNet-v1 trained on AudioSet) scores 521 classes. BarkWatch takes the max over
`Dog, Bark, Yip, Howl, Bow-wow, Growling, Whimper (dog)` as the bark score and the max over speech-like
classes as the speech score. On the ESC-50 samples used during development it scored real barks at
0.8–0.9 and birds/fireworks near 0.

Accuracy levers, in order of effectiveness:

1. **Temporal confirmation** – `min_hits: 2` within `confirm_window_s: 10`: a single false window never
   becomes an episode.
2. **Level gate** – `min_level_dbfs` drops faint distant dogs (other streets). Tune with
   `barkwatch calibrate` then `barkwatch analyze recording.wav -v`.
3. **Threshold** – 0.3–0.5. Lower = more recall, higher = more precision.
4. **DOA sectors** (array only).
5. **Feedback loop** – every false positive marked in HA ("Not a bark" button / notification action)
   is excluded from statistics and becomes training data.
6. **Custom classifier head** (roadmap §19): logistic regression on YAMNet's 1024-d embeddings trained on
   your labelled clips. This typically removes the site-specific false positives (your own TV, a
   particular bird) that a generic model can't know about.

### 6.4 Hybrid (recommended long-term)

`cheap gate (energy) → YAMNet → (optional) custom head or big model on candidate windows only`.
The gate keeps the CPU idle at night; the expensive stages run a few hundred times a day at most.

### 6.5 Measuring accuracy

Record 2–3 typical mornings (`arecord -f S16_LE -r 16000 -c 1 morning.wav`), mark the barks by hand in
Audacity (label track → export), run `barkwatch analyze morning.wav -v`, and compare. Track precision
(episodes that really were barking) via labels: `SELECT label, COUNT(*) FROM episodes GROUP BY label`.
Aim for ≥95 % precision before trusting weekly reports; recall matters less for pattern analysis.

---

## 7. Event model and metrics

| Metric | Definition |
|---|---|
| **Hit** | One 0.975 s window classified as barking |
| **Episode** | ≥`min_hits` hits within `confirm_window_s`, ended by `episode_gap_s` (30 s) of no hits; split after `max_episode_s` |
| Duration | First-hit window start → last hit |
| Bark seconds | hits × hop (approximate time spent actually barking, not counting pauses) |
| Frequency | Episodes per hour/day/week; bark minutes per day |
| Intensity | Peak dBFS per episode (dB SPL with a calibrated mic + offset) |
| Quiet-hour episode | Starts inside quiet hours (weekday 22:00–07:00, weekend 23:00–09:00 by default; Friday/Saturday nights use the weekend profile) |
| Recurring slot | A 15-minute slot with barking on ≥40 % of the days in the period (≥2 days) |

---

## 8. Home Assistant integration

MQTT discovery creates one device, **BarkWatch garden**, with:

| Entity | Source | Use |
|---|---|---|
| `binary_sensor.barkwatch_garden_barking` | `…/active`, retained | Triggers sleep protection; logbook |
| `event.barkwatch_garden_episode` | `…/event` (`started` / `ended` + payload) | Per-episode automations |
| `sensor.…_level`, `…_noise_floor`, `…_bark_level`, `…_bark_score` | `…/live` every 5 s | Graphs, volume scaling |
| `sensor.…_episodes_today` (total_increasing), `…_bark_minutes_today` | `…/stats` every 60 s | Thresholds, long-term statistics |
| `sensor.…_episodes_last_hour`, `…_episodes_7_days` (+ attributes: `hourly_7d`, `weekday_30d`, `last_episode_id`…), `…_episodes_30_days` | `…/stats` | Alerts, heatmaps |
| `sensor.…_first_bark_today`, `…_last_episode`, `…_last_episode_duration` | `…/stats` | Dashboard, summaries |
| `sensor.…_weekly_report` (+ full summary as attributes) | `…/report/weekly`, retained | Weekly notification, pre-emptive noise |

Commands (HA → service) on `barkwatch/garden/cmd`:
`{"action":"label","id":123,"label":"not_bark"}`, `{"action":"keep","id":123}`, `{"action":"refresh"}`.

HA integrations used: **MQTT**, **Mobile App** (actionable notifications), **Media Source / network
storage** (clips), **Workday** (holidays count as days off), **Template**, **Recorder/Statistics**,
optional **Music Assistant**, **Sonos/Cast/DLNA**, **ESPHome** (bed-presence sensor), **Calendar**
(e.g. mark "early start" days).

**Listening to clips in HA:** export `/opt/barkwatch/data/clips` read-only over NFS or SMB and add it in
*Settings → System → Storage → Add network storage* with usage "Media". Clips then appear in the Media
browser under `barkwatch/YYYY/MM/DD/`.

**Recorder:** `bark_score` and `level` update every 5 s; exclude `bark_score` from the recorder and keep
`level` (useful), or raise `mqtt.live_interval_s`.

---

## 9. Reusable open-source projects

| Project | How it helps |
|---|---|
| **YAMNet** (TensorFlow Models, Apache-2.0) | The default classifier used here |
| **LiteRT / tflite-runtime** | On-device inference |
| **Frigate NVR** – audio detection | If you already have a camera with a mic: Frigate runs YAMNet on camera audio and can emit `bark` events to HA via MQTT with zero extra hardware. Less control over episodes/reports; good as a second opinion |
| **BirdNET-Pi / BirdNET-Go** | Reference architecture for "Pi + mic + classifier + SQLite + clips + web UI"; worth reading for retention and spectrogram UI ideas |
| **PANNs** (CNN14), **EfficientAT**, **BEATs** | Stronger AudioSet models for a second stage |
| **Edge Impulse** (free tier; not fully open) / **TensorFlow Model Maker-style transfer learning** | Training a custom bark head or a tiny model for ESP32 |
| **ESPHome** + ESP32-S3 + INMP441 | Future: cheap extra "listening posts"; bed occupancy sensors (pressure/mmWave) for the sleep automations |
| **Mosquitto**, **Node-RED** (optional), **Grafana/InfluxDB** (optional) | Messaging, visual flows, long-horizon dashboards |
| **Audacity**, **Sonic Visualiser**, **Label Studio** | Reviewing and labelling clips |
| **SoX** | Generating white/pink/brown noise (`scripts/generate_noise.sh`) |
| **Music Assistant** | Robust looping/queueing on many speaker brands |

---

## 10. Storage and retention

| Data | Size | Default retention | Rationale |
|---|---|---|---|
| Episodes (SQLite) | ~300 B/row | 730 days | Year-over-year patterns; tiny |
| Minute stats (level/floor/max score) | ~60 B/min → ~30 MB/year | 90 days | Quiet-hour ambient evidence, debugging |
| Clips (FLAC 16 kHz mono, ~20–30 KB/s) | 3 s pre + episode + 3 s post, capped 60 s → typically 0.2–1.5 MB | **30 days** | Verification; privacy-minimising |
| Clips labelled `not_bark` | | 2 days | Kept briefly for review/training export |
| Clips flagged `keep` | | Forever (until unflagged) | Evidence, training set |
| Clips with detected speech | | **Never stored** | Privacy (§14) |
| Hard cap | `max_clip_storage_mb` (4 GB) | Oldest non-kept clips deleted first | Protects the SD/SSD |
| Reports (.md/.csv/.json) | KBs | Forever | They are the product |

Rough volume: 20 episodes/day × 0.6 MB ≈ 12 MB/day ≈ 360 MB for 30 days.

Durability: SQLite WAL + atomic clip writes (`.part` → rename); episodes left open by a crash are closed
on start-up. **Backups:** nightly `sqlite3 barkwatch.db ".backup /backup/barkwatch.db"` plus `restic`
of `data/reports` and kept clips to a NAS. Encrypt the backup repository.

---

## 11. Dashboard mockups

The implementation is `homeassistant/dashboards/barkwatch.yaml` (sections view, built-in cards only).

```
┌─ BarkWatch ─────────────────────────────── [Overview] [Sleep & settings] ──────────────────────────────┐
│ RIGHT NOW                              │ TRENDS                                │ REPORTS & REVIEW          │
│ ┌──────────┐┌──────────┐┌──────────┐   │ Episodes per day (30 d)               │ Last weekly report 2026-W39│
│ │🐕 Barking ││🌙 Quiet  ││ Today    │   │ ▁▂▁▅▃▁▁▇▂▁▁▃▁▁▆▂▁▁▂▅▁▁▃▁▁█▂▁▃▅        │ 41 episodes (+6), 38 min, │
│ │   OFF    ││ hours ON ││ 7 ep.    │   │                                       │ 9 in quiet hours.          │
│ └──────────┘└──────────┘└──────────┘   │ Bark minutes per week                 │ Weekday 4.6 · weekend 9.0  │
│ ┌──────────┐┌──────────┐┌──────────┐   │ ██ ███ █ ████                         │ Earliest morning: 06:12    │
│ │ 6.5 min  ││ Last hr 2││ First    │   │                                       │ • 06:45-06:59 on 4/7 days  │
│ │ today    ││          ││ 06:47    │   │ Time of day (7 d)                     │   (Tue, Thu, Sat, Sun)     │
│ └──────────┘└──────────┘└──────────┘   │ 06h ████████████████████ 9            │ • 18:00-18:14 on 7/7 days  │
│      ╭──── Garden level ────╮          │ 07h ████████ 4                        ├───────────────────────────┤
│      │   ◜‾‾‾‾‾‾‾◝  -52 dBFS │          │ 12h ██ 1                              │ Recent episodes (logbook) │
│      ╰───────────────────────╯          │ 18h ███████████████ 7                 │ 07:02 ended · 41 s -24 dB │
│ Level vs ambient (24 h)                 │                                       │ 06:47 started             │
│ -20┤      ▲    ▲▲                       │ Day of week (30 d)                    ├───────────────────────────┤
│ -50┤~~~~~~╨~~~~╨╨~~~~~~~~~~~~~~~~        │ Mon ████ 6   Sat ███████████ 17       │ Last episode 07:02 41 s   │
│ -70┤______________ ambient ____          │ Tue ██████ 9 Sun ████████████ 19      │ Peak -24 dBFS [Not a bark]│
│     00   06   12   18   24               │                                       │ [🎵 Listen to clips]       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─ Sleep & settings ───────────────────────────────────────────────────────────────────────────────────────┐
│ SLEEP PROTECTION                                   │ QUIET HOURS & ALERTS                                  │
│ [●] Sleep protection   [●] Weekends/holidays only  │ Quiet start 22:00  end (work) 07:00  end (off) 09:00 │
│ [●] Bedroom occupied   [ ] Stop after silence      │ Day off: ON (Saturday)                                │
│ [●] Pre-emptive start  Colour: [Pink ▾]            │ [●] Notifications   [ ] Quiet hours only              │
│ Base vol ──●────── 0.18   Max vol ────●── 0.40     │ Alert ≥ 5 episodes/hour   ≥ 30 bark min/day           │
│ +0.01 per 2 dB above ambient → target now 0.26     │ Cooldown 60 min                                       │
│ ▶ Bedroom speaker – Pink noise  ▮▮  vol 0.26       │                                                       │
│ [▶ Test noise] [■ Fade out]                        │                                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Optional HACS upgrades: `apexcharts-card` for a true 7×24 heatmap and stacked weekday/weekend charts,
`mushroom` for compact chips.

---

## 12. Automations (implemented in the package)

| Automation | Trigger → condition → action |
|---|---|
| Threshold alert | last-hour episodes or today's bark minutes change → above helper thresholds, notifications on, (optionally) quiet hours, cooldown elapsed → actionable notification |
| Notification actions | "Not a bark" → MQTT `label not_bark`; "Mute 2 h" → toggles notifications |
| Quiet-hours logbook | episode `ended` during quiet hours → logbook entry with duration/peak |
| Detector offline | barking sensor `unavailable` for 10 min → notification |
| Daily summary | 21:00 if any episodes → counts, first bark time, 7-day total |
| Weekly summary | new weekly report published → totals, change vs last week, recurring slot |
| Sleep: start | barking ON → sleep protection on, quiet hours, bedroom occupied, (weekends-only ⇒ day off), speaker idle → fade in noise |
| Sleep: volume | target volume rises → raise speaker volume over ~10 s (never lowers mid-night) |
| Sleep: pre-emptive | 20 min before the typical first-bark time, on days off → start noise |
| Sleep: stop | N minutes without barking (optional) **or** quiet hours end **or** bedroom unoccupied → 60 s fade out |
| Weekend arming | 21:30 before Saturday/Sunday → turn on sleep protection + bedroom occupied |

Bedroom occupancy: drive `input_boolean.barkwatch_bedroom_occupied` from whatever you trust – a bed
pressure sensor or mmWave sensor (ESPHome), phone charging in the bedroom after 22:00, or a "Good night"
button. The weekend-arming automation is a simple default.

---

## 13. Sleep protection design notes

* **Colour:** *pink* noise (−3 dB/octave) masks the 0.5–2 kHz bark band better than brown; *brown*
  (−6 dB/octave) sounds softer and many people find it more pleasant; *white* is harshest. Default is
  brown for comfort; switch to pink if barks still get through.
* **Reactive vs pre-emptive:** detection needs 1–2 barks before confirming an episode, so a purely
  reactive system starts *after* the first bark (the one that may have woken you). The pre-emptive
  automation uses the weekly report's typical first-bark time; many people simply run the noise all
  night at a low level on weekends and let BarkWatch only *raise* it.
* **Fade in/out:** abrupt changes wake people. Scripts fade in over ~20 s, ramp up over ~10 s, fade out
  over ~60 s.
* **Volume scaling:** `target = base + 0.01 × (bark_level − ambient) / dB_per_step`, capped at
  `max_volume`. Calibrate once: play noise at `max_volume`, measure at the pillow with a phone SPL app,
  and keep it at or below roughly 45–50 dB(A). Protect your hearing and anyone else in the room.
* **Speaker choice:** a dedicated small speaker near the bed allows lower overall volume than a room
  speaker. Avoid Bluetooth speakers that auto-sleep; prefer Wi-Fi/wired players HA controls reliably.
* **Don't hijack:** automations won't start if the speaker is already playing something.

---

## 14. Privacy and compliance review

> This section is engineering guidance, not legal advice. Rules differ by country and state.

**What the system can capture:** everything audible at the mic – including neighbours' conversations,
passers-by, and your own household. Audio is legally and ethically more sensitive than video in many
jurisdictions.

**Legal hot spots:**
* **Recording conversations** – several jurisdictions criminalise recording private spoken words without
  consent (e.g. Germany §201 StGB; "two-party consent" US states such as California). A dog is not a
  party to a conversation, but incidental speech is.
* **Data protection** – in the EU/UK, recordings that capture people outside your own property may fall
  outside the "household exemption" (CJEU *Ryneš*, C-212/13), making you a data controller with duties
  (purpose limitation, minimisation, retention limits, subject access). UK ICO domestic CCTV guidance
  explicitly calls audio recording highly privacy-intrusive.
* **Tenancy / HOA rules** may restrict external devices.

**Built-in mitigations (privacy by design):**
1. **No continuous recording.** Audio exists only in a RAM ring buffer (~15 s) unless an episode is
   confirmed.
2. **Speech suppression:** with YAMNet, any clip in which speech-like classes exceed
   `privacy.speech_threshold` is deleted before it is ever kept (`clip_status = suppressed_speech`); the
   episode's metadata (time, duration, level) is still logged.
3. **Short clips, short retention:** ≤60 s, 30 days by default, false positives 2 days.
4. **Metadata over audio:** statistics and reports contain no audio.
5. **Local only:** no cloud APIs; MQTT with authentication; clip share read-only and LAN-only.
6. **Access control:** config readable only by the service user; clips directory owned by the service
   user; HA users without admin rights can't browse media if you don't give them the dashboard.
7. **Encryption at rest** (optional): put `data/` on a LUKS volume or use an encrypted backup
   repository.
8. **Direction:** aim the mic at the garden, not the street or the neighbour's windows; use DOA sectors
   to ignore the street.

**Also consider:** tell the neighbour the device exists (and that it records only barking), put a note in
your household privacy notice if you have visitors, and disable clips entirely (`clips.enabled: false`)
if local law is strict – counts and levels alone are usually enough to see patterns.

**Evidence:** logs and clips are useful for a calm conversation and, if needed, a complaint to the local
authority (many councils/municipalities provide noise diaries or their own recording apps). They usually
won't accept consumer-device dBFS figures as a measurement; a calibrated mic and timestamped clips help,
but the authority normally makes its own assessment.

**Welfare angle:** early-morning barking often means a dog left outside alone. A factual pattern
("barking 06:40–07:10 on 5 of 7 days, mostly weekends") is a constructive opener with the owner – or,
in genuine distress cases, with an animal-welfare service.

---

## 15. Cost estimates (approximate, 2026 prices; check current)

| Item | MVP | Production |
|---|---|---|
| Raspberry Pi | €0 (existing) | Pi 5 4 GB ≈ €65–80 |
| PSU, case, cooler | €0 | €25–35 |
| Storage | €0 (existing SD) / €12 high-endurance SD | 256 GB NVMe + HAT ≈ €45–60 |
| Microphone | USB mic €10–25 | ReSpeaker array €65–80 **or** UMM-6/UMIK-1 €80–120 |
| Outdoor enclosure, vent, windscreen, gland | – | €15–30 |
| Bedroom speaker | €0 (existing) | €0–100 (e.g. used Sonos/Nest, or Pi Zero 2 W + speaker bonnet ≈ €35) |
| UPS | – | €30–60 |
| Software | €0 | €0 |
| **Total** | **≈ €10–40** | **≈ €200–400** |

Running cost: a Pi 5 at ~3–4 W ≈ 30 kWh/year (≈ €10/year at €0.30/kWh).

---

## 16. Step-by-step implementation plan

**Phase 0 – Prepare (30 min)**
1. Flash Raspberry Pi OS Lite 64-bit, enable SSH, set hostname `barkwatch`, set timezone
   (`sudo timedatectl set-timezone Europe/…`) – reports and quiet hours use local time.
2. Mosquitto: HA *Settings → Add-ons → Mosquitto broker* (or `docker compose up mosquitto`). Create users
   `barkwatch` and `homeassistant`. Enable the MQTT integration in HA.

**Phase 1 – Audio (1 h)**
3. Plug in the mic, `arecord -l`, then `arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 10 t.wav` and
   listen. Disable AGC in `alsamixer` and set capture gain so a bark peaks around −12 dBFS.
4. Place the mic; run `python -m barkwatch calibrate --seconds 300` at a quiet time and at a busy time.

**Phase 2 – Service (1–2 h)**
5. `git clone` this repo, `./scripts/install_pi.sh` (venv, deps, model, systemd units).
6. Edit `/opt/barkwatch/config.yaml`: MQTT, device, `min_level_dbfs` ≈ quiet p10 + 10 dB.
7. Record a real morning, run `barkwatch analyze morning.wav -v`, tune `threshold`/`min_level_dbfs`.
8. `sudo systemctl enable --now barkwatch barkwatch-report-{daily,weekly,monthly}.timer barkwatch-purge.timer`,
   `journalctl -fu barkwatch`.

**Phase 3 – Home Assistant (1–2 h)**
9. Confirm the **BarkWatch garden** device appears (Settings → Devices → MQTT).
10. Copy `homeassistant/packages/barkwatch.yaml` to `/config/packages/`, replace
    `media_player.bedroom_speaker` and `notify.mobile_app_my_phone`, restart HA.
11. **Set the helpers once** (they start at their minimum values): quiet start 22:00, weekday end 07:00,
    weekend end 09:00, alert thresholds (e.g. 5/hour, 30 min/day), cooldown 60 min, volumes 0.15/0.40.
12. Paste the dashboard YAML into a new dashboard with URL `dashboard-barkwatch`.
13. Generate noise files (`scripts/generate_noise.sh`) and copy them to `/media/sleep/`; test with the
    "Test noise" button and measure the level at the pillow.
14. Optional: NFS/SMB-export the clips dir and add it as HA network storage (Media).

**Phase 4 – Tune (2–4 weeks, a few minutes a day)**
15. Review episodes via the notification action or `barkwatch list` / `barkwatch label ID not_bark`.
16. Adjust thresholds until precision is ≥95 %. Read the first weekly report; enable pre-emptive noise
    once `typical_first_bark` looks right.

**Phase 5 – Harden**
17. Move `data/` to SSD, set up backups, UPS, Prometheus/Uptime-Kuma or the HA offline automation,
    MQTT TLS if the broker is on another host, `unattended-upgrades`.

---

## 17. Minimum viable version (one weekend)

| Saturday | Sunday |
|---|---|
| Pi + any USB mic on the windowsill; install; `calibrate`; record and `analyze` one hour | HA package with only: threshold alert, daily summary, sleep-protection start/stop |
| `detector.type: yamnet`, `clips.format: wav` (no extra deps), MQTT to HA | Dashboard *Overview* view; set helpers; generate brown noise; test the speaker |
| Let it run overnight | Check the first night's episodes; mark false positives; adjust `threshold` |

Everything needed is already in this repository; the MVP is simply the default config with fewer
automations enabled. Skip DOA, SSD, UPS, outdoor enclosure, network clip share, and monthly reports.

---

## 18. Production-quality version

* Dedicated Pi 5 + SSD + UPS; mic in a weatherproof enclosure or array at the window.
* YAMNet + temporal confirmation + level gate + DOA sectors (array) or calibrated SPL (measurement mic).
* Hardened systemd unit (read-only filesystem except `data/`, memory cap), MQTT auth + TLS, offline
  alert via LWT, backups, log rotation (journald), unattended security updates.
* Feedback loop in HA (actionable notifications, "Not a bark" button) feeding labels.
* Full retention policy, speech suppression, encrypted backups, documented privacy notice.
* All 12 automations, both dashboard views, weekly/monthly reports published to HA and archived as
  Markdown/CSV/JSON.
* Regression suite: `tests/` plus a folder of your own labelled clips replayed through
  `barkwatch analyze` after every config or model change.

---

## 19. Roadmap / future enhancements

1. **Custom classifier head** – export YAMNet embeddings for labelled episodes, train logistic
   regression / small MLP (scikit-learn), ship as a `.npz` applied after YAMNet. Expected to remove most
   site-specific false positives. Add `barkwatch export-dataset` to dump labelled windows.
2. **Second-stage model** – EfficientAT or PANNs (ONNX Runtime) only on candidate windows.
3. **Individual dog identification** – cluster embeddings per episode (HDBSCAN) to separate "neighbour's
   terrier" from "dog two gardens away"; per-dog statistics.
4. **Multiple nodes** – ESP32-S3 + INMP441 listening posts streaming to the Pi (or running a
   TFLite-Micro bark model on-device), fused by time-difference-of-arrival for localisation.
5. **Web UI** – spectrogram review page with one-click labelling (BirdNET-Pi-style), served locally.
6. **Calibrated SPL** – per-mic calibration offset → dB(A) with A-weighting filter, Leq/Lmax per episode
   and per quiet-hour period, matching how local authorities describe noise.
7. **Context correlation** – join episodes with weather (rain/wind suppress or cause false positives),
   neighbour's car/door events you can observe legitimately, calendar (holidays), sunrise time
   (dawn-correlated barking), and your sleep data (wearable/bed sensor) to show *actual* wake-ups.
8. **Adaptive sleep protection** – learn per-weekday first-bark distributions and start masking noise at
   the 10th percentile rather than the median; adjust volume to the bark level measured *in the bedroom*
   with a second mic.
9. **PDF/HTML reports** with charts (matplotlib) and an export bundle (report + selected kept clips +
   SHA-256 manifest) for a noise diary.
10. **HA custom integration / add-on** packaging (HA OS add-on with audio access) so no separate Pi is
    needed when HA itself runs on a Pi near the garden.
