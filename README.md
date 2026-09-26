# secnec – BarkWatch

Local-first dog-bark monitoring for Home Assistant on a Raspberry Pi. It detects barking (YAMNet or a
non-ML detector), logs episodes with duration, frequency and level, keeps short verification clips
(and drops any clip that contains human speech), reports daily, weekly and monthly patterns, sends alerts
only past thresholds you set, and can play masking noise in your bedroom.

It only listens. Nothing is ever played toward the dog.

**Full design** (architecture, hardware, microphones, ML vs non-ML, privacy, costs, plan, roadmap):
[`docs/DESIGN.md`](docs/DESIGN.md)

## Quick start (Raspberry Pi OS 64-bit)

```bash
git clone <this repo> && cd secnec
./scripts/install_pi.sh                      # venv, deps, YAMNet model, systemd units
sudoedit /opt/barkwatch/config.yaml          # MQTT host/credentials, audio device
/opt/barkwatch/venv/bin/python -m barkwatch devices
/opt/barkwatch/venv/bin/python -m barkwatch -c /opt/barkwatch/config.yaml calibrate --seconds 120
sudo systemctl enable --now barkwatch barkwatch-report-daily.timer \
     barkwatch-report-weekly.timer barkwatch-report-monthly.timer barkwatch-purge.timer
journalctl -fu barkwatch
```

Or with Docker: `cp config.example.yaml config.yaml && docker compose up -d` (see `docker-compose.yml`).

Home Assistant:
1. The **BarkWatch garden** device appears automatically through MQTT discovery.
2. Copy `homeassistant/packages/barkwatch.yaml` to `/config/packages/` and replace the placeholder speaker
   and notify service.
3. Set the helper values once (quiet hours, thresholds, volumes).
4. Paste `homeassistant/dashboards/barkwatch.yaml` into a new dashboard.
5. Run `scripts/generate_noise.sh` and copy the files to `/media/sleep/`.

## CLI

```
python -m barkwatch run                          # live detection (systemd runs this)
python -m barkwatch analyze clip.wav -v          # tune thresholds on a recording
python -m barkwatch report --period weekly --print [--date 2026-09-20] [--publish]
python -m barkwatch list                         # recent episodes
python -m barkwatch label 42 not_bark            # feedback: excluded from stats, clip purged early
python -m barkwatch label 42 bark --keep true    # keep the clip beyond retention
python -m barkwatch purge                        # apply retention rules
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[yamnet,dev]"
./scripts/download_model.sh
YAMNET_MODEL=models/yamnet.tflite YAMNET_CLASS_MAP=models/yamnet_class_map.csv pytest
```
