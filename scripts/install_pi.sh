#!/usr/bin/env bash
# Installs BarkWatch on Raspberry Pi OS (Bookworm/Trixie, 64-bit) as a systemd service.
set -euo pipefail
PREFIX=/opt/barkwatch
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 libsndfile1 alsa-utils sox libsox-fmt-mp3
sudo useradd --system --home "$PREFIX" --groups audio,plugdev barkwatch 2>/dev/null || true
sudo mkdir -p "$PREFIX"
sudo rsync -a --exclude .git --exclude data "$(dirname "$0")/../" "$PREFIX/"
sudo python3 -m venv "$PREFIX/venv"
sudo "$PREFIX/venv/bin/pip" install --upgrade pip
sudo "$PREFIX/venv/bin/pip" install "$PREFIX[yamnet,flac]"
sudo "$PREFIX/scripts/download_model.sh"
[ -f "$PREFIX/config.yaml" ] || sudo cp "$PREFIX/config.example.yaml" "$PREFIX/config.yaml"
sudo mkdir -p "$PREFIX/data" && sudo chown -R barkwatch:barkwatch "$PREFIX/data"
sudo chmod 640 "$PREFIX/config.yaml" && sudo chown root:barkwatch "$PREFIX/config.yaml"
sudo cp "$PREFIX"/deploy/systemd/*.service "$PREFIX"/deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
echo "Edit $PREFIX/config.yaml (MQTT credentials, audio device), then:"
echo "  sudo systemctl enable --now barkwatch.service barkwatch-report-daily.timer barkwatch-report-weekly.timer barkwatch-report-monthly.timer barkwatch-purge.timer"
