#!/usr/bin/env bash
# Generates 1-hour white/pink/brown noise files for sleep protection (needs sox + libsox-fmt-mp3).
# Copy the results to Home Assistant's media folder, e.g. /media/sleep/ (media-source://media_source/local/sleep/...).
set -euo pipefail
OUT=${1:-noise}
mkdir -p "$OUT"
for colour in white pink brown; do
  # 2 s fade in/out so looping players do not click; 'gain -n -6' normalises to -6 dBFS peak
  sox -n -r 44100 -c 2 "$OUT/${colour}_noise.mp3" synth 3600 "${colour}noise" gain -n -6 fade h 2 3600 2
done
ls -lh "$OUT"
