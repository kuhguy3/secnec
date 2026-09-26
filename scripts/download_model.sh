#!/usr/bin/env bash
# Downloads the YAMNet TFLite model (Apache-2.0) and its AudioSet class map into ./models
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
curl -fL -o models/yamnet.tflite \
  https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite
curl -fL -o models/yamnet_class_map.csv \
  https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
ls -l models
