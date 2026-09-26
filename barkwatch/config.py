"""Configuration model and YAML loader."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    device: str | int | None = None      # sounddevice name/index; None = system default
    capture_rate: int = 16000            # rate requested from the device; must be a multiple of 16000
    channels: int = 1                    # channels opened on the device
    channel_index: int = 0               # which channel is analysed (0 = processed/beamformed on ReSpeaker)
    hop_seconds: float = 0.48            # analysis hop; YAMNet window is fixed at 0.975 s
    gain_db: float = 0.0                 # digital gain applied before analysis


@dataclass
class DetectorConfig:
    type: str = "yamnet"                 # yamnet | heuristic
    threshold: float = 0.35              # bark score needed for a "hit"
    min_level_dbfs: float = -65.0        # ignore hits quieter than this (distant dogs)
    energy_gate_db: float = 4.0          # skip inference when level < noise floor + gate (saves CPU)
    # YAMNet
    model_path: str = "models/yamnet.tflite"
    class_map_path: str = "models/yamnet_class_map.csv"
    bark_classes: list[str] = field(default_factory=lambda: [
        "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)"])
    speech_classes: list[str] = field(default_factory=lambda: [
        "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Shout", "Yell", "Whispering", "Singing"])
    num_threads: int = 2
    # Heuristic (non-ML) detector
    band_low_hz: float = 350.0
    band_high_hz: float = 4000.0
    min_band_ratio: float = 0.55         # share of energy inside the bark band
    onset_db: float = 12.0               # burst must rise this far above the noise floor
    min_burst_ms: float = 60.0
    max_burst_ms: float = 600.0
    max_flatness: float = 0.35           # spectral flatness; barks are harmonic, wind/traffic are flat


@dataclass
class EventConfig:
    episode_gap_s: float = 30.0          # silence that closes an episode
    min_hits: int = 2                    # hits needed to confirm an episode (rejects one-off false positives)
    confirm_window_s: float = 10.0       # ... within this many seconds of the first hit
    max_episode_s: float = 1800.0        # split very long episodes


@dataclass
class ClipConfig:
    enabled: bool = True
    directory: str = "data/clips"
    format: str = "flac"                 # flac (needs soundfile) | wav
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0
    max_clip_s: float = 60.0


@dataclass
class StorageConfig:
    database: str = "data/barkwatch.db"
    clip_retention_days: int = 30
    false_positive_clip_retention_days: int = 2
    event_retention_days: int = 730
    minute_stats_retention_days: int = 90
    max_clip_storage_mb: int = 4096


@dataclass
class MqttConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    tls: bool = False
    base_topic: str = "barkwatch"
    discovery_prefix: str = "homeassistant"
    live_interval_s: float = 5.0
    stats_interval_s: float = 60.0


@dataclass
class QuietHours:
    start: str = "22:00"
    end: str = "07:00"


@dataclass
class ReportConfig:
    directory: str = "data/reports"
    quiet_hours: QuietHours = field(default_factory=QuietHours)
    weekend_quiet_hours: QuietHours = field(default_factory=lambda: QuietHours("23:00", "09:00"))


@dataclass
class PrivacyConfig:
    drop_clips_with_speech: bool = True  # never keep a clip in which human speech was detected
    speech_threshold: float = 0.4


@dataclass
class DoaConfig:
    enabled: bool = False                # ReSpeaker USB Mic Array v2.0 only
    accept_sectors: list[list[int]] = field(default_factory=list)  # e.g. [[200, 290]]; empty = accept all


@dataclass
class Config:
    node_id: str = "garden"
    audio: AudioConfig = field(default_factory=AudioConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    events: EventConfig = field(default_factory=EventConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    reports: ReportConfig = field(default_factory=ReportConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    doa: DoaConfig = field(default_factory=DoaConfig)


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'config'}: expected a mapping")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{path or 'config'}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value or {}, f"{path}.{name}".lstrip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _build(Config, data, "")
