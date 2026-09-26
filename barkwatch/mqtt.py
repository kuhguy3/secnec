"""MQTT publishing with Home Assistant discovery, plus a small command channel."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable

from . import __version__
from .config import MqttConfig

log = logging.getLogger(__name__)


class MqttLink:
    def __init__(self, cfg: MqttConfig, node_id: str, on_command: Callable[[dict], None] | None = None):
        import paho.mqtt.client as mqtt

        self.cfg = cfg
        self.node = node_id
        self.base = f"{cfg.base_topic}/{node_id}"
        self.on_command = on_command
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"barkwatch-{node_id}")
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)
        if cfg.tls:
            self.client.tls_set()
        self.client.will_set(f"{self.base}/status", "offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def start(self) -> None:
        self.client.connect_async(self.cfg.host, self.cfg.port, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.publish(f"{self.base}/status", "offline", qos=1, retain=True).wait_for_publish(2)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect failed: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s", self.cfg.host, self.cfg.port)
        client.subscribe(f"{self.base}/cmd", qos=1)
        client.subscribe(f"{self.cfg.discovery_prefix}/status")  # re-announce when HA restarts
        self.publish_discovery()
        client.publish(f"{self.base}/status", "online", qos=1, retain=True)

    def _on_message(self, client, userdata, msg):
        if msg.topic == f"{self.cfg.discovery_prefix}/status":
            if msg.payload == b"online":
                self.publish_discovery()
            return
        try:
            cmd = json.loads(msg.payload)
        except ValueError:
            log.warning("ignoring non-JSON command: %r", msg.payload[:100])
            return
        if self.on_command and isinstance(cmd, dict):
            self.on_command(cmd)

    # -- state --------------------------------------------------------------
    def publish(self, suffix: str, payload, retain: bool = False) -> None:
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        self.client.publish(f"{self.base}/{suffix}", payload, qos=0, retain=retain)

    def active(self, on: bool) -> None:
        self.publish("active", "ON" if on else "OFF", retain=True)

    def live(self, data: dict) -> None:
        self.publish("live", data, retain=True)

    def stats(self, data: dict) -> None:
        self.publish("stats", data, retain=True)

    def event(self, data: dict) -> None:
        self.publish("event", data)

    def report(self, period: str, data: dict) -> None:
        self.publish(f"report/{period}", data, retain=True)

    # -- discovery ----------------------------------------------------------
    def publish_discovery(self) -> None:
        dev = {"identifiers": [f"barkwatch_{self.node}"], "name": f"BarkWatch {self.node}",
               "manufacturer": "BarkWatch", "model": "Bark monitor", "sw_version": __version__}
        avail = {"availability_topic": f"{self.base}/status"}
        stats_t, live_t = f"{self.base}/stats", f"{self.base}/live"

        def sensor(key: str, name: str, topic: str, tpl: str, **extra) -> tuple[str, str, dict]:
            return "sensor", key, {"name": name, "state_topic": topic, "value_template": tpl, **extra}

        entities = [
            ("binary_sensor", "active", {
                "name": "Barking", "state_topic": f"{self.base}/active", "device_class": "sound",
                "icon": "mdi:dog"}),
            sensor("level", "Level", live_t, "{{ value_json.level }}", unit_of_measurement="dBFS",
                   state_class="measurement", icon="mdi:waveform", suggested_display_precision=1),
            sensor("noise_floor", "Noise floor", live_t, "{{ value_json.floor }}",
                   unit_of_measurement="dBFS", state_class="measurement", icon="mdi:volume-low",
                   suggested_display_precision=1),
            sensor("bark_level", "Bark level", live_t, "{{ value_json.bark_level }}",
                   unit_of_measurement="dBFS", icon="mdi:volume-high", suggested_display_precision=1),
            sensor("bark_score", "Bark score", live_t, "{{ value_json.score }}",
                   state_class="measurement", icon="mdi:percent", suggested_display_precision=2),
            sensor("episodes_today", "Episodes today", stats_t, "{{ value_json.episodes_today }}",
                   state_class="total_increasing", icon="mdi:counter"),
            sensor("bark_minutes_today", "Bark minutes today", stats_t,
                   "{{ value_json.bark_minutes_today }}", unit_of_measurement="min",
                   state_class="total_increasing", icon="mdi:timer-outline"),
            sensor("episodes_last_hour", "Episodes last hour", stats_t,
                   "{{ value_json.episodes_last_hour }}", state_class="measurement", icon="mdi:clock-alert"),
            sensor("episodes_7d", "Episodes 7 days", stats_t, "{{ value_json.episodes_7d }}",
                   state_class="measurement", icon="mdi:calendar-week", json_attributes_topic=stats_t),
            sensor("episodes_30d", "Episodes 30 days", stats_t, "{{ value_json.episodes_30d }}",
                   state_class="measurement", icon="mdi:calendar-month"),
            sensor("first_bark_today", "First bark today", stats_t,
                   "{{ value_json.first_bark_today or None }}", device_class="timestamp"),
            sensor("last_episode", "Last episode", stats_t, "{{ value_json.last_episode or None }}",
                   device_class="timestamp"),
            sensor("last_episode_duration", "Last episode duration", stats_t,
                   "{{ value_json.last_episode_duration_s or 0 }}", unit_of_measurement="s",
                   device_class="duration"),
            sensor("weekly_report", "Weekly report", f"{self.base}/report/weekly",
                   "{{ value_json.episodes }}", icon="mdi:file-chart",
                   json_attributes_topic=f"{self.base}/report/weekly"),
            ("event", "episode", {
                "name": "Episode", "state_topic": f"{self.base}/event",
                "event_types": ["started", "ended"], "icon": "mdi:dog-side"}),
        ]
        for component, key, cfg in entities:
            payload = {**cfg, **avail, "unique_id": f"barkwatch_{self.node}_{key}", "device": dev,
                       "has_entity_name": True}
            self.client.publish(f"{self.cfg.discovery_prefix}/{component}/barkwatch_{self.node}/{key}/config",
                                json.dumps(payload), qos=1, retain=True)
