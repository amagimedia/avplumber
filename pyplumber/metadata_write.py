"""metadata_write.py — MetadataStoreNode: store frame metadata in an external backend.

Supported backends:
  kafka   — publishes a JSON message per frame to a Kafka topic

Node params:
  src             source edge name (VideoFrame)
  dst             optional destination edge (pass-through)
  group / name    standard avplumber node fields
  backend         dict with at least {"type": "kafka", ...}
                  Kafka-specific keys:
                    bootstrap_servers   broker list (default: "localhost:9092")
                    topic               target topic  (required)
                    key                 optional per-message key (string)
  metadata        list of metadata key names to extract from each frame,
                  OR "*" to forward all metadata keys present on the frame.

Each message written to the backend is a JSON object:
  {
    "pts":        <float seconds>,
    "pts_tb":     "<num>/<den>",        # timebase fraction
    "<key1>":     <raw value or parsed JSON>,
    "<key2>":     ...
  }

Usage example:
  MetadataStoreNode({
      "src": "v_with_meta", "dst": "v_out",
      "group": "scene", "name": "MetaToKafka",
      "backend": {"type": "kafka", "bootstrap_servers": "localhost:9092",
                  "topic": "scene-metadata"},
      "metadata": ["scene_overlay"],
  })
"""

import json
import threading
from typing import Any

from pyplumber.node import PythonNode


class _KafkaBackend:
    def __init__(self, cfg: dict):
        from kafka import KafkaProducer  # imported lazily so the module loads without kafka installed
        topic = cfg.get("topic")
        if not topic:
            raise ValueError("backend.topic is required for kafka backend")
        self._topic = topic
        self._key = cfg.get("key", None)
        if isinstance(self._key, str):
            self._key = self._key.encode()
        self._producer = KafkaProducer(
            bootstrap_servers=cfg.get("bootstrap_servers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
            acks=1,
            retries=3,
        )

    def write(self, record: dict) -> None:
        self._producer.send(self._topic, value=record, key=self._key)

    def flush(self) -> None:
        self._producer.flush()

    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


_BACKENDS = {
    "kafka": _KafkaBackend,
}


class MetadataStoreNode(PythonNode):
    """AVPlumber Python node that writes frame metadata to an external backend.

    Frames pass through unchanged (if dst is configured). For every frame
    the node extracts the requested metadata keys, packages them with the
    frame PTS, and sends a JSON record to the configured backend.
    """

    def __init__(self, args: dict):
        super().__init__(args)

        backend_cfg = args.get("backend", {})
        if isinstance(backend_cfg, str):
            backend_cfg = json.loads(backend_cfg)

        backend_type = backend_cfg.get("type", "").lower()
        if backend_type not in _BACKENDS:
            raise ValueError(
                f"Unsupported backend type: {backend_type!r}. "
                f"Supported: {list(_BACKENDS)}"
            )

        self._backend: Any = _BACKENDS[backend_type](backend_cfg)

        meta_cfg = args.get("metadata", [])
        if isinstance(meta_cfg, str):
            if meta_cfg.strip() == "*":
                self._metadata_keys = "*"
            else:
                self._metadata_keys = [k.strip() for k in meta_cfg.split(",") if k.strip()]
        else:
            self._metadata_keys = list(meta_cfg)

        self._lock = threading.Lock()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pts_seconds(self, frame) -> float:
        try:
            pts = frame.pts
            return float(pts.timestamp)
        except Exception:
            return 0.0

    def _pts_timebase(self, frame) -> str:
        try:
            tb = frame.pts.timebase
            return f"{tb.numerator}/{tb.denominator}"
        except Exception:
            return "1/1"

    def _extract_metadata(self, frame) -> dict:
        result = {}
        try:
            proxy = frame.metadata
        except Exception:
            return result

        if self._metadata_keys == "*":
            # VideoFrameMetadataProxy exposes .as_dict to get all keys
            try:
                raw_dict = proxy.as_dict
            except Exception:
                raw_dict = {}
            keys = list(raw_dict.keys())
        else:
            keys = self._metadata_keys
            raw_dict = None

        for key in keys:
            try:
                raw = raw_dict[key] if raw_dict is not None else proxy[key]
                try:
                    result[key] = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    result[key] = raw
            except (KeyError, Exception):
                pass

        return result

    def _build_record(self, frame) -> dict:
        record = {
            "pts":    self._pts_seconds(frame),
            "pts_tb": self._pts_timebase(frame),
        }
        record.update(self._extract_metadata(frame))
        return record

    # ── main loop ─────────────────────────────────────────────────────────────

    def process(self):
        frame = self._src.tryGet(1000)
        if frame is None:
            return

        record = self._build_record(frame)
        with self._lock:
            self._backend.write(record)

        dst = getattr(self, "_dst", None)
        if isinstance(dst, dict):
            for edge in dst.values():
                edge.enqueue(frame)
        elif dst is not None:
            dst.enqueue(frame)

    def doStop(self):
        with self._lock:
            self._backend.close()
