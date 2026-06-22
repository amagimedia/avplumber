"""Native exception reporting for the auto mixer."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

from pyplumber import AVPlumber


class NativeExceptionRegistry:
    """Thread-safe bounded history of native node/group failures."""

    def __init__(self, max_events: int = 50) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._counts_by_node: Counter[str] = Counter()
        self._counts_by_type: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._total = 0

    def record(self, node_name: str, node_type: str, message: str) -> None:
        if (
            str(node_type) == "NodeGroup"
            and str(message) == "Error while changing state: Node factory returned nullptr"
        ):
            return
        event = {
            "time_unix_s": round(time.time(), 3),
            "node_name": str(node_name),
            "node_type": str(node_type),
            "message": str(message),
        }
        with self._lock:
            self._total += 1
            self._events.append(event)
            self._counts_by_node[event["node_name"]] += 1
            self._counts_by_type[event["node_type"]] += 1
        print(
            "[native_exception] "
            f"{event['node_type']} {event['node_name']}: {event['message']}",
            flush=True,
        )

    def summary(self, recent: int = 5) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
            return {
                "total": self._total,
                "recent": events[-recent:],
                "by_node": self._counts_by_node.most_common(),
                "by_type": self._counts_by_type.most_common(),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self._total,
                "events": list(self._events),
                "by_node": self._counts_by_node.most_common(),
                "by_type": self._counts_by_type.most_common(),
            }


class AutoMixerAVPlumber(AVPlumber):
    """AVPlumber with native exception collection enabled."""

    def __init__(self, native_exceptions: NativeExceptionRegistry) -> None:
        self.native_exceptions = native_exceptions
        super().__init__()

    def on_exception(self, node_name: str, node_type: str, message: str) -> None:
        self.native_exceptions.record(node_name, node_type, message)
