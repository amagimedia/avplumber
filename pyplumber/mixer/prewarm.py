"""Warm the permanent transition path and restore program/preview routing."""

import json
import time


class TransitionPrewarm:
    def __init__(self, avp, name: str):
        self.avp = avp
        self.name = name

    def _set_outputs(self, outputs: int) -> None:
        lines = []
        for slot in ("a", "b"):
            node = f"{self.name}_otm_scene_{slot}"
            lines.extend((
                f"node.object.set {node} outputs {outputs}",
            ))
        self.avp.executeCommandsFromString("\n".join(lines))

    def begin(self, scene: str) -> None:
        self.avp.executeCommandsFromString(
            "mixer.preview " + json.dumps({"mixer": self.name, "scene": scene})
        )
        self._set_outputs(2)

    def finish(self) -> None:
        self._set_outputs(1)

    def start_output(self) -> None:
        # Frame timestamps use the host's monotonic clock. Keep older pre-roll
        # behind the gate even if a downstream resampler is still draining it.
        self.avp.executeCommandsFromString(
            f"node.object.set {self.name}_otm_final enable_from "
            + str((time.monotonic_ns() + 999999) // 1000000)
        )
