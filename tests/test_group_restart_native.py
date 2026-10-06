"""Bounded native regressions for restart requests during group creation."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize("case", ["overlap", "stop", "shutdown"])
def test_group_restart(case):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native avplumber extension")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {case}" in result.stdout


def run_case(case):
    from pyplumber import AVPlumber
    from pyplumber.node import PythonNode

    blocked = threading.Event()
    release = threading.Event()
    fail = threading.Event()
    errors = []

    def wait_for(predicate, message):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline, message
            time.sleep(0.01)

    class Probe(PythonNode):
        created = 0
        processed_generation = 0

        def python_node_created(self, wrapper):
            super().python_node_created(wrapper)
            self.created += 1
            if self.created == 2:
                blocked.set()
                assert release.wait(5), "creation barrier was not released"

        def process(self):
            self.processed_generation = self.created
            if fail.is_set():
                fail.clear()
                raise RuntimeError("fresh worker failure")
            time.sleep(0.01)

    avp = AVPlumber()
    avp.on_exception = lambda *error: errors.append(error)
    worker = Probe({"name": "probe", "group": "test", "dst": "unused",
                    "data_type": "VideoFrame", "auto_restart": "group"})
    avp.addNode(worker)
    group = avp.group("test")
    group.startNodes()
    wait_for(lambda: worker.processed_generation == 1, "first start failed")
    group.restartNodes()
    assert blocked.wait(5), "restart never reached creation barrier"
    # None of these may invalidate the ongoing startup or create a restart
    # backlog; the completed transition must service one follow-up request.
    threads = [threading.Thread(target=group.restartNodes) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive(), "restart request blocked on the group lock"

    shutdown_thread = None
    if case == "stop":
        group.stopNodes()
    elif case == "shutdown":
        shutdown_thread = threading.Thread(target=avp.shutdown)
        shutdown_thread.start()
        wait_for(lambda: not avp.manager.shouldWork, "shutdown was not requested")
    release.set()

    if case == "overlap":
        wait_for(lambda: worker.processed_generation >= 3, "pending restart was lost")
        time.sleep(0.2)
        assert worker.created == 3, f"duplicate restarts: {worker.created} creations"
        assert not errors, errors
        # Coalescing must not swallow a genuinely new failure after recovery.
        fail.set()
        wait_for(lambda: worker.processed_generation >= 4, "fresh failure was lost")
        assert worker.created == 4
        assert all(error[0] == "probe" and "fresh worker failure" in error[-1]
                   for error in errors), errors
    elif case == "stop":
        # Wait through the existing retry delay after an explicit cancellation.
        time.sleep(1.3)
        assert worker.created == 2, "pending restart overrode explicit stop"
        assert not avp.node("probe").isWorking
        group.startNodes()
        wait_for(lambda: worker.processed_generation >= 3, "explicit start after stop failed")
    if shutdown_thread:
        shutdown_thread.join(5)
        assert not shutdown_thread.is_alive(), "shutdown blocked behind restart requests"
        assert worker.created == 2, "pending restart overrode shutdown"
    else:
        avp.shutdown()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    run_case(sys.argv[1])
