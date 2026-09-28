"""Run under an external timeout: a failed auto_restart "on" restart is retried every second
until it succeeds, and node.stop ends the retries."""

import threading
import time

from pyplumber import AVPlumber
from pyplumber.node import PythonNode

running = threading.Event()
finish = threading.Event()
restart_failures = []


class Worker(PythonNode):
    def process(self):
        running.set()
        if finish.is_set():
            raise RuntimeError("finished on request")
        time.sleep(0.01)


class Plumber(AVPlumber):
    def on_exception(self, node_name, node_type, message):
        if "no_such_type" in message:
            restart_failures.append(message)


def finish_with_broken_factory():
    """Finish the running worker; its restarts fail until the type is set back."""
    avp.executeCommandsFromString('node.param.set worker type "no_such_type"\n')
    failures = len(restart_failures)
    finish.set()
    deadline = time.monotonic() + 5
    while len(restart_failures) == failures:
        assert time.monotonic() < deadline, "failed restart was not reported"
        time.sleep(0.05)


avp = Plumber()
worker = Worker({"name": "worker", "dst": "unused", "data_type": "VideoFrame",
                 "auto_restart": "on"})
worker_type = worker.parameters["type"]
avp.addNode(worker, True, True)
assert running.wait(5), "worker did not start"

finish_with_broken_factory()
time.sleep(2.5)
assert len(restart_failures) >= 2, "failed restart was not retried"
running.clear()
finish.clear()
avp.executeCommandsFromString(f'node.param.set worker type "{worker_type}"\n')
assert running.wait(3), "restart did not recover once the node could be created"
print("Failed restart retried until it succeeded", flush=True)

finish_with_broken_factory()
avp.executeCommandsFromString("node.stop worker\n")
failures = len(restart_failures)
time.sleep(2.5)
assert len(restart_failures) == failures, "node.stop did not end the restart retries"
avp.shutdown()
print("node.stop ended the restart retries", flush=True)
