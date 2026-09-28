"""Run under an external timeout: shutdown must let Python workers acquire the GIL."""

import threading
import time

from pyplumber import AVPlumber
from pyplumber.node import NodeBase, PythonNode

started = threading.Event()
stopped = threading.Event()


class Worker(PythonNode):
    def process(self):
        started.set()
        time.sleep(0.01)

    def doStop(self):
        stopped.set()


avp = AVPlumber()
worker = Worker({"name": "worker", "group": "test", "dst": "unused",
                 "data_type": "VideoFrame", "auto_restart": "panic"})
avp.addNode(worker)
avp.group("test").startNodes()
assert started.wait(5), "Python worker did not start"
avp.shutdown()
assert stopped.is_set(), "Python doStop did not finish before shutdown returned"
print("Python worker shutdown passed", flush=True)

# Split makes a final process() call after stop. An empty input must not wait
# again after the stop notification has already been consumed.
avp = AVPlumber()
avp.getEdge("empty", "VideoFrame")
avp.addNode(NodeBase({"type": "split", "name": "idle_split", "group": "test",
                      "src": "empty", "dst": ["unused"], "data_type": "VideoFrame"}))
avp.group("test").startNodes()
time.sleep(0.1)
avp.shutdown()
print("Idle Split shutdown passed", flush=True)

# A worker may read again while flushing its tail. Both reads must return once
# stop has been requested; the first must not consume the queue's stop state.
class Reader(PythonNode):
    def process(self):
        started.set()
        self._src.get()
        self._src.get()
        stopped.set()

started.clear()
stopped.clear()
avp = AVPlumber()
reader = Reader({"name": "reader", "group": "test", "src": "empty",
                 "data_type": "VideoFrame"})
avp.addNode(reader)
avp.group("test").startNodes()
assert started.wait(5), "Reader did not start"
avp.shutdown()
assert stopped.is_set(), "Repeated reads did not stop"
print("Repeated queue reads shutdown passed", flush=True)

# A panic shuts the graph down under the manager lock while Python keeps looking nodes up (the
# mixer's aux poll). Lookups must not hold the GIL that stopping the Python worker needs.
panic = threading.Event()


class Trigger(PythonNode):
    def process(self):
        if panic.is_set():
            raise RuntimeError("panic on request")
        time.sleep(0.01)


def poll_until_shut_down():
    try:
        while True:
            avp.node("worker")
    except Exception:
        pass  # the node index is cleared once the panic's shutdown finished


started.clear()
stopped.clear()
avp = AVPlumber()
avp.addNode(Worker({"name": "worker", "group": "test", "dst": "unused", "data_type": "VideoFrame"}))
avp.addNode(Trigger({"name": "trigger", "group": "test", "dst": "unused2", "data_type": "VideoFrame",
                     "auto_restart": "panic"}))
avp.group("test").startNodes()
assert started.wait(5), "Python worker did not start"
poller = threading.Thread(target=poll_until_shut_down, daemon=True)
poller.start()
panic.set()
poller.join(10)
assert not poller.is_alive(), "node() lookups deadlocked the panic shutdown"
assert stopped.is_set(), "Python doStop did not finish during the panic shutdown"
avp.shutdown()
print("Panic shutdown with concurrent node lookups passed", flush=True)
