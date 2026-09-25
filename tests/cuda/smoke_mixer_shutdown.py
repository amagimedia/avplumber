"""Run a small prepared mixer show through complete Python process teardown.

Use --config <show.json> on an NVIDIA host. Include HDR uploads and AUX to
exercise their device references. Outputs go to temporary local UDP ports;
browser sources are excluded so this test cannot take over live windows.
The parent checks process exit, since a global CUDA device can crash *after*
AVPlumber.shutdown() has returned successfully.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "demos/mixer")]


def child(config):
    from mixer import GraphOptions, build_application

    app = build_application(GraphOptions(config=str(config), janus_output=True, remote_control_port=0))
    try:
        app.start()
        time.sleep(1)
    finally:
        app.stop()
    print("shutdown returned", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.config)
        return
    show = json.loads(args.config.read_text())
    if any(s["kind"] == "browser" for s in show["sources"]):
        parser.error("Use a small dedicated show without browser sources")
    with tempfile.TemporaryDirectory() as directory, ExitStack() as sockets:
        for output in [show, *show.get("aux_buses", [])]:
            for rendition in output.get("renditions", []):
                receiver = sockets.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                receiver.bind(("127.0.0.1", 0))
                rendition.update(target="janus", port=receiver.getsockname()[1])
        config = Path(directory) / "show.json"
        config.write_text(json.dumps(show))
        for attempt in range(args.repeats):
            result = subprocess.run([sys.executable, __file__, "--child", "--config", str(config)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=args.timeout)
            if result.returncode or "shutdown returned" not in result.stdout or "cuCtxDestroy" in result.stdout:
                raise AssertionError(f"shutdown exited {result.returncode}:\n{result.stdout[-12000:]}")
            print(f"Clean mixer process exit {attempt + 1}/{args.repeats}", flush=True)


if __name__ == "__main__":
    main()
