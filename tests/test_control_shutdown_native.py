"""shutdown() with control clients connected returns, before and after setReady().

Before setReady() the clients wait at the readiness gate; shutdown() used to join them there
forever, so a mixer whose startup failed never exited while a web UI kept its connection open."""

import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("case", ["before_ready", "after_ready"])
def test_control_shutdown(case, run_script):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native avplumber extension")
    try:
        result = run_script(Path(__file__).resolve(), case, timeout=15)
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"shutdown {case} hung with control clients connected: {error.stdout!r}{error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {case}" in result.stdout


def _run(case):
    from pyplumber import AVPlumber

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    avp = AVPlumber()
    avp.enableControlServer(port, "127.0.0.1")
    clients = [socket.create_connection(("127.0.0.1", port), timeout=5) for _ in range(3)]
    for client in clients:
        client.sendall(b"hello\n")  # queued behind the gate until setReady()
    greeting = b"100 VTR Ready\n201 OK\nHELLO\n\n"
    if case == "after_ready":
        avp.setReady()
        for client in clients:
            received = b""
            while len(received) < len(greeting):
                received += client.recv(4096)
            assert received == greeting, received
    else:
        time.sleep(0.5)  # let every client thread reach the gate
    avp.shutdown()
    for client in clients:
        # Closed without a greeting: a client must not take a stopping instance for a ready one.
        try:
            assert client.recv(4096) == b"", case
        except ConnectionResetError:
            pass
        client.close()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    _run(sys.argv[1])
