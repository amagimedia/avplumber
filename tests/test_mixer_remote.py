"""The mixer control client against a fake avplumber process."""

import json
import socket
import threading

import pytest

from pyplumber.mixer.remote import ControlError, RemoteAvplumber, parse_endpoint, parse_reply


def test_parse_endpoint_and_reply():
    assert parse_endpoint("127.0.0.1:22422") == ("127.0.0.1", 22422)
    with pytest.raises(ValueError):
        parse_endpoint("22422")
    assert parse_reply(b"200 OK\n") == (200, "OK", None)
    assert parse_reply(b'201 OK\n{"pgm_scene":"cam"}\n\n') == (201, "OK", '{"pgm_scene":"cam"}\n')
    assert parse_reply(b"500 ERROR: unknown node type: clip_cache\n")[0] == 500


class _Node:
    def __init__(self, parameters):
        self.parameters = parameters

    def add_to_avplumber(self, avp):
        self.avp = avp


def _serve(bound: socket.socket, seen: list[str], ready: threading.Event):
    bound.listen(1)
    ready.set()
    conn, _addr = bound.accept()
    with conn:
        conn.sendall(b"100 VTR READY\n")
        reader = conn.makefile("rb", buffering=0)
        while True:
            line = reader.readline()
            if not line:
                return
            text = line.decode().strip()
            seen.append(text)
            if text.startswith("node.add "):
                conn.sendall(b"200 OK\n")
            elif text.startswith("queue.plan_capacity"):
                conn.sendall(b"200 OK\n")
            elif text == "group.status g":
                body = '{"state":"running","outcomes":[]}\n'
                conn.sendall(f"201 OK\n{body}\n".encode())
            elif text == "hello":
                conn.sendall(b"200 OK\n")
            elif text.startswith("boom"):
                conn.sendall(b"500 ERROR: nope\n")
            else:
                conn.sendall(b"400 Unknown command: " + text.split()[0].encode() + b"\n")


def _listen():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    seen: list[str] = []
    ready = threading.Event()
    thread = threading.Thread(target=_serve, args=(sock, seen, ready), daemon=True)
    thread.start()
    ready.wait()
    return port, seen


def test_client_sends_node_add_and_raises_on_error():
    port, seen = _listen()
    avp = RemoteAvplumber.connect("127.0.0.1", port)
    avp.addNode(_Node({"type": "null_sink", "name": "n", "group": "g", "pair": (1, 2)}))
    avp.edges.planCapacity("*", 3)
    sent = json.loads(seen[0].removeprefix("node.add "))
    assert sent == {"type": "null_sink", "name": "n", "group": "g", "pair": [1, 2]}
    assert seen[1] == "queue.plan_capacity * 3"
    assert avp.group("g").is_running()
    with pytest.raises(ControlError, match="nope"):
        avp.executeCommandsFromString("boom now")
    avp.shutdown()


def test_public_port_handles_local_commands_and_forwards_the_rest():
    port, seen = _listen()
    avp = RemoteAvplumber.connect("127.0.0.1", port)
    avp.registerControlCommand("mixer.settings", lambda _arg: '{"fps":30}\n')
    public = socket.socket()
    public.bind(("127.0.0.1", 0))
    public_port = public.getsockname()[1]
    public.close()
    avp.enableControlServer(public_port, "127.0.0.1")
    avp.setReady()
    try:
        client = socket.create_connection(("127.0.0.1", public_port))
        reader = client.makefile("rb", buffering=0)
        assert reader.readline() == b"100 VTR READY\n"
        client.sendall(b"mixer.settings\n")
        assert reader.readline() == b"201 OK\n"
        assert reader.readline() == b'{"fps":30}\n'
        assert reader.readline() == b"\n"
        client.sendall(b"hello\n")
        assert reader.readline() == b"200 OK\n"
        assert "hello" in seen
        client.sendall(b"bye\n")
        assert reader.readline() == b"BYE\n"
    finally:
        avp.shutdown()
