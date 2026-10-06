"""The /janus WebSocket tunnel, between a raw TCP client as the browser and a listening socket as
Janus's WebSocket port: the tunnel relays bytes, so neither end has to speak WebSocket."""
import json
from pathlib import Path
import socket
import urllib.request

import pytest


@pytest.fixture
def janus(server, monkeypatch):
    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(2)
        monkeypatch.setattr(server, "JANUS_WS", f"ws://127.0.0.1:{listener.getsockname()[1]}")
        yield listener


def upgrade(http):
    browser = socket.create_connection(("127.0.0.1", http.server_port), timeout=2)
    browser.sendall(b"GET /janus HTTP/1.1\r\nHost: preview\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\n"
                    b"Sec-WebSocket-Key: a2V5\r\nSec-WebSocket-Version: 13\r\n"
                    b"Sec-WebSocket-Protocol: janus-protocol\r\nOrigin: http://preview\r\nCookie: not for Janus\r\n\r\n")
    return browser


def received(peer, end=b"\r\n\r\n"):
    data = b""
    while not data.endswith(end):
        chunk = peer.recv(4096)
        assert chunk, f"closed after {data!r}"
        data += chunk
    return data


def accept(janus):
    upstream = janus.accept()[0]
    upstream.settimeout(2)
    return upstream


def test_upgrade_replays_the_handshake_and_relays_both_ways(http, janus):
    with upgrade(http) as browser, accept(janus) as upstream:
        assert received(upstream).split(b"\r\n") == [
            b"GET / HTTP/1.1", b"Host: 127.0.0.1:%d" % janus.getsockname()[1],
            b"Upgrade: websocket", b"Connection: Upgrade", b"Sec-WebSocket-Key: a2V5",
            b"Sec-WebSocket-Version: 13", b"Sec-WebSocket-Protocol: janus-protocol",
            b"Origin: http://preview", b"", b""]
        reply = b"HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Protocol: janus-protocol\r\n\r\n"
        upstream.sendall(reply)
        assert received(browser) == reply, "Janus answers the handshake, not the preview server"
        browser.sendall(b"from the browser\n")
        assert received(upstream, b"\n") == b"from the browser\n"
        upstream.sendall(b"from Janus\n")
        assert received(browser, b"\n") == b"from Janus\n"
        # One tunnel is one request thread: the server keeps answering beside it.
        with urllib.request.urlopen(f"http://127.0.0.1:{http.server_port}/receiver-stats", timeout=2) as response:
            assert isinstance(json.load(response), list)
        browser.close()
        assert upstream.recv(1) == b"", "Janus must see the browser leave, to end its session"


def test_tunnel_ends_when_janus_closes(http, janus):
    with upgrade(http) as browser:
        accept(janus).close()
        assert browser.recv(1) == b""


def test_upgrade_without_janus_is_a_bad_gateway(http, janus):
    janus.close()
    with upgrade(http) as browser:
        assert browser.recv(4096).startswith(b"HTTP/1.0 502 ")


def test_plain_requests_still_reach_the_rest_api(server, http, janus, monkeypatch):
    # The preview server's own JSON endpoint stands in for the Janus REST API.
    monkeypatch.setattr(server, "JANUS_REST", f"http://127.0.0.1:{http.server_port}/receiver-stats")
    with urllib.request.urlopen(f"http://127.0.0.1:{http.server_port}/janus", timeout=2) as response:
        assert isinstance(json.load(response), list)


def test_transport_module_is_served_as_a_file(server, http, monkeypatch):
    monkeypatch.chdir(Path(server.__file__).parent)
    with urllib.request.urlopen(f"http://127.0.0.1:{http.server_port}/janus-socket.mjs", timeout=2) as response:
        assert b"export class JanusSocket" in response.read()
