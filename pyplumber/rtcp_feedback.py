"""RTCP feedback listener for Janus RTP outputs."""

from __future__ import annotations

import selectors
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable


RTCP_SR = 200
RTCP_SDES = 202
RTCP_PSFB = 206
RTCP_LEGACY_FIR = 192
RTCP_PSFB_PLI = 1
RTCP_PSFB_FIR = 4
NTP_UNIX_EPOCH_OFFSET = 2_208_988_800


@dataclass(frozen=True)
class RtcpPacket:
    fmt: int
    packet_type: int
    payload: bytes


def iter_rtcp_packets(data: bytes) -> Iterable[RtcpPacket]:
    """Yield RTCP packets from a possibly-compound RTCP datagram."""
    offset = 0
    while offset + 4 <= len(data):
        first, packet_type, length_words = struct.unpack_from("!BBH", data, offset)
        version = first >> 6
        if version != 2:
            break
        packet_len = (length_words + 1) * 4
        if packet_len < 4 or offset + packet_len > len(data):
            break
        yield RtcpPacket(
            fmt=first & 0x1F,
            packet_type=packet_type,
            payload=data[offset + 4:offset + packet_len],
        )
        offset += packet_len


def rtcp_keyframe_requests(data: bytes) -> list[str]:
    """Return keyframe request types found in an RTCP datagram."""
    requests: list[str] = []
    for packet in iter_rtcp_packets(data):
        if packet.packet_type == RTCP_PSFB and packet.fmt == RTCP_PSFB_PLI:
            requests.append("pli")
        elif packet.packet_type == RTCP_PSFB and packet.fmt == RTCP_PSFB_FIR:
            requests.append("fir")
        elif packet.packet_type == RTCP_LEGACY_FIR:
            requests.append("fir")
    return requests


def build_rtcp_pli(*, sender_ssrc: int, media_ssrc: int) -> bytes:
    return struct.pack(
        "!BBHII",
        (2 << 6) | RTCP_PSFB_PLI,
        RTCP_PSFB,
        2,
        sender_ssrc & 0xFFFFFFFF,
        media_ssrc & 0xFFFFFFFF,
    )


def build_rtcp_fir(*, sender_ssrc: int, media_ssrc: int, sequence: int = 1) -> bytes:
    return struct.pack(
        "!BBHIIII",
        (2 << 6) | RTCP_PSFB_FIR,
        RTCP_PSFB,
        4,
        sender_ssrc & 0xFFFFFFFF,
        0,
        media_ssrc & 0xFFFFFFFF,
        ((sequence & 0xFF) << 24),
    )


def _build_rtcp_sr(*, media_ssrc: int) -> bytes:
    now = time.time()
    ntp_sec = int(now) + NTP_UNIX_EPOCH_OFFSET
    ntp_frac = int((now - int(now)) * (1 << 32))
    rtp_ts = int(now * 90_000) & 0xFFFFFFFF
    return struct.pack(
        "!BBHIIIIII",
        2 << 6,
        RTCP_SR,
        6,
        media_ssrc & 0xFFFFFFFF,
        ntp_sec & 0xFFFFFFFF,
        ntp_frac & 0xFFFFFFFF,
        rtp_ts,
        0,
        0,
    )


def _build_rtcp_sdes(*, media_ssrc: int, cname: str) -> bytes:
    cname_bytes = cname.encode("ascii", "replace")[:255]
    body = struct.pack("!IBB", media_ssrc & 0xFFFFFFFF, 1, len(cname_bytes))
    body += cname_bytes + b"\x00"
    body += b"\x00" * ((4 - (len(body) % 4)) % 4)
    return struct.pack("!BBH", (2 << 6) | 1, RTCP_SDES, (len(body) + 4) // 4 - 1) + body


class RtcpFeedbackListener:
    """Listens for RTCP PLI/FIR and triggers the configured keyframe callback.

    Every listener of the process is served by one thread (_RtcpPoller): a show has one per
    Janus output, and a thread each, waking five times a second to look at its socket, was 26
    threads on the 192-source demo for a few packets a second between them."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        janus_host: str,
        janus_rtcp_port: int,
        media_ssrc: int,
        on_keyframe_request: Callable[[str], None],
        announce_interval_s: float = 1.0,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.janus_host = janus_host
        self.janus_rtcp_port = janus_rtcp_port
        self.media_ssrc = media_ssrc
        self.on_keyframe_request = on_keyframe_request
        self.announce_interval_s = announce_interval_s
        self._sock: socket.socket | None = None
        self._next_announce = 0.0
        self.local_port: int | None = None
        self.total_packets = 0
        self.keyframe_requests = 0
        self.pli_requests = 0
        self.fir_requests = 0

    def start(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.bind_port))
        sock.setblocking(False)
        self._sock = sock
        self.local_port = sock.getsockname()[1]
        _RtcpPoller.add(self)
        print(
            "[rtcp] listening for Janus PLI/FIR on "
            f"{self.bind_host}:{self.local_port}, announcing to "
            f"{self.janus_host}:{self.janus_rtcp_port}"
        )

    def stop(self) -> None:
        if self._sock is None:
            return
        _RtcpPoller.remove(self)   # returns once the poller no longer touches the socket
        self._sock.close()
        self._sock = None

    def _announce(self) -> None:
        if self._sock is None:
            return
        cname = f"avplumber-{self.media_ssrc:x}"
        packet = (
            _build_rtcp_sr(media_ssrc=self.media_ssrc) +
            _build_rtcp_sdes(media_ssrc=self.media_ssrc, cname=cname)
        )
        self._sock.sendto(packet, (self.janus_host, self.janus_rtcp_port))

    def _announce_if_due(self, now: float) -> float:
        """Announce when due; returns when the next announcement is."""
        if now >= self._next_announce:
            try:
                self._announce()
            except OSError as exc:
                print(f"[rtcp] failed to send RTCP announcement: {exc}")
            self._next_announce = now + self.announce_interval_s
        return self._next_announce

    def _receive(self) -> None:
        """Handle every datagram waiting on the socket."""
        while True:
            try:
                data, addr = self._sock.recvfrom(2048)
            except (BlockingIOError, InterruptedError):
                return
            self.total_packets += 1
            for request in rtcp_keyframe_requests(data):
                self.keyframe_requests += 1
                if request == "pli":
                    self.pli_requests += 1
                elif request == "fir":
                    self.fir_requests += 1
                print(f"[rtcp] received {request.upper()} from {addr[0]}:{addr[1]}")
                self.on_keyframe_request(request)


class _RtcpPoller:
    """The one thread that waits on every listener's socket and sends their announcements. It
    starts with the first listener and ends with the last. Listeners join and leave through a
    queue the thread applies itself, so only it ever touches the selector."""

    _lock = threading.Lock()
    _instance: "_RtcpPoller | None" = None

    def __init__(self) -> None:
        self._selector = selectors.DefaultSelector()
        self._wake_recv, self._wake_send = socket.socketpair()
        self._wake_recv.setblocking(False)
        self._selector.register(self._wake_recv, selectors.EVENT_READ, None)
        self._pending: list[tuple[RtcpFeedbackListener, bool, threading.Event]] = []
        self._listeners: set[RtcpFeedbackListener] = set()
        self._thread = threading.Thread(target=self._run, name="janus-rtcp-feedback", daemon=True)
        self._thread.start()

    @classmethod
    def add(cls, listener: RtcpFeedbackListener) -> None:
        cls._change(listener, True)

    @classmethod
    def remove(cls, listener: RtcpFeedbackListener) -> None:
        cls._change(listener, False)

    @classmethod
    def _change(cls, listener: RtcpFeedbackListener, add: bool) -> None:
        done = threading.Event()
        with cls._lock:
            if cls._instance is None:
                if not add:
                    return
                cls._instance = cls()
            poller = cls._instance
            poller._pending.append((listener, add, done))
        poller._wake_send.send(b"\0")
        done.wait()

    def _apply_pending(self) -> bool:
        """Apply joins and leaves; False once the last listener has left and the thread may end."""
        with _RtcpPoller._lock:
            pending, self._pending = self._pending, []
            for listener, add, done in pending:
                if add:
                    self._selector.register(listener._sock, selectors.EVENT_READ, listener)
                    self._listeners.add(listener)
                elif listener in self._listeners:
                    self._selector.unregister(listener._sock)
                    self._listeners.discard(listener)
                done.set()
            if self._listeners or self._pending:
                return True
            _RtcpPoller._instance = None
            return False

    def _run(self) -> None:
        try:
            while self._apply_pending():
                now = time.monotonic()
                next_due = min(listener._announce_if_due(now) for listener in self._listeners)
                for key, _ in self._selector.select(max(0.0, next_due - time.monotonic())):
                    if key.data is None:
                        while True:
                            try:
                                self._wake_recv.recv(64)
                            except (BlockingIOError, InterruptedError):
                                break
                        continue
                    try:
                        key.data._receive()
                    except Exception as exc:   # one output's callback must not stop the others
                        print(f"[rtcp] {key.data.janus_host}:{key.data.janus_rtcp_port}: {exc}")
        finally:
            self._selector.close()
            self._wake_recv.close()
            self._wake_send.close()
