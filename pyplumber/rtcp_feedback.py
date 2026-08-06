"""RTCP feedback listener for Janus RTP outputs."""

from __future__ import annotations

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
    """Listens for RTCP PLI/FIR and triggers the configured keyframe callback."""

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
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.local_port: int | None = None
        self.total_packets = 0
        self.keyframe_requests = 0
        self.pli_requests = 0
        self.fir_requests = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.bind_port))
        sock.settimeout(0.2)
        self._sock = sock
        self.local_port = sock.getsockname()[1]
        self._thread = threading.Thread(
            target=self._run,
            name="janus-rtcp-feedback",
            daemon=True,
        )
        self._thread.start()
        print(
            "[rtcp] listening for Janus PLI/FIR on "
            f"{self.bind_host}:{self.local_port}, announcing to "
            f"{self.janus_host}:{self.janus_rtcp_port}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _announce(self) -> None:
        if self._sock is None:
            return
        cname = f"avplumber-{self.media_ssrc:x}"
        packet = (
            _build_rtcp_sr(media_ssrc=self.media_ssrc) +
            _build_rtcp_sdes(media_ssrc=self.media_ssrc, cname=cname)
        )
        self._sock.sendto(packet, (self.janus_host, self.janus_rtcp_port))

    def _run(self) -> None:
        next_announce = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_announce:
                try:
                    self._announce()
                except OSError as exc:
                    if not self._stop_event.is_set():
                        print(f"[rtcp] failed to send RTCP announcement: {exc}")
                next_announce = now + self.announce_interval_s
            try:
                data, addr = self._sock.recvfrom(2048) if self._sock else (b"", ("", 0))
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            self.total_packets += 1
            requests = rtcp_keyframe_requests(data)
            for request in requests:
                self.keyframe_requests += 1
                if request == "pli":
                    self.pli_requests += 1
                elif request == "fir":
                    self.fir_requests += 1
                print(f"[rtcp] received {request.upper()} from {addr[0]}:{addr[1]}")
                self.on_keyframe_request(request)
