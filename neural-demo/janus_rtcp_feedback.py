#!/usr/bin/env python3
"""Listen for Janus RTCP PLI/FIR and trigger an AVP keyframe node."""

from __future__ import annotations

import argparse
import socket
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass


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


def iter_rtcp_packets(data: bytes) -> Iterable[RtcpPacket]:
    offset = 0
    while offset + 4 <= len(data):
        first, packet_type, length_words = struct.unpack_from("!BBH", data, offset)
        if first >> 6 != 2:
            break
        packet_len = (length_words + 1) * 4
        if packet_len < 4 or offset + packet_len > len(data):
            break
        yield RtcpPacket(fmt=first & 0x1F, packet_type=packet_type)
        offset += packet_len


def rtcp_keyframe_requests(data: bytes) -> list[str]:
    requests: list[str] = []
    for packet in iter_rtcp_packets(data):
        if packet.packet_type == RTCP_PSFB and packet.fmt == RTCP_PSFB_PLI:
            requests.append("pli")
        elif packet.packet_type == RTCP_PSFB and packet.fmt == RTCP_PSFB_FIR:
            requests.append("fir")
        elif packet.packet_type == RTCP_LEGACY_FIR:
            requests.append("fir")
    return requests


def build_rtcp_sr(*, media_ssrc: int) -> bytes:
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


def build_rtcp_sdes(*, media_ssrc: int, cname: str) -> bytes:
    cname_bytes = cname.encode("ascii", "replace")[:255]
    body = struct.pack("!IBB", media_ssrc & 0xFFFFFFFF, 1, len(cname_bytes))
    body += cname_bytes + b"\x00"
    body += b"\x00" * ((4 - (len(body) % 4)) % 4)
    return struct.pack("!BBH", (2 << 6) | 1, RTCP_SDES, (len(body) + 4) // 4 - 1) + body


def send_avp_command(*, host: str, port: int, command: str) -> None:
    with socket.create_connection((host, port), timeout=0.5) as sock:
        sock.settimeout(0.5)
        try:
            sock.recv(1024)
        except TimeoutError:
            pass
        sock.sendall((command + "\nbye\n").encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=0)
    parser.add_argument("--janus-host", required=True)
    parser.add_argument("--janus-rtcp-port", type=int, required=True)
    parser.add_argument("--media-ssrc", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--force-keyframe-node", default="janus_force_keyframe")
    parser.add_argument("--announce-interval-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = f"node.object.set {args.force_keyframe_node} trigger true"
    cname = f"avplumber-{args.media_ssrc:x}"

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.bind_host, args.bind_port))
        sock.settimeout(0.2)
        local_host, local_port = sock.getsockname()
        print(
            "[rtcp] listening for Janus PLI/FIR on "
            f"{local_host}:{local_port}, announcing to "
            f"{args.janus_host}:{args.janus_rtcp_port}",
            flush=True,
        )

        next_announce = 0.0
        while True:
            now = time.monotonic()
            if now >= next_announce:
                packet = (
                    build_rtcp_sr(media_ssrc=args.media_ssrc)
                    + build_rtcp_sdes(media_ssrc=args.media_ssrc, cname=cname)
                )
                try:
                    sock.sendto(packet, (args.janus_host, args.janus_rtcp_port))
                except OSError as exc:
                    print(f"[rtcp] failed to send RTCP announcement: {exc}", flush=True)
                next_announce = now + args.announce_interval_s

            try:
                data, addr = sock.recvfrom(2048)
            except TimeoutError:
                continue

            for request in rtcp_keyframe_requests(data):
                print(f"[rtcp] received {request.upper()} from {addr[0]}:{addr[1]}", flush=True)
                try:
                    send_avp_command(
                        host=args.control_host,
                        port=args.control_port,
                        command=command,
                    )
                except OSError as exc:
                    print(f"[rtcp] failed to trigger Janus keyframe for {request}: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
