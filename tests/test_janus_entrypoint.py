"""The Janus entrypoint's address choice: the interface address WebRTC viewers can reach, and the
public address of a cloud VM behind 1:1 NAT, with `ip`, `curl` and the DMI strings stubbed."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker-compose/images/janus/janus-entrypoint.sh"

IP_STUB = """#!/bin/sh
case "$*" in
  "-4 route show default") [ -n "$ROUTE_DEV" ] && echo "default via 10.0.0.1 dev $ROUTE_DEV proto dhcp metric 100" ;;
  "-o -4 addr show dev eth0 scope global") echo "2: eth0    inet 10.0.0.5/24 brd 10.0.0.255 scope global dynamic eth0" ;;
esac
"""
CURL_STUB = """#!/bin/sh
echo called >> "$CURL_LOG"
case "$*" in *metadata.google.internal*) echo 34.1.2.3 ;; *) exit 7 ;; esac
"""


def run(tmp_path, *args, vendor="Micro-Star International Co., Ltd.", **env):
    stubs, dmi = tmp_path / "bin", tmp_path / "dmi"
    stubs.mkdir(exist_ok=True)
    dmi.mkdir(exist_ok=True)
    for name, body in (("ip", IP_STUB), ("curl", CURL_STUB)):
        (stubs / name).write_text(body)
        (stubs / name).chmod(stat.S_IRWXU)
    (dmi / "sys_vendor").write_text(vendor + "\n")
    base = {"PATH": f"{stubs}:/usr/bin:/bin", "JANUS_DMI_DIR": str(dmi), "ROUTE_DEV": "eth0",
            "CURL_LOG": str(tmp_path / "curl.log")}
    return subprocess.run(["bash", str(ENTRYPOINT), *args], capture_output=True, text=True, timeout=30,
                          env={**base, **env})


@pytest.mark.parametrize("value", [None, "auto", "127.0.0.1"])
def test_unset_auto_or_loopback_takes_the_default_route_address(tmp_path, value):
    result = run(tmp_path, "--print-host-address", **({} if value is None else {"JANUS_HOST_IP": value}))
    assert result.returncode == 0 and result.stdout == "10.0.0.5\n"
    assert ("loopback" in result.stderr) == (value == "127.0.0.1")


def test_an_explicit_address_is_kept(tmp_path):
    result = run(tmp_path, "--print-host-address", JANUS_HOST_IP="192.168.1.20")
    assert result.returncode == 0 and result.stdout == "192.168.1.20\n"


def test_no_default_route_fails_with_a_hint(tmp_path):
    result = run(tmp_path, "--print-host-address", ROUTE_DEV="")
    assert result.returncode == 1 and "set JANUS_HOST_IP" in result.stderr


def test_nat_auto_off_the_cloud_skips_the_metadata_lookups(tmp_path):
    result = run(tmp_path, "--print-public-address", JANUS_NAT_1_1="auto")
    assert result.returncode == 0 and result.stdout == "\n"
    assert not (tmp_path / "curl.log").exists()


def test_nat_auto_on_gce_reads_the_public_address(tmp_path):
    result = run(tmp_path, "--print-public-address", JANUS_NAT_1_1="auto", vendor="Google")
    assert result.returncode == 0 and result.stdout == "34.1.2.3\n"


def test_an_explicit_public_address_is_kept(tmp_path):
    result = run(tmp_path, "--print-public-address", JANUS_NAT_1_1="203.0.113.7")
    assert result.stdout == "203.0.113.7\n" and not (tmp_path / "curl.log").exists()
