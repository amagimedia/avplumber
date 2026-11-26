"""Automatic GUI selection needs a seat and a real display socket."""
import os
from pathlib import Path
import socket
import subprocess

SCRIPT = Path(__file__).resolve().parents[2] / "demos/mixer/desktop/session.sh"


def test_local_login_and_display(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loginctl = tmp_path / "loginctl"
    loginctl.write_text('#!/bin/sh\nprintf "%s\\n" "User=$(id -u)" "Active=yes" "Remote=$REMOTE" "Seat=seat0" "Type=wayland"\n')
    loginctl.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "XDG_SESSION_ID": "2",
           "XDG_RUNTIME_DIR": str(tmp_path), "WAYLAND_DISPLAY": "desktop", "REMOTE": "no"}

    def selected(**changes):
        return subprocess.run(["bash", "-c", 'source "$1"; local_graphical_login && desktop_display',
                               "bash", str(SCRIPT)], env={**env, **changes}).returncode == 0

    assert not selected()
    with socket.socket(socket.AF_UNIX) as display:
        display.bind("desktop")
        assert selected()
        assert not selected(REMOTE="yes")
        assert not selected(SSH_CONNECTION="forwarded")
        assert not selected(WAYLAND_DISPLAY="missing")
