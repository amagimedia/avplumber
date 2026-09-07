"""Codec selection for the same native playback tests on CPU and NVIDIA."""

import importlib.util
import shutil
import subprocess

import pytest

from replay import load_avp_api


def require_backend(backend):
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if importlib.util.find_spec("_avplumber") is None:
        missing.append("native _avplumber Python module")
    if missing:
        pytest.skip("requires integration host with " + ", ".join(missing))
    if backend == "nvidia":
        if shutil.which("nvidia-smi") is None:
            pytest.skip("requires NVIDIA driver tools")
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode:
            pytest.skip("requires a working NVIDIA driver")


def backend_api(backend):
    api = load_avp_api()
    if backend == "nvidia":
        return api

    class SoftwareAVPlumber(api.AVPlumber):
        def executeCommandsFromString(self, command):
            if command.startswith("hwaccel.init "):
                return
            return super().executeCommandsFromString(command)

        def addNode(self, node, *args, **kwargs):
            params = node.parameters
            if params["type"] == "dec_video":
                params.pop("hwaccel", None)
                params.pop("hwaccel_only_for_codecs", None)
                params["codec_map"] = {"h264": "h264", "hevc": "hevc"}
                params["pixel_format"] = "yuv420p"
                params["options"] = {"threads": "1"}
            elif params["type"] == "enc_video":
                params.pop("hwaccel", None)
                params["codec"] = "libx264"
                old = params["options"]
                params["options"] = {
                    "g": old["g"], "bf": 0, "preset": "ultrafast",
                    "tune": "zerolatency", "profile": "baseline",
                    "x264-params": "aud=1:repeat-headers=1:scenecut=0",
                }
                if old.get("b"):
                    params["options"].update({key: old[key] for key in ("b", "maxrate", "bufsize")})
                else:
                    params["options"]["crf"] = "17"
            return super().addNode(node, *args, **kwargs)

    api.AVPlumber = SoftwareAVPlumber
    return api
