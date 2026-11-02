import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("publish", Path(__file__).parents[1] / "publish.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)

REVISION = "a" * 40
DIGEST = "sha256:" + "b" * 64
PREFIX = "us-east1-docker.pkg.dev/example-project/mixer/mixer"
MIRROR = "europe-west1-docker.pkg.dev/example-project/mixer/mixer"


def test_compose_builds_keep_names_and_pins(tmp_path, monkeypatch):
    services = {"mixer": {"image": PREFIX + "-mixer:" + REVISION, "build": {"context": "/source"}},
                "janus-preview": {"image": PREFIX + "-preview:" + REVISION,
                                  "build": {"context": "/source"}},
                "seed": {"image": PREFIX + "-mixer:" + REVISION},
                "proxy": {"image": "nginx:alpine"}}
    monkeypatch.setattr(publish, "run", lambda *args, **kwargs: json.dumps({"services": services}))
    publish.prepare(tmp_path, PREFIX, REVISION)
    targets = json.loads((tmp_path / "cache.json").read_text())["target"]
    assert set(targets) == {"mixer", "janus-preview"}
    assert "ref=" + PREFIX + "-preview:buildcache" in targets["janus-preview"]["cache-to"][0]
    compose = json.loads((tmp_path / "build.compose.yaml").read_text())
    assert compose["services"]["mixer"]["build"] == {"context": "/source"}
    monkeypatch.setattr(publish, "run", lambda *args: DIGEST)
    publish.write_json(tmp_path / "metadata.json", {name: {"containerimage.digest": DIGEST} for name in targets})
    copies = []
    monkeypatch.setattr(publish, "copy_image", lambda *args: copies.append(args))
    publish.release(tmp_path, [MIRROR])
    pins = json.loads((tmp_path / (MIRROR.replace("/", "--") + ".compose.yaml")).read_text())["services"]
    assert pins["seed"] == pins["mixer"] == {"image": MIRROR + "-mixer@" + DIGEST}
    assert pins["janus-preview"] == {"image": MIRROR + "-preview@" + DIGEST}
    assert pins["proxy"] == {"image": MIRROR + "-proxy@" + DIGEST}
    assert (PREFIX + "-preview@" + DIGEST, MIRROR + "-preview:" + REVISION, DIGEST) in copies
    assert not json.loads((tmp_path / "release.json").read_text())["gpu_playback_verified"]


def test_reject_mismatched_registry_or_release(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "run", lambda *args, **kwargs: json.dumps({"services": {
        "mixer": {"image": "avplumber-mixer:local", "build": {}}, "proxy": {"image": "nginx:alpine"}}}))
    with pytest.raises(ValueError, match="Unexpected image"):
        publish.prepare(tmp_path, PREFIX, REVISION)


def test_mirror_verifies_digest(monkeypatch):
    monkeypatch.setattr(publish.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(publish, "run", lambda *args: "sha256:" + "c" * 64)
    with pytest.raises(ValueError, match="expected"):
        publish.copy_image(PREFIX + "@" + DIGEST, MIRROR + ":" + REVISION, DIGEST)
