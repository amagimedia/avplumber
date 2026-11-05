"""The show document's audio section: shape, resolution against audio_mixer's own channels, and the
live controller. The node itself (set_processor, the ducking) is test_audio_mixer_native.py's."""
import json

import pytest

from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.audio_processors import AudioProcessorController, parse_audio, resolve
from test_graph import CONFIG


def _ducker(**over):
    return {"id": "duck", "type": "ducker", "target": ["rene"], "sidechain": ["denise", "leo"], **over}


def _audio(*processors, channel="program"):
    return {"channels": [{"id": channel, "type": "output", "processors": list(processors)}]}


def test_a_ducker_takes_defaults_and_resolves_to_the_node_parameter():
    channels = parse_audio(_audio(_ducker(threshold_db=-30)))
    assert resolve(channels, ["rene", "denise", "leo"], ["program", "aux2"]) == {"program": [{
        "id": "duck", "type": "ducker", "target": ["rene"], "sidechain": ["denise", "leo"],
        "sidechain_tap": "post", "enabled": True, "threshold_db": -30.0, "depth_db": -20.0,
        "attack_ms": 10.0, "hold_ms": 300.0, "release_ms": 500.0}]}


def test_no_section_adds_nothing():
    assert parse_audio(None) == ()
    assert resolve((), ["rene"], ["program"]) == {}


@pytest.mark.parametrize("bad, match", [
    (_ducker(type="compressor"), "type must be ducker"),
    (_ducker(target=[]), "target must be a non-empty list"),
    (_ducker(sidechain=["rene"]), "both target and sidechain"),
    (_ducker(sidechain_tap="side"), "sidechain_tap"),
    (_ducker(depth_db=6), "depth_db must be a number"),
    (_ducker(attack_ms=True), "attack_ms must be a number"),
    (_ducker(enabled="yes"), "enabled must be a boolean"),
    (_ducker(threshold=-40), "unknown keys threshold"),
])
def test_a_bad_ducker_is_a_config_error(bad, match):
    with pytest.raises(mixer_config.ConfigError, match=match):
        parse_audio(_audio(bad))


def test_processors_are_only_on_outputs():
    with pytest.raises(mixer_config.ConfigError, match="only on output"):
        parse_audio({"channels": [{"id": "rene", "type": "input", "processors": [_ducker()]}]})


def test_a_channel_the_mixer_did_not_create_is_an_error():
    """A typo, or an AUX bus without audio, must not quietly do nothing."""
    with pytest.raises(mixer_config.ConfigError, match="no output channel 'aux9'"):
        resolve(parse_audio(_audio(_ducker(), channel="aux9")), ["rene", "denise", "leo"], ["program"])
    with pytest.raises(mixer_config.ConfigError, match="no input channel leo"):
        resolve(parse_audio(_audio(_ducker())), ["rene", "denise"], ["program"])


def test_the_document_carries_it_into_settings():
    camera = {**CONFIG["sources"][0], "has_audio": True}
    cfg = mixer_config.parse({**CONFIG, "sources": [camera, *CONFIG["sources"][1:]], "audio": _audio(
        {"id": "d", "type": "ducker", "target": [camera["id"]], "sidechain": ["other"]})})
    rows = cfg.settings()["audio_processors"]
    assert [(r["output"], r["id"], r["target"]) for r in rows] == [("program", "d", [camera["id"]])]


class _Avp:
    def __init__(self, status=None):
        self.sent, self.status = [], status

    def node(self, _name):
        if self.status is None:
            raise RuntimeError("no node yet")
        return self

    def getObject(self, _key):
        return self.status

    def executeCommandsFromString(self, command):
        self.sent.append(command)


def test_the_controller_checks_the_whole_request_before_sending():
    avp = _Avp()
    controller = AudioProcessorController(avp, "audio_mixer", parse_audio(_audio(_ducker())))
    for bad in ({"output": "program", "id": "nope", "depth_db": -10},
                {"output": "program", "id": "duck", "depth_db": -10, "hold_ms": float("nan")},
                {"output": "program", "id": "duck"}):
        with pytest.raises(mixer_config.ConfigError):
            controller.set(bad)
    assert avp.sent == []
    rows = controller.set({"output": "program", "id": "duck", "depth_db": -12, "enabled": False})
    assert len(avp.sent) == 1
    _, name, key, payload = avp.sent[0].split(" ", 3)
    assert (name, key) == ("audio_mixer", "set_processor")
    assert json.loads(payload) == {"output": "program", "id": "duck", "depth_db": -12.0, "enabled": False}
    assert rows[0]["depth_db"] == -12.0 and rows[0]["enabled"] is False


def test_the_controller_reports_what_the_node_applies():
    avp = _Avp({"buses": [{"id": "program", "processors": [
        {"id": "duck", "type": "ducker", "enabled": True, "threshold_db": -35.0, "gain_db": -7.5}]}]})
    controller = AudioProcessorController(avp, "audio_mixer", parse_audio(_audio(_ducker())))
    (row,) = controller.state()
    assert row["threshold_db"] == -35.0 and row["gain_db"] == -7.5 and row["output"] == "program"
