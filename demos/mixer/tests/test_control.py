import json

import pytest

from control import mixer_command, parse_mixer_status, parse_scene_list


def test_mixer_command_uses_generic_json_payload():
    command = mixer_command("fade", "main", scene="grid_4_page_0", duration_sec=0.5)
    prefix, payload = command.split(" ", 1)
    assert prefix == "mixer.fade"
    assert json.loads(payload) == {
        "mixer": "main",
        "scene": "grid_4_page_0",
        "duration_sec": 0.5,
    }


def test_status_and_scene_parsing():
    status = parse_mixer_status(
        '{"pgm_scene":"fullscreen_0","pvw_scene":"grid_2_page_0","transition":"idle"}'
    )
    assert status.pgm_scene == "fullscreen_0"
    assert status.pvw_scene == "grid_2_page_0"
    assert status.transition == "idle"
    assert parse_scene_list('["fullscreen_0", "grid_2_page_0"]') == [
        "fullscreen_0",
        "grid_2_page_0",
    ]


def test_invalid_response_shapes_are_rejected():
    with pytest.raises(ValueError, match="object"):
        parse_mixer_status("[]")
    with pytest.raises(ValueError, match="list"):
        parse_scene_list("{}")
