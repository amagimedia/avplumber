from types import SimpleNamespace

import pytest

from pyplumber.auto_mixer.run_config import derive_run_config


def _args(**overrides):
    values = {
        "inputs": ["rene.ts", "sergio.ts", "genaro.ts"],
        "output": None,
        "janus_preview": False,
        "janus_output": True,
        "janus_video_only": False,
        "janus_host": "127.0.0.1",
        "janus_video_port": 5004,
        "janus_audio_port": 5006,
        "janus_video_bitrate_kbps": 2500,
        "switch_pts_lead_ms": 600,
        "fade_frames": 15,
        "fade": None,
        "auto_switch_transition": "cut",
        "wipe": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_derive_run_config_defaults_to_cut_and_janus_only():
    args = _args()

    config = derive_run_config(args)

    assert args.auto_switch_transition == "cut"
    assert args.fade == 0.5
    assert config.n_inputs == 3
    assert config.janus_enabled is True
    assert config.record_enabled is False
    assert config.switch_pts_lead_ms == 600
    assert config.rene_input_index == 0
    assert config.sergio_input_index == 1
    assert config.genaro_input_index == 2
    assert config.output_targets == ("Janus RTP video=127.0.0.1:5004 audio=127.0.0.1:5006",)


def test_derive_run_config_fade_seconds_updates_frame_count():
    args = _args(fade=0.25)

    derive_run_config(args)

    assert args.fade_frames == 8


def test_derive_run_config_rejects_invalid_switch_pts_lead():
    with pytest.raises(ValueError, match="switch-pts-lead-ms"):
        derive_run_config(_args(switch_pts_lead_ms=50))
