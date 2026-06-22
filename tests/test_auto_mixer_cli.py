from types import SimpleNamespace

import pytest

from pyplumber.auto_mixer.run_config import derive_run_config


def _args(**overrides):
    values = {
        "inputs": ["cam0.ts", "cam1.ts", "cam2.ts"],
        "output": None,
        "output_format": None,
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
        "talkshow_profile": False,
        "program_audio_input": None,
        "special_speaker_index": None,
        "special_speaker_margin_db": 3.0,
        "vad_only_priority_speaker_index": None,
        "static_face_crop_input": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_derive_run_config_defaults_to_generic_inputs_and_janus_only():
    args = _args()

    config = derive_run_config(args)

    assert args.auto_switch_transition == "cut"
    assert args.fade == 0.5
    assert config.n_inputs == 3
    assert config.janus_enabled is True
    assert config.record_enabled is False
    assert config.switch_pts_lead_ms == 600
    assert config.talkshow_profile is False
    assert config.program_audio_input_index == 0
    assert config.special_speaker_index is None
    assert config.vad_only_priority_speaker_index is None
    assert config.static_face_crop_inputs == ()
    assert config.output_targets == ("Janus RTP video=127.0.0.1:5004 audio=127.0.0.1:5006",)


def test_derive_run_config_talkshow_profile_uses_index_defaults():
    args = _args(talkshow_profile=True)

    config = derive_run_config(args)

    assert config.talkshow_profile is True
    assert config.program_audio_input_index == 0
    assert config.special_speaker_index == 0
    assert config.special_speaker_margin_db == 3.0
    assert config.vad_only_priority_speaker_index == 1
    assert config.static_face_crop_inputs == (2,)


def test_derive_run_config_allows_generic_role_overrides():
    config = derive_run_config(
        _args(
            inputs=["cam0.ts", "cam1.ts", "cam2.ts"],
            talkshow_profile=True,
            program_audio_input=1,
            special_speaker_index=2,
            special_speaker_margin_db=4.5,
            vad_only_priority_speaker_index=0,
            static_face_crop_input=[1],
        )
    )

    assert config.program_audio_input_index == 1
    assert config.special_speaker_index == 2
    assert config.special_speaker_margin_db == 4.5
    assert config.vad_only_priority_speaker_index == 0
    assert config.static_face_crop_inputs == (2, 1)


def test_derive_run_config_validates_program_audio_input_index():
    with pytest.raises(ValueError, match="program-audio-input"):
        derive_run_config(_args(inputs=["cam0.ts", "cam1.ts"], program_audio_input=2))


def test_derive_run_config_validates_generic_role_indices():
    with pytest.raises(ValueError, match="special-speaker-index"):
        derive_run_config(_args(inputs=["cam0.ts", "cam1.ts"], special_speaker_index=2))
    with pytest.raises(ValueError, match="vad-only-priority-speaker-index"):
        derive_run_config(_args(inputs=["cam0.ts", "cam1.ts"], vad_only_priority_speaker_index=2))
    with pytest.raises(ValueError, match="static-face-crop-input"):
        derive_run_config(_args(inputs=["cam0.ts", "cam1.ts"], static_face_crop_input=[2]))


def test_derive_run_config_fade_seconds_updates_frame_count():
    args = _args(fade=0.25)

    derive_run_config(args)

    assert args.fade_frames == 8


def test_derive_run_config_rejects_invalid_switch_pts_lead():
    with pytest.raises(ValueError, match="switch-pts-lead-ms"):
        derive_run_config(_args(switch_pts_lead_ms=50))


def test_derive_run_config_infers_supported_record_output_formats():
    assert derive_run_config(_args(output="rtmp://host/app/out")).record_output_format == "flv"
    assert derive_run_config(_args(output="srt://host:9000")).record_output_format == "mpegts"
    assert derive_run_config(_args(output="/tmp/out.ts")).record_output_format == "mpegts"
    assert derive_run_config(_args(output="/tmp/out.flv")).record_output_format == "flv"


def test_derive_run_config_allows_explicit_supported_record_output_format():
    config = derive_run_config(_args(output="/tmp/out", output_format="mpegts"))

    assert config.record_output_format == "mpegts"


def test_derive_run_config_rejects_unsupported_record_outputs():
    with pytest.raises(ValueError, match="Recording output"):
        derive_run_config(_args(output="/tmp/out.mp4"))
    with pytest.raises(ValueError, match="SRT outputs require"):
        derive_run_config(_args(output="srt://host:9000", output_format="flv"))
    with pytest.raises(ValueError, match="RTMP outputs require"):
        derive_run_config(_args(output="rtmp://host/app/out", output_format="mpegts"))
