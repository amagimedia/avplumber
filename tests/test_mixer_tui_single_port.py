from pathlib import Path
import sys


TUI_DIR = Path("tools/mixer_tui").resolve()
if str(TUI_DIR) not in sys.path:
    sys.path.insert(0, str(TUI_DIR))

from commands import auto_switch_set_command
from state import parse_auto_switch_status, parse_mixer_status, parse_scene_list


def test_tui_uses_main_control_port_for_auto_switch_settings():
    source = Path("tools/mixer_tui/mixer_tui.py").read_text(encoding="utf-8")

    assert "--auto-port" not in source
    assert "--auto-host" not in source
    assert "_auto_conn" not in source
    assert 'self._conn.command("auto_switch.status"' in source
    assert "resp = await self._conn.command(cmd, raise_for_status=False)" in source


def test_tui_serializes_shared_control_connection_commands():
    source = Path("tools/mixer_tui/avp_client.py").read_text(encoding="utf-8")

    assert "self._command_lock = asyncio.Lock()" in source
    command_method = source.split("async def command(self, cmd: str, raise_for_status: bool = True)", 1)[1]
    command_method = command_method.split("async def disconnect", 1)[0]
    assert "async with self._command_lock:" in command_method


def test_tui_uses_bounded_poll_ticks_instead_of_stale_worker_loop():
    source = Path("tools/mixer_tui/mixer_tui.py").read_text(encoding="utf-8")

    assert 'self._connection_timer = self.set_interval(0.5, self._connection_tick' in source
    assert 'self._poll_timer = self.set_interval(0.5, self._poll_status_tick' in source
    assert 'self._scene_timer = self.set_interval(10.0, self._fetch_scenes_tick' in source
    assert "await self._refresh_auto_status_tick()" in source
    assert "async def _start_connection_loop" not in source
    assert "async def _poll_status(self)" not in source


def test_ai_transition_shortcut_is_not_swallowed_by_fade_input_focus():
    source = Path("tools/mixer_tui/mixer_tui.py").read_text(encoding="utf-8")
    widget_source = Path("tools/mixer_tui/widgets.py").read_text(encoding="utf-8")

    assert 'Binding("a", "toggle_ai_transition_mode", "AI transition", show=False, priority=True)' in source
    assert "self.call_after_refresh(self._clear_initial_focus)" in source
    assert "self.set_focus(None)" in source
    assert "class FadeFramesInput(Input):" in widget_source
    assert "self.app.action_toggle_ai_transition_mode()" in widget_source



def test_tui_has_no_html_overlay_controls():
    source = Path("tools/mixer_tui/mixer_tui.py").read_text(encoding="utf-8")
    command_source = Path("tools/mixer_tui/commands.py").read_text(encoding="utf-8")

    assert "--overlay" not in source
    assert "btn_overlay" not in source
    assert "toggle_overlay" not in source
    assert "overlay_toggle_commands" not in command_source


def test_auto_switch_set_command_uses_main_protocol_payload():
    cmd = auto_switch_set_command(
        fade_duration_s=0.5,
        transition_mode="wipe",
        wipe_file="/tmp/w.mov",
    )

    assert cmd == 'auto_switch.set {"fade_duration_s":0.5,"transition_mode":"wipe","wipe_file":"/tmp/w.mov"}'


def test_status_parsers_extract_tui_state():
    mixer = parse_mixer_status('{"pgm_scene":"a","pvw_scene":"b","transition":"wipe"}\n')
    auto = parse_auto_switch_status('{"transition_mode":"cut","fade_duration_s":0.5,"wipe_file":"/tmp/w.mov"}\n')
    scenes = parse_scene_list('["a","b"]\n')

    assert mixer.pgm_scene == "a"
    assert mixer.pvw_scene == "b"
    assert mixer.transition == "wipe"
    assert auto.transition_mode == "cut"
    assert auto.fade_duration_s == 0.5
    assert auto.wipe_file == "/tmp/w.mov"
    assert scenes == ["a", "b"]
