import json
import unittest

from pyplumber.auto_mixer.auto_switch_control import (
    AutoSwitchControlCommands,
    handle_auto_switch_command,
)


class FakeSwitcher:
    def __init__(self):
        self.transition_mode = "cut"
        self.fade_duration_s = 0.5
        self.wipe_file = None

    def settings(self):
        return {
            "running": True,
            "transition_mode": self.transition_mode,
            "fade_duration_s": self.fade_duration_s,
            "wipe_file": self.wipe_file,
        }

    def configure(self, *, transition_mode=None, fade_duration_s=None, wipe_file=None):
        if transition_mode is not None:
            self.transition_mode = transition_mode
        if fade_duration_s is not None:
            self.fade_duration_s = float(fade_duration_s)
        if wipe_file is not None:
            self.wipe_file = wipe_file
        return self.settings()

    def set_transition_mode(self, mode):
        self.transition_mode = mode
        return self.settings()

    def set_fade_duration(self, seconds):
        self.fade_duration_s = float(seconds)
        return self.settings()


class FakeNativeExceptions:
    def summary(self):
        return {"total": 1, "recent": [{"node_name": "x"}], "by_node": [], "by_type": []}

    def snapshot(self):
        return {"total": 1, "events": [{"node_name": "x"}], "by_node": [], "by_type": []}


class FakeAVPlumber:
    def __init__(self):
        self.commands = {}

    def registerControlCommand(self, command, callback, no_lock=False):
        self.commands[command] = (callback, no_lock)


class AutoSwitchControlTest(unittest.TestCase):
    def test_status_includes_native_exception_summary(self):
        result = handle_auto_switch_command(
            FakeSwitcher(),
            "auto_switch.status",
            native_exceptions=FakeNativeExceptions(),
        )

        self.assertEqual("cut", result["transition_mode"])
        self.assertEqual(1, result["native_exceptions"]["total"])

    def test_set_validates_payload_and_updates_switcher(self):
        switcher = FakeSwitcher()

        result = handle_auto_switch_command(
            switcher,
            "auto_switch.set",
            '{"transition_mode":"wipe","fade_duration_s":0.75,"wipe_file":"/tmp/w.mov"}',
        )

        self.assertEqual("wipe", result["transition_mode"])
        self.assertEqual(0.75, result["fade_duration_s"])
        self.assertEqual("/tmp/w.mov", result["wipe_file"])

        with self.assertRaisesRegex(ValueError, "unknown setting"):
            handle_auto_switch_command(switcher, "auto_switch.set", '{"bad": true}')

    def test_registers_on_main_control_port(self):
        avp = FakeAVPlumber()
        switcher = FakeSwitcher()

        AutoSwitchControlCommands(avp, switcher, native_exceptions=FakeNativeExceptions()).start()

        self.assertIn("auto_switch.status", avp.commands)
        self.assertIn("auto_switch.set", avp.commands)
        self.assertTrue(avp.commands["auto_switch.status"][1])

        callback, _ = avp.commands["auto_switch.set"]
        payload = callback('{"transition_mode":"fade","fade_duration_s":1.0}')
        self.assertTrue(payload.endswith("\n"))
        data = json.loads(payload)
        self.assertEqual("fade", data["transition_mode"])
        self.assertEqual(1.0, data["fade_duration_s"])


if __name__ == "__main__":
    unittest.main()
