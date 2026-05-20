import unittest

from pyplumber.auto_switcher import AutoSwitcher


class FakeMixer:
    def __init__(self, scenes):
        self._scenes = set(scenes)
        self.calls = []

    def scenes(self):
        return set(self._scenes)

    def cut(self, scene, start_pts_ms=-1):
        self.calls.append(("cut", scene, {"start_pts_ms": start_pts_ms}))

    def fade(self, scene, duration_sec=1.0, start_pts_ms=-1):
        self.calls.append(
            ("fade", scene, {"duration_sec": duration_sec, "start_pts_ms": start_pts_ms})
        )

    def wipe(self, scene, wipe_file, duration_sec=None, start_pts_ms=-1):
        kwargs = {"wipe_file": wipe_file, "start_pts_ms": start_pts_ms}
        if duration_sec is not None:
            kwargs["duration_sec"] = duration_sec
        self.calls.append(("wipe", scene, kwargs))


class FakeEntry:
    audio_event_pts_ms = 1000
    audio_event_observed_ts = 9.0


class AutoSwitcherWipeTest(unittest.TestCase):
    def test_default_auto_transition_is_cut(self):
        mixer = FakeMixer({"cam0", "cam1"})
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=object(),
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            fade_duration_s=0.5,
            wipe_file="alpha.mov",
        )

        self.assertEqual("cut", switcher.settings()["transition_mode"])
        self.assertTrue(switcher._try_switch_to(1, now=10.0))
        self.assertEqual(
            [("cut", "cam1", {"start_pts_ms": -1})],
            mixer.calls,
        )

    def test_auto_wipe_uses_media_duration(self):
        mixer = FakeMixer({"cam0", "cam1"})
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=object(),
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            fade_duration_s=0.5,
            wipe_file="alpha.mov",
            transition_mode="wipe",
        )

        self.assertTrue(switcher._try_switch_to(1, now=10.0))

        self.assertEqual(
            [("wipe", "cam1", {"wipe_file": "alpha.mov", "start_pts_ms": -1})],
            mixer.calls,
        )

    def test_scheduled_auto_wipe_does_not_force_fade_duration(self):
        mixer = FakeMixer({"cam0", "cam1"})
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=object(),
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            fade_duration_s=0.5,
            wipe_file="alpha.mov",
            transition_mode="wipe",
            switch_pts_lead_ms=600,
        )

        self.assertTrue(switcher._try_switch_to(1, now=10.0, entry=FakeEntry()))

        self.assertEqual(
            [("wipe", "cam1", {"wipe_file": "alpha.mov", "start_pts_ms": 2600})],
            mixer.calls,
        )


if __name__ == "__main__":
    unittest.main()
