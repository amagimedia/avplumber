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
    def __init__(
        self,
        index=0,
        *,
        speaking=True,
        visual_speaking=True,
        level_db=-20.0,
        since=0.0,
    ):
        self.index = index
        self.speaking = speaking
        self.visual_speaking = visual_speaking
        self.level_db = level_db
        self.audio_since = since
        self.visual_since = since
        self.combined_speaking_since = since if speaking and visual_speaking else None
        self.last_change_ts = since
        self.audio_event_pts_ms = 1000
        self.audio_event_observed_ts = 9.0


class FakeRegistry:
    def __init__(self, entries):
        self._entries = {entry.index: entry for entry in entries}

    def get(self, index):
        return self._entries.get(index)


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

    def test_vad_only_priority_input_wins_before_audio_visual_candidates(self):
        mixer = FakeMixer({"cam0", "cam1"})
        registry = FakeRegistry([
            FakeEntry(0, speaking=True, visual_speaking=True, level_db=-10.0),
            FakeEntry(1, speaking=True, visual_speaking=False, level_db=-80.0),
        ])
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=registry,
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            min_dwell_speaking_s=0.0,
            cooldown_s=0.0,
            vad_only_priority_speaker_index=1,
        )

        switcher._tick_once()

        self.assertEqual([("cut", "cam1", {"start_pts_ms": -1})], mixer.calls)

    def test_audio_visual_candidates_choose_loudest_level(self):
        mixer = FakeMixer({"cam0", "cam1"})
        registry = FakeRegistry([
            FakeEntry(0, speaking=True, visual_speaking=True, level_db=-12.0),
            FakeEntry(1, speaking=True, visual_speaking=True, level_db=-8.0),
        ])
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=registry,
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            min_dwell_speaking_s=0.0,
            cooldown_s=0.0,
        )

        switcher._tick_once()

        self.assertEqual([("cut", "cam1", {"start_pts_ms": -1})], mixer.calls)

    def test_special_speaker_must_clear_configured_db_margin(self):
        mixer = FakeMixer({"cam0", "cam1"})
        registry = FakeRegistry([
            FakeEntry(0, speaking=True, visual_speaking=True, level_db=-10.0),
            FakeEntry(1, speaking=True, visual_speaking=True, level_db=-12.0),
        ])
        switcher = AutoSwitcher(
            mixer=mixer,
            registry=registry,
            n_inputs=2,
            scene_for_input=lambda i: f"cam{i}",
            min_dwell_program_s=0.0,
            min_dwell_speaking_s=0.0,
            cooldown_s=0.0,
            special_speaker_index=0,
            special_speaker_margin_db=3.0,
        )

        switcher._tick_once()

        self.assertEqual([("cut", "cam1", {"start_pts_ms": -1})], mixer.calls)


if __name__ == "__main__":
    unittest.main()
