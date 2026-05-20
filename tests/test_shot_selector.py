import unittest

from pyplumber.auto_mixer.shot_rules import (
    DEFAULT_MANUAL_SUGGESTION_FAMILIES,
    ShotRules,
)
from pyplumber.auto_mixer.shot_selector import HistoryAwareShotSelector


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def fixed_videoconf_rules():
    return ShotRules(
        name="fixed-videoconf",
        single_speaker_weights=(("videoconf", 100),),
        stack_rules=(),
        conversation_window_s=6.0,
        stack_hold_s=4.0,
        manual_suggestion_window_s=15.0,
        manual_suggestion_families=DEFAULT_MANUAL_SUGGESTION_FAMILIES,
        avoid_recent_scenes=3,
    )


class ShotSelectorManualSuggestionTest(unittest.TestCase):
    def test_manual_geometry_suggestion_is_not_bound_to_clicked_speaker(self):
        clock = FakeClock()
        selector = HistoryAwareShotSelector(
            {
                "videoconf_0",
                "videoconf_1",
                "full_face_0",
                "full_face_1",
            },
            n_inputs=2,
            rules=fixed_videoconf_rules(),
            clock=clock,
            seed=1,
        )

        self.assertEqual("videoconf_0", selector(0))

        selector.observe_program_scene("full_face_1", now=clock())
        self.assertEqual("full_face_0", selector(0))

        clock.advance(14.9)
        self.assertEqual("full_face_1", selector(1))

        clock.advance(0.2)
        self.assertEqual("videoconf_1", selector(1))

    def test_manual_stack_geometry_keeps_auto_switching_inside_stack(self):
        clock = FakeClock()
        selector = HistoryAwareShotSelector(
            {
                "videoconf_0",
                "videoconf_1",
                "videoconf_2",
                "vstack_0_1",
                "vstack_1_0",
                "vstack_2_0",
            },
            n_inputs=3,
            rules=fixed_videoconf_rules(),
            clock=clock,
            seed=1,
        )

        selector.observe_program_scene("vstack_1_0", now=clock())
        self.assertEqual("vstack_2_0", selector(2))


if __name__ == "__main__":
    unittest.main()
