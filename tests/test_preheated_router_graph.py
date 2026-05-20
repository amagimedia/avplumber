import json
import unittest

from pyplumber.auto_mixer.preheated import (
    build_preheated_scene_sources,
    define_preheated_scenes,
)
from pyplumber.auto_mixer.shot_selector import AutoShotSceneBuilder
from pyplumber.mixer import MixerGraphBuilder


class FakeAVPlumber:
    def __init__(self):
        self.nodes = []
        self.commands = []

    def addNode(self, node):
        self.nodes.append(node.parameters)

    def executeCommandsFromString(self, commands):
        self.commands.extend(line for line in commands.splitlines() if line.strip())


class PreheatedRouterGraphTest(unittest.TestCase):
    def test_wipe_selector_falls_back_to_direct_program_branch(self):
        avp = FakeAVPlumber()
        mx = MixerGraphBuilder(avp, name="mixer", enable_wipe=True)

        mx.add_source("cam0", pre_otm_edge="cam0", input_group="input_0")
        mx.add_scene(
            "full_cam0",
            {"cam0": {"graph": "scale_cuda=w=1920:h=1080", "dst_x": 0, "dst_y": 0}},
        )
        mx.set_initial_scene("full_cam0")
        mx.build()

        wipe_sel = next(n for n in avp.nodes if n["name"] == "mixer_wipe_sel")
        self.assertEqual("source_switcher", wipe_sel["type"])
        self.assertEqual(["mixer_final_direct", "mixer_wipe_overlay_out"], wipe_sel["src"])
        self.assertEqual(0, wipe_sel["fallback_active"])
        self.assertEqual(0, wipe_sel["timeline_reference_input"])
        self.assertFalse(wipe_sel["fallback_when_active_missing"])

        wipe_overlay = next(n for n in avp.nodes if n["name"] == "mixer_wipe_overlay")
        self.assertTrue(wipe_overlay["defer_preliminary_init"])

    def test_preheated_sources_use_two_native_routers_and_explicit_scene_routes(self):
        avp = FakeAVPlumber()
        mx = MixerGraphBuilder(avp, name="mixer", enable_wipe=False)

        preheated = build_preheated_scene_sources(
            avp,
            mx,
            face_edges=[f"face_{i}" for i in range(6)],
            orig_edges=[f"orig_{i}" for i in range(6)],
            input_groups=[f"input_{i}" for i in range(6)],
        )
        define_preheated_scenes(mx, 6, preheated)
        mx.set_initial_scene("full_face_0")
        mx.build()

        routers = [n for n in avp.nodes if n["type"] == "preheat_video_router"]
        self.assertEqual(2, len(routers))
        self.assertEqual(14, len(next(n for n in routers if n["name"] == "preheat_face_router")["dst"]))
        self.assertEqual(8, len(next(n for n in routers if n["name"] == "preheat_orig_router")["dst"]))

        hot_selectors = [
            n for n in avp.nodes
            if n["type"] == "source_switcher" and n["name"].startswith("hot_")
        ]
        self.assertEqual([], hot_selectors)

        source_otms = [
            n for n in avp.nodes
            if n["type"] == "one_to_many" and n["name"].startswith("mixer_otm_hot_")
        ]
        self.assertEqual([], source_otms)

        routed_sources = [line for line in avp.commands if line.startswith("mixer.routed_source ")]
        self.assertEqual(11, len(routed_sources))
        self.assertIn("mixer.init_routes mixer", avp.commands)

        full_face_cmd = next(
            line for line in avp.commands
            if line.startswith("mixer.scene mixer full_face_0 ")
        )
        scene_json = json.loads(full_face_cmd.split(" ", 3)[3])
        self.assertEqual({"hot_face_full_0": 0}, scene_json["routes"])
        self.assertNotIn("controls", scene_json)

    def test_regular_mixer_edges_split_inputs_before_the_routers(self):
        avp = FakeAVPlumber()
        mx = MixerGraphBuilder(avp, name="mixer", enable_wipe=False)

        preheated = build_preheated_scene_sources(
            avp,
            mx,
            face_edges=[f"face_{i}" for i in range(2)],
            orig_edges=[f"orig_{i}" for i in range(2)],
            input_groups=[f"input_{i}" for i in range(2)],
            include_regular_mixer_edges=True,
        )

        self.assertEqual(["v0_face_for_mixer", "v1_face_for_mixer"], preheated.face_mixer_edges)
        self.assertEqual(["v0_orig_for_mixer", "v1_orig_for_mixer"], preheated.orig_mixer_edges)

        face_split = next(n for n in avp.nodes if n["name"] == "split_preheat_face_0")
        orig_split = next(n for n in avp.nodes if n["name"] == "split_preheat_orig_0")
        self.assertEqual(["v0_face_for_mixer", "v0_face_for_preheat_router"], face_split["dst"])
        self.assertEqual(["v0_orig_for_mixer", "v0_orig_for_preheat_router"], orig_split["dst"])

        routers = [n for n in avp.nodes if n["type"] == "preheat_video_router"]
        face_router = next(n for n in routers if n["name"] == "preheat_face_router")
        orig_router = next(n for n in routers if n["name"] == "preheat_orig_router")
        self.assertEqual(
            ["v0_face_for_preheat_router", "v1_face_for_preheat_router"],
            face_router["src"],
        )
        self.assertEqual(
            ["v0_orig_for_preheat_router", "v1_orig_for_preheat_router"],
            orig_router["src"],
        )

    def test_dynamic_auto_scenes_use_routes_without_selector_controls(self):
        avp = FakeAVPlumber()
        mx = MixerGraphBuilder(avp, name="mixer", enable_wipe=False)

        preheated = build_preheated_scene_sources(
            avp,
            mx,
            face_edges=[f"face_{i}" for i in range(3)],
            orig_edges=[f"orig_{i}" for i in range(3)],
            input_groups=[f"input_{i}" for i in range(3)],
        )
        AutoShotSceneBuilder(mx, n_inputs=3, preheated=preheated).register_initial_scenes()
        mx.set_initial_scene("auto_pip")
        mx.build()

        auto_pip_cmd = next(
            line for line in avp.commands
            if line.startswith("mixer.scene mixer auto_pip ")
        )
        auto_pip_json = json.loads(auto_pip_cmd.split(" ", 3)[3])
        self.assertEqual(
            {"hot_face_full_0": 0, "hot_orig_pip_thumb_0": 1},
            auto_pip_json["routes"],
        )
        self.assertNotIn("controls", auto_pip_json)


if __name__ == "__main__":
    unittest.main()
