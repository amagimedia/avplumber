import unittest
from types import SimpleNamespace

from pyplumber.auto_mixer.config import CANVAS_H, CANVAS_W
from pyplumber.auto_mixer.outputs import (
    build_html_overlay_output,
    build_janus_rtp_output,
    build_mux_output,
)


class FakeAVPlumber:
    def __init__(self):
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node.parameters)


class HtmlOverlayOutputTest(unittest.TestCase):
    def test_overlay_graph_starts_off_and_uses_tui_control_names(self):
        avp = FakeAVPlumber()

        out_edge = build_html_overlay_output(
            avp,
            "mixer_final",
            socket_path="/tmp/test.sock",
            source_hwaccel="@drm",
        )

        self.assertEqual("video_output", out_edge)

        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        self.assertEqual("ipc_dmabuf_source", by_name["html_overlay_src"]["type"])
        self.assertEqual("/tmp/test.sock", by_name["html_overlay_src"]["socket"])
        self.assertEqual("@drm", by_name["html_overlay_src"]["hwaccel"])
        self.assertEqual("output", by_name["html_overlay_src"]["group"])
        self.assertEqual("on", by_name["html_overlay_src"]["auto_restart"])

        overlay_otm = by_name["otm_html_overlay"]
        self.assertEqual("one_to_many", overlay_otm["type"])
        self.assertEqual("mixer_final", overlay_otm["src"])
        self.assertEqual(["no_overlay", "pre_overlay"], overlay_otm["dst"])
        self.assertEqual(1, overlay_otm["outputs"])
        self.assertTrue(overlay_otm["drop"])

        source_otm = by_name["otm_html_overlay_src"]
        self.assertEqual("one_to_many", source_otm["type"])
        self.assertEqual(0, source_otm["outputs"])
        self.assertTrue(source_otm["drop"])
        self.assertEqual("output", source_otm["group"])

        selector = by_name["overlay_sel"]
        self.assertEqual("source_switcher", selector["type"])
        self.assertEqual(["no_overlay", "post_overlay"], selector["src"])
        self.assertEqual("video_output", selector["dst"])
        self.assertEqual(0, selector["active"])
        self.assertEqual(0, selector["fallback_active"])
        self.assertTrue(selector["fallback_when_active_missing"])
        self.assertEqual(40, selector["fallback_wait_ms"])

        overlay_filter = by_name["html_overlay_filter"]
        self.assertTrue(overlay_filter["defer_preliminary_init"])

        html_assume = by_name["html_overlay_assume"]
        self.assertEqual(CANVAS_W, html_assume["width"])
        self.assertEqual(CANVAS_H, html_assume["height"])
        self.assertEqual("yuva420p", html_assume["real_pixel_format"])

    def test_overlay_source_hwaccel_is_optional(self):
        avp = FakeAVPlumber()

        build_html_overlay_output(avp, "mixer_final", socket_path="/tmp/test.sock")

        source = next(node for node in avp.nodes if node["name"] == "html_overlay_src")
        self.assertNotIn("hwaccel", source)


class MuxOutputPolicyTest(unittest.TestCase):
    def test_mux_output_errors_panic_but_clean_finish_stops_by_default(self):
        avp = FakeAVPlumber()

        build_mux_output(
            avp,
            ["video_enc", "audio_enc"],
            mux_edge="program_mux",
            output_url="/tmp/out.mp4",
            output_format="mp4",
        )

        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        self.assertEqual("off", by_name["program_mux_mux"]["auto_restart"])
        self.assertEqual("panic", by_name["program_mux_mux"]["on_error"])
        self.assertEqual("off", by_name["program_mux_output"]["auto_restart"])
        self.assertEqual("panic", by_name["program_mux_output"]["on_error"])

    def test_janus_rtp_outputs_restart_node_on_clean_finish(self):
        avp = FakeAVPlumber()
        args = SimpleNamespace(
            codec="h264_nvenc",
            janus_host="localhost",
            janus_video_port=5004,
            janus_audio_port=5002,
            janus_video_bitrate_kbps=1000,
            janus_video_codec="h264_nvenc",
            janus_audio_bitrate="96k",
            janus_video_pt=96,
            janus_audio_pt=111,
        )

        build_janus_rtp_output(avp, "video_out", "audio_out", args)

        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        for name in (
            "janus_video_rtp_mux_mux",
            "janus_video_rtp_mux_output",
            "janus_audio_rtp_mux_mux",
            "janus_audio_rtp_mux_output",
        ):
            self.assertEqual("on", by_name[name]["auto_restart"])
            self.assertEqual("panic", by_name[name]["on_error"])

    def test_janus_rtp_output_can_be_video_only(self):
        avp = FakeAVPlumber()
        args = SimpleNamespace(
            codec="h264_nvenc",
            janus_host="localhost",
            janus_video_port=5004,
            janus_audio_port=5002,
            janus_video_bitrate_kbps=1000,
            janus_video_codec="h264_nvenc",
            janus_audio_bitrate="96k",
            janus_video_pt=96,
            janus_audio_pt=111,
        )

        build_janus_rtp_output(avp, "video_out", None, args)

        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        self.assertIn("janus_video_rtp_mux_mux", by_name)
        self.assertNotIn("janus_enc_audio", by_name)
        self.assertNotIn("janus_audio_rtp_mux_mux", by_name)


if __name__ == "__main__":
    unittest.main()
