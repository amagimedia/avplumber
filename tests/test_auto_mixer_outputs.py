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


class MuxOutputPolicyTest(unittest.TestCase):
    def test_html_overlay_graph_starts_bypassed_with_host_control_names(self):
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
        self.assertEqual("html_dma", by_name["html_overlay_src"]["dst"])
        self.assertEqual("on", by_name["html_overlay_src"]["auto_restart"])

        self.assertEqual("html_cuda", by_name["html_to_cuda"]["dst"])
        self.assertEqual("html_cuda_fps", by_name["html_fps"]["dst"])
        self.assertEqual("html_overlay_yuva", by_name["html_convert"]["dst"])
        self.assertEqual("convert_cuda=format=yuva420p", by_name["html_convert"]["graph"])

        html_assume = by_name["html_overlay_assume"]
        self.assertEqual(CANVAS_W, html_assume["width"])
        self.assertEqual(CANVAS_H, html_assume["height"])
        self.assertEqual("yuva420p", html_assume["real_pixel_format"])

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

        overlay_filter = by_name["html_overlay_filter"]
        self.assertEqual(["pre_overlay_norm", "html_overlay_enabled"], overlay_filter["src"])
        self.assertIn("overlay_many_cuda=inputs=2", overlay_filter["graph"])
        self.assertTrue(overlay_filter["defer_preliminary_init"])

        selector = by_name["overlay_sel"]
        self.assertEqual("source_switcher", selector["type"])
        self.assertEqual(["no_overlay", "post_overlay"], selector["src"])
        self.assertEqual("video_output", selector["dst"])
        self.assertEqual(0, selector["active"])
        self.assertEqual(0, selector["fallback_active"])
        self.assertTrue(selector["fallback_when_active_missing"])
        self.assertEqual(100, selector["fallback_wait_ms"])

    def test_html_overlay_graph_can_start_enabled_for_docker_url(self):
        avp = FakeAVPlumber()

        out_edge = build_html_overlay_output(
            avp,
            "mixer_final",
            socket_path="/tmp/test.sock",
            start_enabled=True,
        )

        self.assertEqual("video_output", out_edge)
        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        self.assertEqual(2, by_name["otm_html_overlay"]["outputs"])
        self.assertEqual(1, by_name["otm_html_overlay_src"]["outputs"])
        self.assertEqual(1, by_name["overlay_sel"]["active"])
        self.assertNotIn("hwaccel", by_name["html_overlay_src"])

    def test_mux_output_errors_panic_but_clean_finish_stops_by_default(self):
        avp = FakeAVPlumber()

        build_mux_output(
            avp,
            ["video_enc", "audio_enc"],
            mux_edge="program_mux",
            output_url="/tmp/out.ts",
            output_format="mpegts",
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
            janus_video_ssrc=0x41565001,
            janus_audio_bitrate="96k",
            janus_video_pt=96,
            janus_audio_pt=111,
        )

        build_janus_rtp_output(avp, "video_out", "audio_out", args)

        by_name = {node["name"]: node for node in avp.nodes if "name" in node}
        enc_options = by_name["janus_enc_video"]["options"]
        self.assertEqual("1000k", enc_options["b"])
        self.assertEqual("1000k", enc_options["maxrate"])
        self.assertEqual("1000k", enc_options["bufsize"])
        self.assertEqual(30, enc_options["g"])
        self.assertEqual(0, enc_options["bf"])
        self.assertEqual("p6", enc_options["preset"])
        self.assertEqual("baseline", enc_options["profile"])
        self.assertEqual("4.0", enc_options["level"])
        self.assertEqual("ull", enc_options["tune"])
        self.assertEqual("cbr", enc_options["rc"])
        self.assertEqual(0, enc_options["rc-lookahead"])
        self.assertEqual(1, enc_options["forced-idr"])
        self.assertEqual(1, enc_options["aud"])
        self.assertEqual(1, enc_options["zerolatency"])
        self.assertEqual(0, enc_options["delay"])
        self.assertEqual("force_keyframe", by_name["janus_force_keyframe"]["type"])
        self.assertEqual("1/1", by_name["janus_force_keyframe"]["interval_sec"])
        self.assertEqual("bsf", by_name["janus_repeat_headers"]["type"])
        self.assertEqual("dump_extra=freq=keyframe", by_name["janus_repeat_headers"]["bsf"])
        self.assertEqual(
            ["janus_v_repeat_headers"],
            by_name["janus_video_rtp_mux_mux"]["src"],
        )
        self.assertEqual(
            "rtp://localhost:5004?pkt_size=1200&rtcp_port=5005",
            by_name["janus_video_rtp_mux_output"]["url"],
        )
        self.assertEqual(
            {"payload_type": 96, "rtpflags": "skip_rtcp", "ssrc": 0x41565001},
            by_name["janus_video_rtp_mux_output"]["options"],
        )
        self.assertEqual(
            "rtp://localhost:5002?pkt_size=1200&rtcp_port=5003",
            by_name["janus_audio_rtp_mux_output"]["url"],
        )
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
            janus_video_ssrc=0x41565001,
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
