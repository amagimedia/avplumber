import unittest
from types import SimpleNamespace

from pyplumber.auto_mixer.outputs import (
    build_janus_rtp_output,
    build_mux_output,
)


class FakeAVPlumber:
    def __init__(self):
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node.parameters)


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
        self.assertEqual("p4", enc_options["preset"])
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
