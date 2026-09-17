"""Video-only H.264/HEVC RTP output to a Janus Streaming mountpoint, shared by demos."""

from __future__ import annotations

from dataclasses import dataclass

from .color import TEN_BIT_FORMATS

RTP_PACKET_SIZE = 1_200
DEFAULT_KEYFRAME_MIN_INTERVAL_MS = 150


@dataclass(frozen=True)
class JanusVideoConfig:
    host: str = "127.0.0.1"
    video_port: int = 5004
    payload_type: int = 96
    ssrc: int = 0x41565001
    bitrate_kbps: int = 4_500
    rtcp_bind: str = "0.0.0.0"
    rtcp_port: int = 0
    keyframe_min_interval_ms: int = DEFAULT_KEYFRAME_MIN_INTERVAL_MS

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("Janus host is required")
        if not 1 <= self.video_port < 65535:
            raise ValueError("Janus video port and its RTCP pair must be valid")
        if not 0 <= self.payload_type <= 127:
            raise ValueError("RTP payload type must be between 0 and 127")
        if not 0 <= self.ssrc <= 0xFFFFFFFF:
            raise ValueError("RTP SSRC must be a 32-bit unsigned integer")
        if self.bitrate_kbps <= 0:
            raise ValueError("Janus bitrate must be positive")
        if not 0 <= self.rtcp_port <= 65535:
            raise ValueError("RTCP port must be between 0 and 65535")
        if (type(self.keyframe_min_interval_ms) is not int
                or not 0 <= self.keyframe_min_interval_ms <= 2_147_483_647):
            raise ValueError("keyframe_min_interval_ms must be a non-negative integer")

    @property
    def rtcp_port_remote(self) -> int:
        return self.video_port + 1

    @property
    def rtp_url(self) -> str:
        return (f"rtp://{self.host}:{self.video_port}?pkt_size={RTP_PACKET_SIZE}"
                f"&rtcp_port={self.rtcp_port_remote}")


JANUS_KEYFRAME_NODE = "janus_force_keyframe"
KEYFRAME_COMMAND = f"node.object.set {JANUS_KEYFRAME_NODE} trigger true"


class RtcpFeedbackGroup:
    """Manage feedback for multiple independently encoded renditions."""

    def __init__(self, listeners):
        self.listeners = tuple(listeners)

    def start(self):
        started = []
        try:
            for listener in self.listeners:
                listener.start()
                started.append(listener)
        except Exception:
            for listener in reversed(started):
                listener.stop()
            raise

    def stop(self):
        for listener in reversed(self.listeners):
            listener.stop()


def add_nodes(avp, nodes, **defaults):
    """Register *nodes*, filling in shared parameters (``group``) a node does not set itself."""
    for node in nodes:
        for key, value in defaults.items():
            node.parameters.setdefault(key, value)
        avp.addNode(node)


def build_janus_output(avp, api, src_edge: str, janus: JanusVideoConfig, *, fps: int,
                       width: int, height: int, hwaccel: str = "@gpu", fps_den: int = 1,
                       group: str = "output", codec: str = "", profile: str = "",
                       preset: str = "p7", enc_format: str = "nv12", color=None,
                       hdr_metadata=None, prefix: str = "janus"):
    """Add ``force_fps -> keyframe -> nvenc -> bsf -> rtp mux -> output``; return the RTCP listener.

    Defaults to HEVC (Main/Main10), which current Safari and Chrome negotiate
    over WebRTC on hardware-decode machines: ~2x the efficiency of H.264 at the
    same bitrate and it carries 10-bit + HDR signaling. ``enc_format`` is the
    encoder's CUDA input (``p010le`` keeps 10-bit, ``nv12`` is 8-bit) and
    ``color`` supplies the VUI so an HLG program signals BT.2020/arib-std-b67.
    """
    node_name = lambda suffix: f"{prefix}_{suffix}"
    keyframe_node = node_name("force_keyframe")
    bitrate = f"{janus.bitrate_kbps}k"
    ten_bit = enc_format in TEN_BIT_FORMATS
    if not codec:
        codec = "hevc_nvenc" if ten_bit else "h264_nvenc"   # HEVC only when the input is 10-bit
    if not profile:
        profile = ("main10" if ten_bit else "main") if "hevc" in codec else "baseline"
    add_nodes(avp, [
        api.ForceFPS({"name": node_name("fps"), "src": src_edge, "dst": node_name("fps"),
                      "fps": f"{fps}/{fps_den}"}),
        api.ForceKeyFrame({"name": keyframe_node, "src": node_name("fps"), "dst": node_name("keyframed"),
                           "interval_sec": "1/1", "min_interval_ms": janus.keyframe_min_interval_ms}),
        api.AssumeVideoFormat({"name": node_name("format"), "src": node_name("keyframed"), "dst": node_name("video"),
                               "width": width, "height": height, "pixel_format": "cuda",
                               "real_pixel_format": enc_format}),
        api.EncVideo({
            "name": node_name("encoder"), "src": node_name("video"), "dst": node_name("encoded"),
            "codec": codec, "hwaccel": hwaccel,
            **({"hdr_metadata": hdr_metadata} if hdr_metadata else {}),
            "options": {
                "b": bitrate, "maxrate": bitrate, "bufsize": bitrate, "g": max(1, round(fps / fps_den)), "bf": 0,
                # p5..p7 are NVENC's quality presets; with tune=ull it stays a
                # one-pass, no-lookahead, no-reordering encode, so the extra quality
                # costs GPU time rather than latency. B-frames stay off: they need
                # reordering, which WebRTC's jitter budget will not absorb.
                "preset": preset, "profile": profile, "tune": "ull", "rc": "cbr",
                "rc-lookahead": 0, "zerolatency": 1, "delay": 0, "forced-idr": 1,
                "no-scenecut": 1, "strict_gop": 1, "aud": 1, "spatial-aq": 1, "temporal-aq": 0,
                **(color or {}),
            },
        }),
        api.Bsf({"name": node_name("repeat_headers"), "src": node_name("encoded"), "dst": node_name("repeat_headers"),
                 "bsf": "dump_extra=freq=keyframe"}),
        api.Mux({"name": node_name("mux"), "src": [node_name("repeat_headers")], "dst": node_name("video_rtp_mux"),
                 "ts_sort_wait": 0, "auto_restart": "on", "on_error": "panic"}),
        api.Output({"name": node_name("rtp_output"), "src": node_name("video_rtp_mux"), "url": janus.rtp_url,
                    "format": "rtp", "auto_restart": "on", "on_error": "panic",
                    "options": {"payload_type": janus.payload_type, "rtpflags": "skip_rtcp", "ssrc": janus.ssrc}}),
    ], group=group, auto_restart="panic")
    return api.RtcpFeedbackListener(
        bind_host=janus.rtcp_bind, bind_port=janus.rtcp_port, janus_host=janus.host,
        janus_rtcp_port=janus.rtcp_port_remote, media_ssrc=janus.ssrc,
        on_keyframe_request=lambda _request: avp.executeCommandsFromString(f"node.object.set {keyframe_node} trigger true"),
    )
