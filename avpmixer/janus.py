"""Video-only H.264 RTP output to a Janus Streaming mountpoint, shared by demos."""

from __future__ import annotations

from dataclasses import dataclass

RTP_PACKET_SIZE = 1_200


@dataclass(frozen=True)
class JanusVideoConfig:
    host: str = "127.0.0.1"
    video_port: int = 5004
    payload_type: int = 96
    ssrc: int = 0x41565001
    bitrate_kbps: int = 4_500
    rtcp_bind: str = "0.0.0.0"
    rtcp_port: int = 0

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

    @property
    def rtcp_port_remote(self) -> int:
        return self.video_port + 1

    @property
    def rtp_url(self) -> str:
        return (f"rtp://{self.host}:{self.video_port}?pkt_size={RTP_PACKET_SIZE}"
                f"&rtcp_port={self.rtcp_port_remote}")


JANUS_KEYFRAME_NODE = "janus_force_keyframe"
KEYFRAME_COMMAND = f"node.object.set {JANUS_KEYFRAME_NODE} trigger true"


def build_janus_output(avp, api, src_edge: str, janus: JanusVideoConfig, *, fps: int,
                       width: int, height: int, hwaccel: str = "@gpu", fps_den: int = 1,
                       group: str = "output", profile: str = "baseline", preset: str = "p7",
                       prefix: str = "janus"):
    """Add ``force_fps -> keyframe -> nvenc -> bsf -> rtp mux -> output``; return the RTCP listener."""
    bitrate = f"{janus.bitrate_kbps}k"
    avp.addNode(api.ForceFPS({
        "name": "janus_fps", "src": src_edge, "dst": "janus_fps", "fps": f"{fps}/{fps_den}",
        "group": group,
    }))
    avp.addNode(api.ForceKeyFrame({
        "name": JANUS_KEYFRAME_NODE, "src": "janus_fps", "dst": "janus_keyframed",
        "interval_sec": "1/1", "auto_restart": "panic", "group": group,
    }))
    avp.addNode(api.AssumeVideoFormat({
        "name": "janus_format", "src": "janus_keyframed", "dst": "janus_video",
        "width": width, "height": height, "pixel_format": "cuda", "real_pixel_format": "nv12",
        "auto_restart": "panic", "group": group,
    }))
    avp.addNode(api.EncVideo({
        "name": "janus_encoder", "src": "janus_video", "dst": "janus_encoded",
        "codec": "h264_nvenc", "hwaccel": hwaccel,
        "options": {
            "b": bitrate, "maxrate": bitrate, "bufsize": bitrate, "g": fps, "bf": 0,
            # p7 is NVENC's highest-quality preset; with tune=ull it stays a
            # one-pass, no-lookahead, no-reordering encode, so the extra quality
            # costs GPU time rather than latency. B-frames stay off: they need
            # reordering, and WebRTC negotiates constrained baseline anyway.
            "preset": preset, "profile": profile, "tune": "ull", "rc": "cbr",
            "rc-lookahead": 0, "zerolatency": 1, "delay": 0, "forced-idr": 1,
            "no-scenecut": 1, "strict_gop": 1, "aud": 1, "spatial-aq": 1, "temporal-aq": 0,
        },
        "auto_restart": "panic", "group": group,
    }))
    avp.addNode(api.Bsf({
        "name": "janus_repeat_headers", "src": "janus_encoded", "dst": "janus_repeat_headers",
        "bsf": "dump_extra=freq=keyframe", "auto_restart": "panic", "group": group,
    }))
    avp.addNode(api.Mux({
        "name": "janus_mux", "src": ["janus_repeat_headers"], "dst": "janus_video_rtp_mux",
        "ts_sort_wait": 0, "auto_restart": "on", "on_error": "panic", "group": group,
    }))
    avp.addNode(api.Output({
        "name": "janus_rtp_output", "src": "janus_video_rtp_mux", "url": janus.rtp_url,
        "format": "rtp",
        "options": {"payload_type": janus.payload_type, "rtpflags": "skip_rtcp", "ssrc": janus.ssrc},
        "auto_restart": "on", "on_error": "panic", "group": group,
    }))
    return api.RtcpFeedbackListener(
        bind_host=janus.rtcp_bind, bind_port=janus.rtcp_port, janus_host=janus.host,
        janus_rtcp_port=janus.rtcp_port_remote, media_ssrc=janus.ssrc,
        on_keyframe_request=lambda _request: avp.executeCommandsFromString(KEYFRAME_COMMAND),
    )
