"""Output selection for the DMA-BUF browser scaling graph."""


def resolve_output_config(env, default_rtp_url, rtp_payload_type, rtp_ssrc):
    output_format = env.get("OUTPUT_FORMAT", "rtp").strip().lower()
    output_url = env.get("OUTPUT_URL", default_rtp_url).strip()
    if not output_format:
        raise ValueError("OUTPUT_FORMAT must not be empty")
    if not output_url:
        raise ValueError("OUTPUT_URL must not be empty")

    output_options = {}
    if output_format == "rtp":
        output_options = {
            "payload_type": rtp_payload_type,
            "rtpflags": "skip_rtcp",
            "ssrc": rtp_ssrc,
        }
    return output_format, output_url, output_options
