#pragma once

#include "../node_common.hpp"
#include "fabric_wire.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace avp_fabric {

inline uint64_t monotonicNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::string hexPrefix(const uint8_t *data, size_t size, size_t max_bytes = 24) {
    std::ostringstream os;
    const size_t n = std::min(size, max_bytes);
    for (size_t i = 0; i < n; ++i) {
        if (i) os << ' ';
        os << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(data[i]);
    }
    return os.str();
}

inline const char *planeName(uint32_t plane) {
    if (plane == PLANE_FILL) return "fill";
    if (plane == PLANE_ALPHA) return "alpha";
    return "unknown";
}

/// Throws on a plane this build does not know rather than routing it somewhere: a record from a
/// newer producer, or a corrupt header, would otherwise land on the fill output and be decoded as
/// a picture.
inline void validatePlane(uint32_t plane, const std::string &node_name) {
    if (plane != PLANE_FILL && plane != PLANE_ALPHA) {
        throw Error(node_name + " received wire plane " + std::to_string(plane) +
                    ", which this build does not know (known: " + std::to_string(PLANE_FILL) +
                    " fill, " + std::to_string(PLANE_ALPHA) + " alpha): the producer is newer " +
                    "than this consumer.");
    }
}

// jpeg444/nvjpeg444 are labels for a JPEG bitstream whose SOF carries 4:4:4 sampling.
// The wire codec stays CODEC_JPEG because the JPEG headers describe the stream itself.
// Lossless JPEG is deliberately absent: CODEC_JPEG cannot distinguish a SOF3 lossless codestream
// from a baseline one, so a consumer would hand it to an MJPEG decoder that cannot parse it. It
// needs a wire code and an AV_CODEC_ID_LJPEG mapping of its own before a name can accept it.
inline bool isJpegCodecName(const std::string &codec) {
    return codec == "jpeg" || codec == "mjpeg" || codec == "nvjpeg" ||
           codec == "jpeg444" || codec == "nvjpeg444";
}

inline bool isHevcCodecName(const std::string &codec) {
    return codec == "hevc_intra" || codec == "hevc_lossless" || codec == "h265_intra";
}

inline bool isJpeg2000CodecName(const std::string &codec) {
    return codec == "jpeg2000" || codec == "j2k" || codec == "nvjpeg2k";
}

// One wire code for every ProRes profile: the frame header carries the profile, so the fill and the
// 4444 alpha profiles are told apart by the payload, and the pixel format travels in the media
// header. The names exist so a graph reads like the profile it encodes.
inline bool isProresCodecName(const std::string &codec) {
    return codec == "prores" || codec == "prores4444" || codec == "prores_4444";
}

inline AVCodecID codecIdFromName(const std::string &codec) {
    if (codec == "h264_intra") return AV_CODEC_ID_H264;
    if (isHevcCodecName(codec)) return AV_CODEC_ID_HEVC;
    if (isJpegCodecName(codec)) return AV_CODEC_ID_MJPEG;
    if (isJpeg2000CodecName(codec)) return AV_CODEC_ID_JPEG2000;
    if (isProresCodecName(codec)) return AV_CODEC_ID_PRORES;
    throw Error("fabric unsupported codec: " + codec);
}

inline uint32_t codecToWire(AVCodecID codec) {
    switch (codec) {
        case AV_CODEC_ID_H264: return CODEC_H264_INTRA;
        case AV_CODEC_ID_HEVC: return CODEC_HEVC_INTRA;
        case AV_CODEC_ID_MJPEG: return CODEC_JPEG;
        case AV_CODEC_ID_JPEG2000: return CODEC_JPEG2000;
        case AV_CODEC_ID_PRORES: return CODEC_PRORES;
        default: throw Error(std::string("fabric unsupported upstream codec: ") + avcodec_get_name(codec));
    }
}

/// The libavcodec id to decode a wire codec code with.
///
/// Throws on a code this build does not know rather than reporting AV_CODEC_ID_NONE: the caller
/// would then be left with its configured default, and a newer producer's HEVC or JPEG 2000
/// payload would go to an H.264 decoder as garbage with nothing in the log to say why.
inline AVCodecID codecIdFromWire(uint32_t codec, const std::string &node_name) {
    if (codec == CODEC_H264_INTRA) return AV_CODEC_ID_H264;
    if (codec == CODEC_HEVC_INTRA) return AV_CODEC_ID_HEVC;
    if (codec == CODEC_JPEG) return AV_CODEC_ID_MJPEG;
    if (codec == CODEC_JPEG2000) return AV_CODEC_ID_JPEG2000;
    if (codec == CODEC_PRORES) return AV_CODEC_ID_PRORES;
    throw Error(node_name + " received wire codec " + std::to_string(codec) +
                ", which this build does not know (highest known is " +
                std::to_string(CODEC_PRORES) + "): the producer is newer than this consumer.");
}

inline void validateWireRecord(const WireHeader &wire, size_t len, size_t max_payload_bytes, const std::string &node_name) {
    if (wire.magic != MAGIC) throw Error(node_name + " bad magic");
    if (!headerBytes(wire.version)) {
        throw Error(node_name + " unsupported wire version " + std::to_string(wire.version));
    }
    if (wire.message_type != MSG_MEDIA && wire.message_type != MSG_FRAME_STATUS) {
        throw Error(node_name + " unsupported message_type");
    }
    if (wire.header_bytes != headerBytes(wire.version)) {
        throw Error(node_name + " unsupported header size");
    }
    if (wire.payload_bytes > max_payload_bytes) throw Error(node_name + " payload exceeds max_payload_bytes");
    if (len < wire.header_bytes || len - wire.header_bytes != wire.payload_bytes) throw Error(node_name + " message length mismatch");
}

} // namespace avp_fabric
