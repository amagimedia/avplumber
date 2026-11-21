#pragma once

#include "../node_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace avp_fabric {

inline constexpr uint32_t MAGIC = 0x46505641;
// The layout of WireHeader and MediaHeader below. Adding a codec code or a message type does not
// change it; bump this only when an existing field moves, changes width, or changes meaning, and
// expect every peer to be rebuilt.
inline constexpr uint16_t VERSION = 1;
inline constexpr uint16_t MSG_MEDIA = 1;
inline constexpr uint16_t MSG_FRAME_STATUS = 2;
inline constexpr uint32_t FLAG_KEYFRAME = 1u << 0;
inline constexpr uint32_t CODEC_H264_INTRA = 1;
inline constexpr uint32_t CODEC_JPEG = 2;
inline constexpr uint32_t CODEC_HEVC_INTRA = 3;
inline constexpr uint32_t CODEC_JPEG2000 = 4;

struct __attribute__((packed)) WireHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint16_t message_type;
    uint16_t flags;
    uint32_t header_crc;
    uint32_t payload_crc;
    uint32_t payload_bytes;
};

struct __attribute__((packed)) MediaHeader {
    uint64_t stream_id_hash;
    uint32_t replica_id;
    uint64_t generation;
    int64_t pts;
    int64_t dts;
    int32_t time_base_num;
    int32_t time_base_den;
    uint32_t media_type;
    uint32_t codec;
    uint32_t packet_format;
    uint32_t packet_flags;
    int64_t duration;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t real_pixel_format;
    uint64_t sender_wallclock_ns;
};

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

inline AVCodecID codecIdFromName(const std::string &codec) {
    if (codec == "h264_intra") return AV_CODEC_ID_H264;
    if (isHevcCodecName(codec)) return AV_CODEC_ID_HEVC;
    if (isJpegCodecName(codec)) return AV_CODEC_ID_MJPEG;
    if (isJpeg2000CodecName(codec)) return AV_CODEC_ID_JPEG2000;
    throw Error("fabric unsupported codec: " + codec);
}

/// The wire code a `codec` parameter value is sent as. Kept beside codecIdFromName() so a codec
/// cannot be taught to the sender and missed in the receiver's name table.
inline uint32_t codecToWire(const std::string &codec) {
    if (codec == "h264_intra") return CODEC_H264_INTRA;
    if (isHevcCodecName(codec)) return CODEC_HEVC_INTRA;
    if (isJpegCodecName(codec)) return CODEC_JPEG;
    if (isJpeg2000CodecName(codec)) return CODEC_JPEG2000;
    throw Error("fabric unsupported codec: " + codec);
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
    throw Error(node_name + " received wire codec " + std::to_string(codec) +
                ", which this build does not know (highest known is " +
                std::to_string(CODEC_JPEG2000) + "): the producer is newer than this consumer.");
}

inline void validateWireRecord(const WireHeader &wire, size_t len, size_t max_payload_bytes, const std::string &node_name) {
    if (wire.magic != MAGIC) throw Error(node_name + " bad magic");
    if (wire.version != VERSION) {
        throw Error(node_name + " unsupported wire version " + std::to_string(wire.version) +
                    ", this build speaks " + std::to_string(VERSION));
    }
    if (wire.message_type != MSG_MEDIA && wire.message_type != MSG_FRAME_STATUS) {
        throw Error(node_name + " unsupported message_type");
    }
    if (wire.header_bytes != sizeof(WireHeader) + sizeof(MediaHeader)) {
        throw Error(node_name + " unsupported header size");
    }
    if (wire.payload_bytes > max_payload_bytes) throw Error(node_name + " payload exceeds max_payload_bytes");
    if (len != wire.header_bytes + wire.payload_bytes) throw Error(node_name + " message length mismatch");
}

} // namespace avp_fabric
