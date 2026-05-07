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
inline constexpr uint16_t MSG_MEDIA = 1;
inline constexpr uint16_t MSG_FRAME_STATUS = 2;
inline constexpr uint32_t FLAG_KEYFRAME = 1u << 0;
inline constexpr uint32_t CODEC_H264_INTRA = 1;
inline constexpr uint32_t CODEC_JPEG = 2;

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

inline AVCodecID codecIdFromName(const std::string &codec) {
    if (codec == "h264_intra") return AV_CODEC_ID_H264;
    if (codec == "jpeg" || codec == "mjpeg" || codec == "nvjpeg") return AV_CODEC_ID_MJPEG;
    throw Error("fabric unsupported codec: " + codec);
}

inline AVCodecID codecIdFromWire(uint32_t codec) {
    if (codec == CODEC_H264_INTRA) return AV_CODEC_ID_H264;
    if (codec == CODEC_JPEG) return AV_CODEC_ID_MJPEG;
    return AV_CODEC_ID_NONE;
}

inline void validateWireRecord(const WireHeader &wire, size_t len, size_t max_payload_bytes, const std::string &node_name) {
    if (wire.magic != MAGIC) throw Error(node_name + " bad magic");
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
