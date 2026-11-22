#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace avp_fabric {

inline constexpr uint32_t MAGIC = 0x46505641;
inline constexpr uint16_t VERSION_SINGLE = 1;
inline constexpr uint16_t VERSION_PLANES = 2;
inline constexpr uint16_t MSG_MEDIA = 1;
inline constexpr uint16_t MSG_FRAME_STATUS = 2;
inline constexpr uint32_t FLAG_KEYFRAME = 1u << 0;
inline constexpr uint32_t CODEC_H264_INTRA = 1;
inline constexpr uint32_t CODEC_JPEG = 2;
inline constexpr uint32_t CODEC_HEVC_INTRA = 3;
inline constexpr uint32_t CODEC_JPEG2000 = 4;
inline constexpr uint32_t CODEC_PRORES = 5;
// FFV1 is carried only as bitstream version 1 (`level: 1` in the encoder options). From version 2
// on - libavcodec's default - the configuration record (coder, quantisation tables, bit depth)
// lives in AVCodecContext.extradata instead of the frames, and this wire carries media packets
// only: a consumer would reject every frame with "read_quant_table error". Version 1 repeats the
// record in-band on every keyframe, which every fabric packet already is.
inline constexpr uint32_t CODEC_FFV1 = 6;

// Key/fill transport: one frame travels as two media records, the fill (the picture) and the alpha
// (its key). Both carry the same stream identity, PTS, timebase and geometry, so a consumer pairs
// them by PTS; nothing in the wire pairs them, and a consumer that never configured an alpha path
// fails on the first alpha record instead of decoding it as a picture.
inline constexpr uint32_t PLANE_FILL = 0;
inline constexpr uint32_t PLANE_ALPHA = 1;

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
    uint32_t plane;
    uint32_t packet_format;
    uint32_t packet_flags;
    int64_t duration;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t real_pixel_format;
    uint64_t sender_wallclock_ns;
};

// Version 1 has the same packed fields except for the four-byte plane tag.
constexpr size_t v1MediaBytes = sizeof(MediaHeader) - sizeof(MediaHeader::plane);
constexpr size_t planeOffset = offsetof(MediaHeader, plane);
static_assert(sizeof(WireHeader) + v1MediaBytes == 116);
static_assert(sizeof(WireHeader) + sizeof(MediaHeader) == 120);

inline uint16_t wireVersion(uint32_t codec, bool key_fill) {
    // A v1 reader cannot reject unknown codec IDs. Use v2 for those records
    // so legacy receivers reject the header before handing off the payload.
    return !key_fill && (codec == CODEC_H264_INTRA || codec == CODEC_JPEG)
        ? VERSION_SINGLE : VERSION_PLANES;
}

inline size_t headerBytes(uint16_t version) {
    if (version == VERSION_SINGLE) return sizeof(WireHeader) + v1MediaBytes;
    if (version == VERSION_PLANES) return sizeof(WireHeader) + sizeof(MediaHeader);
    return 0;
}

inline void writeMediaHeader(uint8_t *dst, const MediaHeader &media, uint16_t version) {
    if (version == VERSION_SINGLE) {
        std::memcpy(dst, &media, planeOffset);
        std::memcpy(dst + planeOffset, &media.packet_format, v1MediaBytes - planeOffset);
    } else {
        std::memcpy(dst, &media, sizeof(media));
    }
}

// Call after validating the version and header length.
inline MediaHeader readMediaHeader(const uint8_t *src, uint16_t version) {
    MediaHeader media{};
    if (version == VERSION_SINGLE) {
        std::memcpy(&media, src, planeOffset);
        std::memcpy(&media.packet_format, src + planeOffset, v1MediaBytes - planeOffset);
        media.plane = PLANE_FILL;
    } else {
        std::memcpy(&media, src, sizeof(media));
    }
    return media;
}

} // namespace avp_fabric
