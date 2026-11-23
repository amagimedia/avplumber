#include "nodes/fabric/fabric_wire.hpp"

#include <array>
#include <cassert>
#include <numeric>

using namespace avp_fabric;

int main() {
    // The deployed v1 header ends with packet_format, flags, duration, geometry,
    // pixel formats and wallclock. A misplaced plane shifts every one of them.
    std::array<uint8_t, 92> legacy{};
    std::iota(legacy.begin(), legacy.end(), 1);
    const auto media = readMediaHeader(legacy.data(), VERSION_SINGLE);
    assert(media.plane == PLANE_FILL);
    assert(media.codec == 0x34333231);
    assert(media.packet_format == 0x38373635);
    assert(media.sender_wallclock_ns == 0x5c5b5a5958575655ULL);
    std::array<uint8_t, 92> encoded{};
    writeMediaHeader(encoded.data(), media, VERSION_SINGLE);
    assert(encoded == legacy);

    for (auto plane : {PLANE_FILL, PLANE_ALPHA}) {
        auto key_fill = media;
        key_fill.plane = plane;
        std::array<uint8_t, 96> bytes{};
        writeMediaHeader(bytes.data(), key_fill, VERSION_PLANES);
        assert(bytes[52] == plane && bytes[56] == legacy[52]);
        const auto decoded = readMediaHeader(bytes.data(), VERSION_PLANES);
        assert(std::memcmp(&decoded, &key_fill, sizeof(decoded)) == 0);
    }
    for (auto codec : {CODEC_JPEG, CODEC_H264_INTRA}) {
        assert(wireVersion(codec, false) == VERSION_SINGLE);
        assert(wireVersion(codec, true) == VERSION_PLANES);
    }
    for (auto codec : {CODEC_HEVC_INTRA, CODEC_JPEG2000, CODEC_PRORES}) {
        assert(wireVersion(codec, false) == VERSION_PLANES);
    }
    assert(headerBytes(VERSION_SINGLE) == 116 && headerBytes(VERSION_PLANES) == 120);
    assert(headerBytes(3) == 0);
}
