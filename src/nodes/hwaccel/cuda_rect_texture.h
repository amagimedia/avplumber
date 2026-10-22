#pragma once
// Descriptor carried in AVFrame.opaque_ref by zero-copy DMA-BUF imports: the frame's pixels
// live in a CUDA array (the mapped EGL image, tiled GPU layout) reachable only through a
// texture object. data[0] holds textureDataTag(handle) so the frame passes generic validity
// checks; it is NOT a device pointer, and lies far above any GPU virtual address, so a read
// through it faults at a recognisable address instead of near 0. The compositor is the only
// consumer that reads such frames; everything else in the graph just passes references through.
#include <cstdint>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

namespace avp::mixer {

struct TextureFrameDesc {
    uint32_t magic = kMagic;
    uint32_t reserved = 0;
    unsigned long long tex = 0;   // CUtexObject: uchar4 texels, point sampling, unnormalized coordinates
    int width = 0;
    int height = 0;
    static constexpr uint32_t kMagic = 0x58545641u;   // "AVTX"
};
static_assert(sizeof(TextureFrameDesc) == 24, "opaque_ref layout");

/// data[0] of a texture-backed frame: the handle under a tag no GPU virtual address reaches.
constexpr unsigned long long kTextureDataTag = 0xA77E000000000000ull;
inline unsigned long long textureDataTag(unsigned long long tex) { return kTextureDataTag | (tex & 0xFFFFFFFFFFFFull); }
inline bool isTextureDataTag(unsigned long long ptr) { return (ptr & 0xFFFF000000000000ull) == kTextureDataTag; }

/// The descriptor of a texture-backed frame, or nullptr for ordinary device-memory frames.
inline const TextureFrameDesc *textureFrameDesc(const AVFrame *f) {
    if (!f || !f->opaque_ref || f->opaque_ref->size < (int)sizeof(TextureFrameDesc)) return nullptr;
    const auto *d = reinterpret_cast<const TextureFrameDesc *>(f->opaque_ref->data);
    return d->magic == TextureFrameDesc::kMagic ? d : nullptr;
}

}
