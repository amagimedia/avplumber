#pragma once
// Descriptor carried in AVFrame.opaque_ref by zero-copy DMA-BUF imports: the frame's pixels
// live in a CUDA array (the mapped EGL image, tiled GPU layout) reachable only through a
// texture object. data[0] holds the texture handle so the frame passes generic validity
// checks, but it is NOT a device pointer. The compositor is the only consumer that reads
// such frames; everything else in the graph just passes references through.
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

/// The descriptor of a texture-backed frame, or nullptr for ordinary device-memory frames.
inline const TextureFrameDesc *textureFrameDesc(const AVFrame *f) {
    if (!f || !f->opaque_ref || f->opaque_ref->size < (int)sizeof(TextureFrameDesc)) return nullptr;
    const auto *d = reinterpret_cast<const TextureFrameDesc *>(f->opaque_ref->data);
    return d->magic == TextureFrameDesc::kMagic ? d : nullptr;
}

}
