#pragma once
// A compositor's debug overlay: what the models say about the picture a layer shows, drawn on the
// composed frame at the size the layer is shown. The compositor knows this interface only; the
// drawing (nodes/neural_net/draw/face_debug_overlay.cpp) is built with the neural-net nodes and
// registers its factory at load, so a build without them composes exactly as before.
#include <cstddef>
#include <memory>
#include <vector>

struct AVFrame;

namespace avp::mixer {

/// One drawn layer that asks for the overlay (`"debug": true`): the source frame, whose metadata
/// says what to draw, the rectangle of its storage the layer shows and where on the canvas.
struct DebugOverlayTile {
    const AVFrame *source = nullptr;
    double crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
    int dst_x = 0, dst_y = 0, dst_w = 0, dst_h = 0;
};

class DebugOverlay {
public:
    virtual ~DebugOverlay() = default;
    /// Draws every tile's overlay on the NV12 CUDA frame `canvas` with one kernel launch queued on
    /// `stream` (a CUstream) under the current CUDA context, after what the stream already holds.
    /// It does not wait: the caller's own completion (event or stream) covers it, and the next
    /// call must not come before that. Throws on a CUDA failure.
    virtual void draw(void *stream, AVFrame *canvas, const std::vector<DebugOverlayTile> &tiles) = 0;
    /// Frees the device memory and the kernel module under the current CUDA context, with no draw
    /// in flight. The overlay can be used again afterwards.
    virtual void release() = 0;
    /// The items and 16x16 tiles of the last draw, for the node's status.
    virtual size_t lastItems() const = 0;
    virtual size_t lastTiles() const = 0;
};

using DebugOverlayFactory = std::unique_ptr<DebugOverlay> (*)();

/// The factory the drawing code registered, null in a build without it.
inline DebugOverlayFactory &debugOverlayFactory() {
    static DebugOverlayFactory factory = nullptr;
    return factory;
}

}   // namespace avp::mixer
