#include "mixer/primitives/compositor_layers.hpp"
#include <cassert>
#include <limits>

void print_stack_trace() {}

int main() {
    using namespace avp::mixer;
    av::VideoFrame frame(AV_PIX_FMT_NV12, 320, 192);
    auto *raw = frame.raw();
    raw->crop_left = raw->crop_right = 8;
    raw->crop_top = 4;
    raw->crop_bottom = 8;
    const auto resolve = [&](const char *json, int w = 608, int h = 360) {
        return resolveDrawOps({&frame}, parseLayersArray(Parameters::parse(json)), w, h, AV_PIX_FMT_NV12);
    };
    auto native = resolve(R"([{}])");
    assert(native[0].layer.crop_x == 8 && native[0].layer.crop_y == 4);
    assert(native[0].layer.crop_w == 304 && native[0].layer.crop_h == 180);
    assert(!copiesWholeFrame(native, 304, 180));

    // The application crop starts inside the decoder's visible picture, not the coded surface.
    auto cropped = resolve(R"([{"crop":{"x":20,"y":10,"w":100,"h":60}}])");
    assert(cropped[0].layer.crop_x == 28 && cropped[0].layer.crop_y == 14);
    assert(cropped[0].layer.crop_w == 100 && cropped[0].layer.crop_h == 60);
    auto remaining = resolve(R"([{"crop":{"x":20,"y":10}}])");
    assert(remaining[0].layer.crop_w == 284 && remaining[0].layer.crop_h == 170);
    auto clipped = resolve(R"([{"crop":{"x":300,"y":176,"w":100,"h":60}}])");
    assert(clipped[0].layer.crop_w == 4 && clipped[0].layer.crop_h == 4);
    assert(!resolve(R"([{"crop":{"x":304,"y":0}}])")[0].src);

    // Contain uses the visible aspect ratio, with no bars when it matches the canvas.
    auto scaled = resolve(R"([{"dst_w":608,"dst_h":360,"fit":"contain"}])");
    assert(scaled[0].layer.dst_x == 0 && scaled[0].layer.dst_y == 0);
    assert(scaled[0].layer.dst_w == 608 && scaled[0].layer.dst_h == 360);
    assert(scaled[0].layer.crop_x == 8 && scaled[0].layer.crop_y == 4);
    assert(scaled[0].layer.crop_w == 304 && scaled[0].layer.crop_h == 180);
    // Tile clipping happens in visible-picture coordinates before adding decoder offsets.
    auto tiled = resolve(R"([{"dst_x":-152,"dst_w":304,"dst_h":180,
        "tile":{"x":0,"y":0,"w":304,"h":180},"scene_canvas":{"w":304,"h":180}}])");
    assert(tiled[0].layer.crop_x == 160 && tiled[0].layer.crop_y == 4);
    assert(tiled[0].layer.crop_w == 152 && tiled[0].layer.dst_w == 152);

    // Hardware frame formats have no chroma geometry themselves; use their sw_format.
    raw->hw_frames_ctx = av_buffer_allocz(sizeof(AVHWFramesContext));
    assert(raw->hw_frames_ctx);
    auto *hw = reinterpret_cast<AVHWFramesContext *>(raw->hw_frames_ctx->data);
    hw->sw_format = AV_PIX_FMT_NV12;
    for (const char *name : {"cuda", "cuarray"}) {
        const auto format = av_get_pix_fmt(name);
        if (format == AV_PIX_FMT_NONE) continue; // CUarray requires experimental FFmpeg 9.
        raw->format = format;
        assert(resolve(R"([{}])")[0].layer == native[0].layer);
    }
    raw->format = AV_PIX_FMT_CUDA;
    // FFmpeg already subtracts right/bottom and clears those fields for hardware frames.
    assert(av_frame_apply_cropping(raw, 0) == 0);
    assert(raw->width == 312 && raw->height == 184);
    assert(raw->crop_left == 8 && raw->crop_top == 4);
    assert(resolve(R"([{}])")[0].layer == native[0].layer);

    // An origin off the chroma grid keeps luma exact; plane rounding shifts chroma half a sample.
    raw->crop_left = 1;
    assert(resolve(R"([{}])")[0].layer.crop_x == 1);
    raw->crop_left = 8;
    raw->crop_top = 1;
    assert(resolve(R"([{}])")[0].layer.crop_y == 1);
    raw->crop_top = 4;
    // A crop that leaves nothing is ignored, as libavcodec ignores such a decoder crop: the
    // whole frame is drawn and the op description says so.
    const auto ignored = [&] {
        const auto ops = resolve(R"([{}])");
        assert(ops[0].src);
        assert(ops[0].layer.crop_x == 0 && ops[0].layer.crop_y == 0);
        assert(ops[0].layer.crop_w == raw->width && ops[0].layer.crop_h == raw->height);
        return describeDrawOps(ops).find("invalid frame crop ignored") != std::string::npos;
    };
    assert(describeDrawOps(resolve(R"([{}])")).find("ignored") == std::string::npos);
    raw->crop_left = std::numeric_limits<size_t>::max();
    assert(ignored());
    raw->crop_left = raw->width;
    assert(ignored());
    raw->crop_left = raw->crop_top = 0;
    raw->crop_right = std::numeric_limits<size_t>::max();
    assert(ignored());
    raw->crop_right = 0;
    raw->crop_bottom = raw->height;
    assert(ignored());
    raw->crop_right = raw->crop_bottom = 0;
    raw->width = 1920;
    raw->height = 1080; // Already-applied 1088 -> 1080 bottom crop stays 1080.
    const auto unchanged = resolve(R"([{}])", 1920, 1080);
    assert(unchanged[0].layer.crop_w == 1920 && unchanged[0].layer.crop_h == 1080);
    assert(copiesWholeFrame(unchanged, 1920, 1080));

    // The layer log reports what is drawn, not where: per-frame metadata moving a crop keeps the
    // shape; another source size, order, layer count, filter, blend or a layer left out changes it.
    {
        av::VideoFrame small(AV_PIX_FMT_NV12, 320, 192), large(AV_PIX_FMT_NV12, 640, 360);
        const auto shape = [](const char *json, const av::VideoFrame &source) {
            return drawOpsShape(resolveDrawOps({&source}, parseLayersArray(Parameters::parse(json)), 608, 360,
                                               AV_PIX_FMT_NV12));
        };
        const auto still = shape(R"([{"crop":{"x":20,"y":10,"w":100,"h":60}}])", small);
        assert(still.size() == 1);
        assert(still == shape(R"([{"crop":{"x":22,"y":12,"w":96,"h":58},"dst_x":4,"dst_w":200,"dst_h":120}])", small));
        assert(still != shape(R"([{"crop":{"x":20,"y":10,"w":100,"h":60}}])", large));
        assert(still != shape(R"([{"z":1}])", small));
        assert(still != shape(R"([{},{"input":0}])", small));
        assert(still != shape(R"([{"filter":"bicubic"}])", small));
        assert(still != shape(R"([{"blend":true}])", small));
        assert(still != shape(R"([{"crop":{"x":400}}])", small));
    }
}
