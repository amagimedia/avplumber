#include "mixer/primitives/compositor_layers.hpp"
#include <cassert>

void print_stack_trace() {}

int main() {
    using namespace avp::mixer;
    av::VideoFrame frame(AV_PIX_FMT_NV12, 320, 180);
    const std::vector<const av::VideoFrame *> sources{&frame};
    const auto layers = parseLayersArray(Parameters::parse(R"([
        {"input":0,"dst_x":-40,"dst_y":0,"dst_w":200,"dst_h":240,
         "tile":{"x":80,"y":240,"w":80,"h":120},"scene_canvas":{"w":320,"h":480}},
        {"input":0,"dst_x":160,"dst_y":240,"dst_w":160,"dst_h":240,"blend":true,
         "tile":{"x":80,"y":240,"w":80,"h":120},"scene_canvas":{"w":320,"h":480}},
        {"input":0,"dst_x":0,"dst_y":0,"dst_w":320,"dst_h":480,"fit":"contain",
         "tile":{"x":0,"y":0,"w":160,"h":240},"scene_canvas":{"w":320,"h":480}}
    ])"));
    const auto ops = resolveDrawOps(sources, layers, 320, 480, AV_PIX_FMT_NV12);
    assert(ops.size() == 3);
    for (const auto &op : ops) assert(op.src == &frame);
    // An off-canvas layer is cropped proportionally, without bleeding into its neighbour.
    assert(ops[0].layer.dst_x == 80 && ops[0].layer.dst_w == 40);
    assert(ops[0].layer.crop_x == 64 && ops[0].layer.crop_w == 256);
    assert(ops[1].layer.dst_x == 120 && ops[1].layer.dst_y == 300);
    assert(ops[1].layer.dst_w == 40 && ops[1].layer.dst_h == 60 && ops[1].layer.blend);
    // Fit is resolved against the original scene before shrinking into the tile.
    assert(ops[2].layer.dst_w == 160 && ops[2].layer.dst_h == 90);
    assert(ops[2].layer.dst_x == 0 && ops[2].layer.dst_y == 74);
}
