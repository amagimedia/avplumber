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

    // Opacity is not read from JSON; a partial fade is carried.
    auto faded = parseLayersArray(Parameters::parse(R"([{"input":0,"blend":true,"opacity":0.5},{"input":0,"blend":true}])"));
    assert(faded[0].opacity == 1.f);
    faded[1].opacity = 0.5f;
    const auto fops = resolveDrawOps(sources, faded, 320, 480, AV_PIX_FMT_NV12);
    assert(fops.size() == 2 && fops[1].src == &frame && fops[1].layer.opacity == 0.5f);

    // A canvas that would only repeat the frame: one unblended layer, whole frame, same size.
    auto copies = [&](const char *json, int canvas_w, int canvas_h) {
        const auto one = parseLayersArray(Parameters::parse(json));
        return copiesWholeFrame(resolveDrawOps(sources, one, canvas_w, canvas_h, AV_PIX_FMT_NV12), canvas_w, canvas_h);
    };
    assert(copies(R"([{"input":0,"dst_x":0,"dst_y":0,"dst_w":320,"dst_h":180}])", 320, 180));
    assert(copies(R"([{"input":0}])", 320, 180));
    assert(copies(R"([{"input":0,"dst_x":0,"dst_y":0,"dst_w":320,"dst_h":180,"fit":"contain"}])", 320, 180));
    assert(!copies(R"([{"input":0,"dst_x":0,"dst_y":0,"dst_w":160,"dst_h":90}])", 160, 90));
    assert(!copies(R"([{"input":0,"crop":{"x":0,"y":0,"w":160,"h":90},"dst_x":0,"dst_y":0,"dst_w":320,"dst_h":180}])", 320, 180));
    assert(!copies(R"([{"input":0,"dst_x":2,"dst_y":0}])", 320, 180));
    assert(!copies(R"([{"input":0,"blend":true}])", 320, 180));
    assert(!copies(R"([{"input":0},{"input":0}])", 320, 180));
    // Letterboxed: the frame covers only part of a taller canvas.
    assert(!copies(R"([{"input":0,"dst_x":0,"dst_y":0,"dst_w":320,"dst_h":240,"fit":"contain"}])", 320, 240));

    // Filter settings: a layer that names none takes the parser's default (bilinear; the
    // transform node passes auto), and each setting is read on its own.
    const char *filter_layers = R"([
        {"input":0,"dst_w":640,"dst_h":360},
        {"input":0,"dst_w":640,"dst_h":360,"filter":"bicubic","bicubic_param":0.5},
        {"input":0,"dst_w":80,"dst_h":44,"filter":"multisample","samples":8},
        {"input":0,"dst_w":80,"dst_h":44,"filter":"auto","bicubic_above":3.0,"multisample_above":5.5},
        {"input":0,"dst_w":80,"dst_h":44,"filter":"bilinear"}
    ])";
    const auto plain = parseLayersArray(Parameters::parse(filter_layers));
    const FilterSpec defaults;
    assert(plain[0].filter == defaults && defaults.mode == ScaleFilter::Bilinear);
    assert(defaults.bicubic_param == 0.f && defaults.samples == 4);
    assert(defaults.bicubic_above == 1.3 && defaults.multisample_above == 2.);
    assert(plain[1].filter.mode == ScaleFilter::Bicubic && plain[1].filter.bicubic_param == 0.5f);
    assert(plain[2].filter.mode == ScaleFilter::Multisample && plain[2].filter.samples == 8);
    assert(plain[3].filter.mode == ScaleFilter::Auto && plain[3].filter.bicubic_above == 3. &&
           plain[3].filter.multisample_above == 5.5);
    assert(!(plain[0] == plain[1]) && plain[4].filter.mode == ScaleFilter::Bilinear);
    const auto automatic = parseLayersArray(Parameters::parse(filter_layers), ScaleFilter::Auto);
    assert(automatic[0].filter.mode == ScaleFilter::Auto && automatic[4].filter.mode == ScaleFilter::Bilinear);
    assert(automatic[1].filter.mode == ScaleFilter::Bicubic && automatic[2].filter.mode == ScaleFilter::Multisample);
    // The resolved op decides auto: 320x180 to 640x360 enlarges 2x, to 80x44 shrinks 4x.
    const auto auto_ops = resolveDrawOps(sources, automatic, 640, 360, AV_PIX_FMT_NV12);
    assert(drawFilter(auto_ops[0].layer) == ScaleFilter::Bicubic);
    assert(drawFilter(auto_ops[3].layer) == ScaleFilter::Bilinear);   // its own limit is 5.5
    assert(drawFilter(resolveDrawOps(sources, plain, 640, 360, AV_PIX_FMT_NV12)[0].layer) == ScaleFilter::Bilinear);
    auto shrinking = parseLayersArray(Parameters::parse(R"([{"input":0,"dst_w":80,"dst_h":44}])"), ScaleFilter::Auto);
    assert(drawFilter(resolveDrawOps(sources, shrinking, 640, 360, AV_PIX_FMT_NV12)[0].layer) == ScaleFilter::Multisample);
    assert(describeDrawOps(resolveDrawOps(sources, shrinking, 640, 360, AV_PIX_FMT_NV12)).find("multisample") != std::string::npos);
    assert(describeDrawOps(resolveDrawOps(sources, plain, 640, 360, AV_PIX_FMT_NV12)).find("bilinear") == std::string::npos);
    // Per-frame metadata moves a layer and keeps the filter it does not name.
    auto configured = parseLayersArray(Parameters::parse(R"([{"input":0,"filter":"multisample","samples":8,"dst_w":80,"dst_h":44}])"));
    applyLayerMetadata(configured, R"({"0":{"crop":{"x":16,"y":8,"w":160,"h":90},"dst_w":80,"dst_h":44}})");
    assert(configured[0].crop_x == 16 && configured[0].filter.mode == ScaleFilter::Multisample && configured[0].filter.samples == 8);
    applyLayerMetadata(configured, R"({"0":{"dst_w":80,"dst_h":44,"filter":"bicubic"}})");
    assert(configured[0].filter.mode == ScaleFilter::Bicubic && configured[0].filter.samples == 8);
    // Values outside the settings' ranges are errors.
    for (const char *bad : {R"([{"filter":"lanczos"}])", R"([{"samples":6}])", R"([{"bicubic_above":0.5}])",
                            R"([{"multisample_above":0}])", R"([{"filter":3}])"}) {
        bool thrown = false;
        try { parseLayersArray(Parameters::parse(bad)); } catch (const std::exception &) { thrown = true; }
        assert(thrown);
    }
    // Per-input metadata keeps its source identity when layers omit unused inputs.
    auto sparse = parseLayersArray(Parameters::parse(R"([{"input":3,"dst_w":80,"dst_h":44}])"));
    applyLayerMetadata(sparse, R"({"0":{"dst_x":999},"3":{"dst_x":16}})");
    assert(sparse[0].input == 3 && sparse[0].dst_x == 16);
    applyLayerMetadata(sparse, R"({"layers":[{"dst_x":999},{},{},{"dst_x":24}]})");
    assert(sparse[0].input == 3 && sparse[0].dst_x == 24);

}
