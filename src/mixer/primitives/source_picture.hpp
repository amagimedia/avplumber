#pragma once
#include "layer_zoom.hpp"

namespace avp::mixer {

// A source can retain its logical tile coordinates while delivering only the original
// camera pixels. The same per-frame maps used by zooms resolve its ordinary layers.
inline void applySourcePictures(std::vector<LayerSpec> &layers,
                                const std::vector<const av::VideoFrame *> &sources,
                                int zoom_input = -1) {
    for (size_t i = 0; i < layers.size(); ++i) {
        auto &layer = layers[i];
        const int input = int(layer.sourceIndex(i));
        if (input == zoom_input || input < 0 || size_t(input) >= sources.size()) continue;
        const auto *frame = sources[input];
        const auto *entry = frame && frame->raw()
            ? av_dict_get(frame->raw()->metadata, "mixer_source_picture_v1", nullptr, 0) : nullptr;
        if (!entry || !entry->value) continue;
        const auto description = Parameters::parse(entry->value);
        const auto &ref = description.at("reference");
        ZoomSpec mapping;
        mapping.input = input;
        mapping.picture = {{ref.at("x"), ref.at("y"), ref.at("w"), ref.at("h")},
                           description.at("metadata_key"), description.value("mapped", std::vector<int>{})};
        mapping.from_crop = mapping.to_crop = {layer.crop_x, layer.crop_y, layer.crop_w, layer.crop_h};
        mapping.from_dst = mapping.to_dst = {layer.dst_x, layer.dst_y, layer.dst_w, layer.dst_h};
        mapping.duration_ns = 1; // no start timestamp: resolves the identical endpoints every frame
        ZoomFrameState state;
        std::string why;
        auto original = layer;
        std::vector<LayerSpec> one{layer};
        one[0].input = input;
        if (applyLayerZoom(one, mapping, frame, 0, state, &why)) {
            layer = one[0];
            layer.input = original.input;
            layer.rotation_deg = original.rotation_deg;
            layer.pivot_set = original.pivot_set;
            layer.pivot_x = original.pivot_x;
            layer.pivot_y = original.pivot_y;
        } else {
            // Logical sheet pixels do not exist on this source: never sample an unrelated region.
            layer.dst_w = layer.dst_h = 0;
        }
    }
}

}
