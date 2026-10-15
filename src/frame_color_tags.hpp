#pragma once
#include <string>
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}
#include "util.hpp"

/// Colour fields a node writes on every video frame it passes, for a source whose colour is
/// declared: what a `setparams` filter would write, without a filter graph and its thread.
/// `{"color_trc": "bt709", "color_primaries": "bt709", "colorspace": "bt709", "color_range": "tv",
/// "alpha_mode": "premultiplied"}`; any subset, names as FFmpeg spells them.
struct FrameColorTags {
    int trc = -1, primaries = -1, space = -1, range = -1, alpha = -1;

    bool any() const { return trc >= 0 || primaries >= 0 || space >= 0 || range >= 0 || alpha >= 0; }

    static FrameColorTags parse(const Parameters &tags) {
        if (!tags.is_object()) throw Error("frame_color: an object of colour field names is required");
        FrameColorTags result;
        for (auto it = tags.begin(); it != tags.end(); ++it) {
            if (!it.value().is_string()) throw Error("frame_color: " + it.key() + " must be a name");
            const std::string name = it.value().get<std::string>();
            int value;
            if (it.key() == "color_trc") value = result.trc = av_color_transfer_from_name(name.c_str());
            else if (it.key() == "color_primaries") value = result.primaries = av_color_primaries_from_name(name.c_str());
            else if (it.key() == "colorspace") value = result.space = av_color_space_from_name(name.c_str());
            else if (it.key() == "color_range") value = result.range = av_color_range_from_name(name.c_str());
            else if (it.key() == "alpha_mode")
                value = result.alpha = name == "premultiplied" ? AVALPHA_MODE_PREMULTIPLIED
                                     : name == "straight" ? AVALPHA_MODE_STRAIGHT : -1;
            else throw Error("frame_color: unknown field " + it.key());
            if (value < 0) throw Error("frame_color: unknown " + it.key() + " " + name);
        }
        return result;
    }

    void apply(AVFrame *frame) const {
        if (trc >= 0) frame->color_trc = static_cast<AVColorTransferCharacteristic>(trc);
        if (primaries >= 0) frame->color_primaries = static_cast<AVColorPrimaries>(primaries);
        if (space >= 0) frame->colorspace = static_cast<AVColorSpace>(space);
        if (range >= 0) frame->color_range = static_cast<AVColorRange>(range);
        if (alpha >= 0) frame->alpha_mode = static_cast<AVAlphaMode>(alpha);
    }
};
