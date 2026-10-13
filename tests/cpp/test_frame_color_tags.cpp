#include "frame_color_tags.hpp"
#include <cassert>

void print_stack_trace() {}

static bool rejected(const Parameters &tags) {
    try { FrameColorTags::parse(tags); }
    catch (const std::exception &) { return true; }
    return false;
}

int main() {
    AVFrame *frame = av_frame_alloc();
    assert(frame->color_trc == AVCOL_TRC_UNSPECIFIED && frame->alpha_mode == AVALPHA_MODE_UNSPECIFIED);

    // What the mixer's SDR sources declare: the four fields a setparams filter would write.
    const auto sdr = FrameColorTags::parse({{"color_trc", "bt709"}, {"color_primaries", "bt709"},
                                            {"colorspace", "bt709"}, {"color_range", "tv"}});
    assert(sdr.any());
    sdr.apply(frame);
    assert(frame->color_trc == AVCOL_TRC_BT709 && frame->color_primaries == AVCOL_PRI_BT709);
    assert(frame->colorspace == AVCOL_SPC_BT709 && frame->color_range == AVCOL_RANGE_MPEG);
    assert(frame->alpha_mode == AVALPHA_MODE_UNSPECIFIED);   // a field not named is left alone

    // An HLG source, and a browser page whose alpha is premultiplied: stamping overrides what was there.
    FrameColorTags::parse({{"color_trc", "arib-std-b67"}, {"color_primaries", "bt2020"},
                           {"colorspace", "bt2020nc"}, {"alpha_mode", "premultiplied"}}).apply(frame);
    assert(frame->color_trc == AVCOL_TRC_ARIB_STD_B67 && frame->color_primaries == AVCOL_PRI_BT2020);
    assert(frame->colorspace == AVCOL_SPC_BT2020_NCL && frame->color_range == AVCOL_RANGE_MPEG);
    assert(frame->alpha_mode == AVALPHA_MODE_PREMULTIPLIED);
    FrameColorTags::parse({{"alpha_mode", "straight"}}).apply(frame);
    assert(frame->alpha_mode == AVALPHA_MODE_STRAIGHT && frame->color_trc == AVCOL_TRC_ARIB_STD_B67);

    assert(!FrameColorTags::parse(Parameters::object()).any());
    assert(rejected({{"color_trc", "rec709-ish"}}));      // a name FFmpeg does not know
    assert(rejected({{"transfer", "bt709"}}));            // a field this does not write
    assert(rejected({{"alpha_mode", "associated"}}));
    assert(rejected({{"color_range", 1}}));               // names, not numbers
    assert(rejected(Parameters::array()));
    av_frame_free(&frame);
    return 0;
}
