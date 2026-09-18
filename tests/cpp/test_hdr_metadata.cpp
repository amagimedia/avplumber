#include "hdr_metadata.hpp"
#include <cassert>
#include <cstring>

// util.hpp's Error prints a stack trace through this symbol from util.cpp.
void print_stack_trace() {}

int main() {
    const Parameters md = {{"primaries", {{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}}},
                           {"white_point", {0.3127, 0.3290}},
                           {"max_luminance", 1000.0}, {"min_luminance", 0.0001},
                           {"max_cll", 1000}, {"max_fall", 400}};
    AVCodecContext *ctx = avcodec_alloc_context3(nullptr);
    assert(ctx);
    attachHdrMetadata(ctx, md);
    assert(ctx->nb_decoded_side_data == 2);
    const AVFrameSideData *mastering = av_frame_side_data_get(ctx->decoded_side_data, ctx->nb_decoded_side_data,
                                                              AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    const AVFrameSideData *light = av_frame_side_data_get(ctx->decoded_side_data, ctx->nb_decoded_side_data,
                                                          AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    assert(mastering && light);
    size_t mastering_size = 0, light_size = 0;
    av_free(av_mastering_display_metadata_alloc_size(&mastering_size));
    av_free(av_content_light_metadata_alloc(&light_size));
    assert(mastering->size == mastering_size && light->size == light_size);
    const auto *mdm = reinterpret_cast<const AVMasteringDisplayMetadata *>(mastering->data);
    assert(mdm->has_primaries && mdm->has_luminance);
    assert(mdm->display_primaries[0][0].num == 35400 && mdm->display_primaries[0][0].den == 50000);   // 0.708
    assert(mdm->white_point[0].num == 15635 && mdm->white_point[1].num == 16450);                    // D65
    assert(mdm->max_luminance.num == 10000000 && mdm->max_luminance.den == 10000);                   // 1000 nits
    assert(mdm->min_luminance.num == 1);                                                              // 0.0001 nits
    const auto *cll = reinterpret_cast<const AVContentLightMetadata *>(light->data);
    assert(cll->MaxCLL == 1000 && cll->MaxFALL == 400);
    avcodec_free_context(&ctx);

    for (const char *missing : {"primaries", "white_point", "max_cll"}) {
        Parameters bad = md;
        bad.erase(missing);
        AVCodecContext *c = avcodec_alloc_context3(nullptr);
        bool thrown = false;
        try { attachHdrMetadata(c, bad); } catch (const Error &e) { thrown = std::strstr(e.what(), missing) != nullptr; }
        assert(thrown);
        avcodec_free_context(&c);
    }
    Parameters shape = md;
    shape["primaries"] = {{0.7, 0.3}};
    AVCodecContext *c = avcodec_alloc_context3(nullptr);
    bool thrown = false;
    try { attachHdrMetadata(c, shape); } catch (const Error &) { thrown = true; }
    assert(thrown);
    avcodec_free_context(&c);
}
