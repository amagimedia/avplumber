#pragma once
// Static HDR metadata (mastering display + content light level) as AVFrame side
// data on a codec context, so encoders such as hevc_nvenc emit the HDR10 SEIs.
// This only serialises the values it is given; the color numbers live with the
// color contracts in the Python layer (pyplumber.mixer.color), not here.
#include <cmath>
#include "util.hpp"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/mem.h>
}

// hdr_metadata: {"primaries": [[rx,ry],[gx,gy],[bx,by]], "white_point": [wx,wy],
//                "max_luminance": nits, "min_luminance": nits, "max_cll": nits, "max_fall": nits}
inline void attachHdrMetadata(AVCodecContext *ctx, const Parameters &md) {
    // Fixed 1/50000 and 1/10000 grids as the HDR10 SEI requires; av_d2q would pick any denominator.
    auto q = [](double v, int den) { return av_make_q((int)std::lround(v * den), den); };
    auto sideDataSize = [](void *(*alloc)(size_t *)) {
        size_t size = 0;
        av_free(alloc(&size));
        return size;
    };
    for (const char *key : {"primaries", "white_point", "max_luminance", "min_luminance", "max_cll", "max_fall"})
        if (!md.contains(key)) throw Error(std::string("hdr_metadata: missing ") + key);
    const auto &prim = md.at("primaries");
    const auto &white = md.at("white_point");
    if (!prim.is_array() || prim.size() != 3 || !white.is_array() || white.size() != 2)
        throw Error("hdr_metadata: primaries must be three [x, y] pairs and white_point one [x, y] pair");
    AVFrameSideData *sd = av_frame_side_data_new(&ctx->decoded_side_data, &ctx->nb_decoded_side_data,
                                                 AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
                                                 sideDataSize([](size_t *s) -> void * { return av_mastering_display_metadata_alloc_size(s); }), 0);
    if (!sd) throw Error("hdr_metadata: cannot allocate mastering display metadata");
    auto *mdm = reinterpret_cast<AVMasteringDisplayMetadata *>(sd->data);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            mdm->display_primaries[i][j] = q(prim[i][j].get<double>(), 50000);   // chromaticity in 1/50000
    mdm->white_point[0] = q(white[0].get<double>(), 50000);
    mdm->white_point[1] = q(white[1].get<double>(), 50000);
    mdm->max_luminance = q(md.at("max_luminance").get<double>(), 10000);        // nits in 1/10000
    mdm->min_luminance = q(md.at("min_luminance").get<double>(), 10000);
    mdm->has_primaries = mdm->has_luminance = 1;
    sd = av_frame_side_data_new(&ctx->decoded_side_data, &ctx->nb_decoded_side_data,
                                AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
                                sideDataSize([](size_t *s) -> void * { return av_content_light_metadata_alloc(s); }), 0);
    if (!sd) throw Error("hdr_metadata: cannot allocate content light level metadata");
    auto *cll = reinterpret_cast<AVContentLightMetadata *>(sd->data);
    cll->MaxCLL = md.at("max_cll").get<unsigned>();
    cll->MaxFALL = md.at("max_fall").get<unsigned>();
}
