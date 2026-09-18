#include "mixer/primitives/pixel_layout.hpp"
#include <cassert>
using namespace avp::mixer;

int main() {
    assert(av_pix_fmt_count_planes(AV_PIX_FMT_NV12) == 2 && av_pix_fmt_count_planes(AV_PIX_FMT_P210) == 2);
    assert(av_pix_fmt_count_planes(AV_PIX_FMT_YUV420P) == 3 && av_pix_fmt_count_planes(AV_PIX_FMT_BGRA) == 1);
    assert(sampleBytes(AV_PIX_FMT_NV12) == 1 && sampleBytes(AV_PIX_FMT_P010) == 2);
    assert(storageShift(AV_PIX_FMT_P010) == 6 && storageShift(AV_PIX_FMT_NV12) == 0);
    assert(chromaXAlign(AV_PIX_FMT_NV12) == 2 && chromaYAlign(AV_PIX_FMT_NV12) == 2);
    assert(chromaXAlign(AV_PIX_FMT_P210) == 2 && chromaYAlign(AV_PIX_FMT_P210) == 1);
    assert(alignCoord(7, 2) == 6 && alignCoord(7, 1) == 7);

    int x = -4, y = 2, w = 20, h = 30;
    assert(clipRect(x, y, w, h, 10, 10) && x == 0 && y == 2 && w == 10 && h == 8);
    assert(!clipRect(x, y, w, h, 0, 10));

    // NV12 chroma plane: half height, same byte width; P210 chroma: full height, 4 bytes per pair.
    int bx, by, bw, bh;
    lumaRectToPlaneRegion(AV_PIX_FMT_NV12, 4, 6, 8, 10, 1, bx, by, bw, bh);
    assert(bx == 4 && by == 3 && bw == 8 && bh == 5);
    lumaRectToPlaneRegion(AV_PIX_FMT_P210, 4, 6, 8, 10, 1, bx, by, bw, bh);
    assert(bx == 8 && by == 6 && bw == 16 && bh == 10);
    lumaRectToPlaneRegion(AV_PIX_FMT_P210, 4, 6, 8, 10, 0, bx, by, bw, bh);
    assert(bx == 8 && bw == 16 && bh == 10);
    lumaRectToPlaneRegion(AV_PIX_FMT_BGRA, 3, 1, 5, 2, 0, bx, by, bw, bh);
    assert(bx == 12 && bw == 20 && bh == 2);

    assert(isYuvPromoteConvertible(AV_PIX_FMT_NV12, AV_PIX_FMT_P210));
    assert(isYuvPromoteConvertible(AV_PIX_FMT_NV12, AV_PIX_FMT_P010));
    assert(!isYuvPromoteConvertible(AV_PIX_FMT_P010, AV_PIX_FMT_NV12));   // never demote
    assert(!isYuvPromoteConvertible(AV_PIX_FMT_NV12, AV_PIX_FMT_NV12));
    assert(!isYuvPromoteConvertible(AV_PIX_FMT_YUV420P, AV_PIX_FMT_P010)); // planar chroma
    assert(isRgbToYuvConvertible(AV_PIX_FMT_BGRA, AV_PIX_FMT_NV12));
    assert(isRgbToYuvConvertible(AV_PIX_FMT_RGB24, AV_PIX_FMT_P210));
    assert(!isRgbToYuvConvertible(AV_PIX_FMT_BGRA, AV_PIX_FMT_YUV420P));
    assert(isAlphaCompatible(AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVA420P));
    assert(!isAlphaCompatible(AV_PIX_FMT_NV12, AV_PIX_FMT_YUVA420P));
    assert(packedAlphaOffset(AV_PIX_FMT_BGRA) == 3 && packedAlphaOffset(AV_PIX_FMT_BGR0) == -1);
    assert(canvasAccepts(AV_PIX_FMT_NV12, AV_PIX_FMT_NV12) && canvasAccepts(AV_PIX_FMT_BGRA, AV_PIX_FMT_P210));
    assert(!canvasAccepts(AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12));

    assert(alphaPlaneIndex(AV_PIX_FMT_YUVA420P) == 3 && alphaPlaneIndex(AV_PIX_FMT_BGRA) == -1);
    uint16_t v = 0;
    assert(planeClearValue(AV_PIX_FMT_P010, nullptr, 0, v) && v == 64);
    assert(planeClearValue(AV_PIX_FMT_P010, nullptr, 1, v) && v == 512);
    assert(planeClearValue(AV_PIX_FMT_NV12, nullptr, 1, v) && v == 128);
    assert(planeClearValue(AV_PIX_FMT_YUVA420P, nullptr, 3, v) && v == 255);
    assert(planeClearValue(AV_PIX_FMT_BGR0, nullptr, 0, v) && v == 0);
    assert(!planeClearValue(AV_PIX_FMT_YUYV422, nullptr, 0, v));
    AVFrame full{};
    full.color_range = AVCOL_RANGE_JPEG;
    assert(blackLumaValue(AV_PIX_FMT_P010, &full) == 0 && blackLumaValue(AV_PIX_FMT_P010, nullptr) == 64);
}
