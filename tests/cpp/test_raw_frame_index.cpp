#include "nodes/hwaccel/raw_frame_index.hpp"
#include <iostream>
#include <stdexcept>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

static bool rejected(AVPixelFormat format, int width, int height, AVRational rate, int64_t file_size) {
    try { RawFrameIndex(format, width, height, rate, file_size); }
    catch (const std::invalid_argument &) { return true; }
    return false;
}

void a_picture_is_its_rows_without_padding() {
    const RawFrameIndex nv12(AV_PIX_FMT_NV12, 1920, 1080, {30, 1}, 3110400 * 2);
    CHECK(nv12.frame_size == 3110400);
    CHECK(nv12.row_bytes[0] == 1920 && nv12.row_bytes[1] == 1920 && nv12.row_bytes[2] == 0);
    CHECK(nv12.rows[0] == 1080 && nv12.rows[1] == 540);
    CHECK(nv12.plane_offset[0] == 0 && nv12.plane_offset[1] == 1920 * 1080);

    const RawFrameIndex p010(AV_PIX_FMT_P010LE, 1920, 1080, {30, 1}, 6220800);
    CHECK(p010.frame_size == 6220800);                       // two bytes per sample
    CHECK(p010.row_bytes[0] == 3840 && p010.row_bytes[1] == 3840);
    CHECK(p010.plane_offset[1] == 3840 * 1080);

    // An odd height keeps the chroma row the last luma row needs.
    const RawFrameIndex odd(AV_PIX_FMT_NV12, 1920, 1079, {30, 1}, 1 << 24);
    CHECK(odd.rows[1] == 540 && odd.frame_size == 1920 * (1079 + 540));
}

void only_whole_pictures_count() {
    const int64_t size = 96 * 64 * 3 / 2;
    CHECK(RawFrameIndex(AV_PIX_FMT_NV12, 96, 64, {30, 1}, size * 6 + size / 2).frames == 6);
    CHECK(RawFrameIndex(AV_PIX_FMT_NV12, 96, 64, {30, 1}, size).frames == 1);
    CHECK(rejected(AV_PIX_FMT_NV12, 96, 64, {30, 1}, size - 1));   // not one whole picture
    CHECK(rejected(AV_PIX_FMT_NV12, 96, 64, {30, 1}, 0));
    CHECK(rejected(AV_PIX_FMT_NV12, 0, 64, {30, 1}, size));
    CHECK(rejected(AV_PIX_FMT_NV12, 96, 64, {0, 1}, size));
    CHECK(rejected(AV_PIX_FMT_NONE, 96, 64, {30, 1}, size));
}

void a_looped_file_wraps_and_a_single_pass_ends() {
    const int64_t size = 96 * 64 * 3 / 2;
    const RawFrameIndex index(AV_PIX_FMT_NV12, 96, 64, {30, 1}, size * 6);
    CHECK(index.offsetOf(0, true) == 0 && index.offsetOf(5, true) == 5 * size);
    CHECK(index.offsetOf(6, true) == 0 && index.offsetOf(7, true) == size);
    CHECK(index.offsetOf(6000001, true) == size);            // 6,000,001 = 6 * 1,000,000 + 1
    CHECK(index.offsetOf(5, false) == 5 * size);
    CHECK(index.offsetOf(6, false) < 0 && index.offsetOf(7, false) < 0);   // past the last picture
}

void timestamps_count_pictures_in_the_frame_period() {
    const int64_t size = 96 * 64 * 3 / 2;
    const RawFrameIndex index(AV_PIX_FMT_NV12, 96, 64, {30, 1}, size * 6);
    CHECK(index.time_base.num == 1 && index.time_base.den == 30);
    // Continuous: the count goes on across passes. Otherwise each pass starts at zero again.
    CHECK(index.ptsOf(5, true, true) == 5 && index.ptsOf(7, true, true) == 7);
    CHECK(index.ptsOf(5, true, false) == 5 && index.ptsOf(7, true, false) == 1);
    CHECK(index.ptsOf(5, false, false) == 5);

    const RawFrameIndex ntsc(AV_PIX_FMT_NV12, 96, 64, {30000, 1001}, size);
    CHECK(ntsc.time_base.num == 1001 && ntsc.time_base.den == 30000);
    const RawFrameIndex unreduced(AV_PIX_FMT_NV12, 96, 64, {60, 2}, size);
    CHECK(unreduced.time_base.num == 1 && unreduced.time_base.den == 30);
}

int main() {
    try {
        a_picture_is_its_rows_without_padding();
        only_whole_pictures_count();
        a_looped_file_wraps_and_a_single_pass_ends();
        timestamps_count_pictures_in_the_frame_period();
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
