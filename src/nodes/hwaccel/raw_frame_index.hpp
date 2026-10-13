#pragma once
// Geometry and timing of headerless raw video: the bytes of one picture as libavutil lays them
// out without row padding, and where picture n of a file is and when it is shown. libavutil only.
#include <cstdint>
#include <stdexcept>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

/// One picture as contiguous planes, each row exactly its samples.
struct RawPictureLayout {
    int64_t frame_size = 0;
    int row_bytes[4] = {};
    int rows[4] = {};
    int64_t plane_offset[4] = {};

    RawPictureLayout(AVPixelFormat format, int width, int height) {
        ptrdiff_t linesizes[4];
        size_t sizes[4];
        if (format == AV_PIX_FMT_NONE || width <= 0 || height <= 0 ||
            av_image_fill_linesizes(row_bytes, format, width) < 0)
            throw std::invalid_argument("raw video: unusable pixel format or picture size");
        for (int i = 0; i < 4; ++i) linesizes[i] = row_bytes[i];
        if (av_image_fill_plane_sizes(sizes, format, height, linesizes) < 0)
            throw std::invalid_argument("raw video: unusable pixel format or picture size");
        for (int i = 0; i < 4; ++i) {
            plane_offset[i] = frame_size;
            rows[i] = row_bytes[i] ? int(sizes[i] / size_t(row_bytes[i])) : 0;
            frame_size += int64_t(sizes[i]);
        }
    }
};

/// The pictures of a file of `file_size` bytes played at `framerate`. A trailing part of a
/// picture does not count.
struct RawFrameIndex: RawPictureLayout {
    int64_t frames = 0;
    AVRational time_base{0, 1};   // one tick per picture

    RawFrameIndex(AVPixelFormat format, int width, int height, AVRational framerate, int64_t file_size):
        RawPictureLayout(format, width, height) {
        if (framerate.num <= 0 || framerate.den <= 0) throw std::invalid_argument("raw video: unusable frame rate");
        av_reduce(&time_base.num, &time_base.den, framerate.den, framerate.num, INT32_MAX);
        frames = file_size / frame_size;
        if (frames <= 0) throw std::invalid_argument("raw video: the file holds no whole picture");
    }

    /// Byte offset of picture `n` counted from the start of playback; negative past the end of
    /// a file that is read once.
    int64_t offsetOf(uint64_t n, bool loop) const {
        if (loop) return int64_t(n % uint64_t(frames)) * frame_size;
        return n < uint64_t(frames) ? int64_t(n) * frame_size : -1;
    }

    /// Timestamp of picture `n` in `time_base`: counting on across passes, or from zero again
    /// at each pass.
    int64_t ptsOf(uint64_t n, bool loop, bool continuous) const {
        return loop && !continuous ? int64_t(n % uint64_t(frames)) : int64_t(n);
    }
};
