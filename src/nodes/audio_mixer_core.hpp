#pragma once

extern "C" {
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace avp::audio_mixer {

inline void gainsForProgress(const std::string& kind, double t,
                             const std::vector<int>& from_idx, const std::vector<int>& to_idx,
                             size_t n_sources, std::vector<float>& gains, float& program_gain,
                             bool from_has_program, bool to_has_program) {
    gains.assign(n_sources, 0.f);
    program_gain = 0.f;
    t = std::clamp(t, 0.0, 1.0);
    auto set_many = [&](const std::vector<int>& idx, float g) {
        for (int i : idx) {
            if (i >= 0 && size_t(i) < n_sources)
                gains[size_t(i)] = std::max(gains[size_t(i)], g);
        }
    };
    if (kind == "dip") {
        if (t < 0.5) {
            set_many(from_idx, float(1.0 - 2.0 * t));
            if (from_has_program) program_gain = float(1.0 - 2.0 * t);
        } else {
            set_many(to_idx, float(2.0 * t - 1.0));
            if (to_has_program) program_gain = float(2.0 * t - 1.0);
        }
        return;
    }
    auto add_many = [&](const std::vector<int>& idx, float g) {
        for (int i : idx) {
            if (i >= 0 && size_t(i) < n_sources)
                gains[size_t(i)] = std::min(1.f, gains[size_t(i)] + g);
        }
    };
    add_many(from_idx, float(1.0 - t));
    add_many(to_idx, float(t));
    if (from_has_program) program_gain += float(1.0 - t);
    if (to_has_program) program_gain += float(t);
    program_gain = std::min(1.f, program_gain);
}

inline size_t eventSampleOffset(int64_t pts_ns, int64_t block_start_ns, int64_t frame_start,
                                int sample_rate, size_t frame_samples, bool have_media_pts) {
    const int64_t sample = have_media_pts
        ? av_rescale_rnd(pts_ns - block_start_ns, sample_rate, 1000000000LL, AV_ROUND_UP)
        : av_rescale_rnd(pts_ns, sample_rate, 1000000000LL, AV_ROUND_UP) - frame_start;
    return size_t(std::clamp<int64_t>(sample, 0, int64_t(frame_samples)));
}

inline void addSamples(float* dst, const float* src, size_t begin, size_t end, float gain) {
    for (size_t sample = begin; sample < end; ++sample)
        dst[sample] += src[sample] * gain;
}

}  // namespace avp::audio_mixer
