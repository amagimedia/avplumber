#pragma once

extern "C" {
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
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
    auto listed = [](const std::vector<int>& idx, int i) {
        return std::find(idx.begin(), idx.end(), i) != idx.end();
    };
    auto set_many = [&](const std::vector<int>& idx, float g, bool arriving) {
        for (int i : idx) {
            if (i < 0 || size_t(i) >= n_sources || listed(arriving ? from_idx : to_idx, i))
                continue;
            gains[size_t(i)] = std::max(gains[size_t(i)], g);
        }
    };
    // A source named on both sides of the take stays up. The ramps belong to
    // sources that leave or arrive.
    for (int i : from_idx) {
        if (i >= 0 && size_t(i) < n_sources && listed(to_idx, i))
            gains[size_t(i)] = 1.f;
    }
    if (from_has_program && to_has_program)
        program_gain = 1.f;
    if (kind == "dip") {
        if (t < 0.5) {
            set_many(from_idx, float(1.0 - 2.0 * t), false);
            if (from_has_program && !to_has_program) program_gain = float(1.0 - 2.0 * t);
        } else {
            set_many(to_idx, float(2.0 * t - 1.0), true);
            if (to_has_program && !from_has_program) program_gain = float(2.0 * t - 1.0);
        }
        return;
    }
    set_many(from_idx, float(1.0 - t), false);
    set_many(to_idx, float(t), true);
    if (from_has_program && !to_has_program) program_gain = float(1.0 - t);
    if (to_has_program && !from_has_program) program_gain = float(t);
}

/// The first sample of video tick `tick`. Ticks and samples share no period (1601.6 samples a
/// 29.97 fps tick), so a tick holds the samples from its own first one up to the next tick's.
inline int64_t tickSample(int64_t tick, int sample_rate, int64_t fps_num, int64_t fps_den) {
    return av_rescale_rnd(tick, int64_t(sample_rate) * fps_den, fps_num, AV_ROUND_DOWN);
}

/// The video tick holding `sample`: the last one whose tickSample is at or before it.
inline int64_t sampleTick(int64_t sample, int sample_rate, int64_t fps_num, int64_t fps_den) {
    return av_rescale_rnd(sample + 1, fps_num, int64_t(sample_rate) * fps_den, AV_ROUND_UP) - 1;
}

/// The first sample at or after `ns` on the monotonic clock realtime's set_pts stamps.
inline int64_t sampleAt(int64_t ns, int sample_rate) {
    return av_rescale_rnd(ns, sample_rate, 1000000000, AV_ROUND_UP);
}

inline void addSamples(float* dst, const float* src, size_t begin, size_t end, float gain) {
    for (size_t sample = begin; sample < end; ++sample)
        dst[sample] += src[sample] * gain;
}

/// As addSamples, but a per-sample curve (one entry per index in [0, end)) carries its own
/// gain alongside the constant `gain`, so a change to the curve's target declicks sample-accurately.
inline void addSamplesCurve(float* dst, const float* src, const float* curve, size_t begin, size_t end, float gain) {
    for (size_t sample = begin; sample < end; ++sample)
        dst[sample] += src[sample] * curve[sample] * gain;
}

/// One source's samples on the mixer's sample index (monotonic time × sample rate), from their
/// arrival until the block holding them is read. A block takes what arrived and is silence where
/// nothing did: a late or stopped source never holds the others back. The step into silence fades
/// the last sample out, the step out of it fades the first ones in, each over `ramp` samples, so a
/// gap does not click.
class SampleQueue {
public:
    struct Counts {
        uint64_t dropped = 0;   // arrived after their block was read, twice, or more than max_ahead early
        uint64_t missing = 0;   // filled with silence in a block
        uint64_t jumps = 0;     // frames that started more than `snap` away from the previous one's end
    };

    SampleQueue(size_t channels, int64_t first, int64_t snap, int64_t ramp, int64_t max_ahead)
        : lanes_(channels), tail_(channels, 0.f), read_(first), last_real_(first - ramp - 1),
          resume_(first), snap_(snap), ramp_(ramp), max_ahead_(max_ahead) {}

    /// The index after the last sample held or read.
    int64_t end() const { return read_ + int64_t(lanes_.at(0).size()); }
    const Counts& counts() const { return counts_; }

    /// `planes`: one vector per channel, the frame's first sample at index `first`.
    void write(int64_t first, const std::vector<std::vector<float>>& planes) {
        const size_t n = planes.empty() ? 0 : planes[0].size();
        if (!n) return;
        const int64_t end = this->end();
        if (last_real_ == end - 1 && first != end) {
            // A frame that continues the previous one goes right after it, whatever its
            // timestamp's rounding; a bigger step is a discontinuity, placed where it says.
            if (std::llabs(first - end) <= snap_) first = end;
            else ++counts_.jumps;
        }
        const size_t skip = size_t(std::clamp<int64_t>(end - first, 0, int64_t(n)));
        const int64_t at = first + int64_t(skip);
        if (skip == n || first + int64_t(n) > read_ + max_ahead_) {
            counts_.dropped += n;
            return;
        }
        counts_.dropped += skip;
        for (int64_t i = end; i < at; ++i)
            for (size_t ch = 0; ch < lanes_.size(); ++ch) lanes_[ch].push_back(filler(ch, i));
        if (last_real_ != at - 1) resume_ = at;
        for (size_t k = skip; k < n; ++k) {
            const float gain = rampIn(first + int64_t(k));
            for (size_t ch = 0; ch < lanes_.size(); ++ch) {
                const float v = ch < planes.size() && k < planes[ch].size() ? planes[ch][k] * gain : 0.f;
                lanes_[ch].push_back(v);
                tail_[ch] = v;
            }
        }
        last_real_ = first + int64_t(n) - 1;
    }

    /// The next `n` samples into `out` (one vector per channel), silence for what has not arrived.
    void read(size_t n, std::vector<std::vector<float>>& out) {
        out.resize(lanes_.size());
        const size_t have = std::min(n, lanes_.at(0).size());
        for (size_t ch = 0; ch < lanes_.size(); ++ch) {
            auto& lane = lanes_[ch];
            out[ch].resize(n);
            std::copy(lane.begin(), lane.begin() + std::ptrdiff_t(have), out[ch].begin());
            lane.erase(lane.begin(), lane.begin() + std::ptrdiff_t(have));
            for (size_t k = have; k < n; ++k) out[ch][k] = filler(ch, read_ + int64_t(k));
        }
        counts_.missing += n - have;
        read_ += int64_t(n);
    }

    /// Discards what is held before `sample` and reads on from there.
    void skipTo(int64_t sample) {
        if (sample <= read_) return;
        const size_t drop = size_t(std::min<int64_t>(sample - read_, int64_t(lanes_.at(0).size())));
        for (auto& lane : lanes_) lane.erase(lane.begin(), lane.begin() + std::ptrdiff_t(drop));
        read_ = sample;
    }

private:
    float filler(size_t ch, int64_t i) const {
        const double gain = 1.0 - double(i - last_real_) / double(ramp_ + 1);
        return gain > 0 ? tail_[ch] * float(gain) : 0.f;
    }
    float rampIn(int64_t i) const {
        return i - resume_ < ramp_ ? float(double(i - resume_ + 1) / double(ramp_ + 1)) : 1.f;
    }

    std::vector<std::deque<float>> lanes_;   // [channel], from read_ on
    std::vector<float> tail_;                // the last sample that arrived, as written
    int64_t read_;                           // the next sample a block reads
    int64_t last_real_;                      // the index of tail_
    int64_t resume_;                         // the first sample after the last silence
    int64_t snap_, ramp_, max_ahead_;
    Counts counts_;
};

}  // namespace avp::audio_mixer
