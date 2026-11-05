#pragma once

extern "C" {
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <cmath>
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
    // Equal power across the overlap, for the takes whose two sides are heard together (fade and
    // wipe; `dip` has returned above, having gone through real silence instead). What leaves and
    // what arrives is different material, so their powers add rather than their amplitudes:
    // linear 1-t and t sum to half the power halfway through, a 3 dB sag in the middle of every
    // transition, where cos and sin hold the sum at one. The video keeps its own law -- alpha is
    // linear in `t`, which is already eased by the take's curve -- because a picture mixes by
    // amplitude where loudness follows power. A source named on both sides is pinned at 1 above
    // and left out of both ramps, so none is lifted to cos + sin (+3 dB) here.
    constexpr double kHalfPi = 1.57079632679489661923;
    const double leaving = std::cos(t * kHalfPi), arriving = std::sin(t * kHalfPi);
    set_many(from_idx, float(leaving), false);
    set_many(to_idx, float(arriving), true);
    if (from_has_program && !to_has_program) program_gain = float(leaving);
    if (to_has_program && !from_has_program) program_gain = float(arriving);
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

/// The last `capacity` samples of a mix by sample index, for buses that use it some samples late:
/// an AUX bus's PGM tile shows the program a few frames back, so its "program" audio is read back
/// as far. A sample never written, or overwritten since, reads as silence.
class SampleHistory {
public:
    SampleHistory(size_t channels, size_t capacity)
        : lanes_(channels, std::vector<float>(capacity, 0.f)), capacity_(int64_t(capacity)) {}

    int64_t capacity() const { return capacity_; }

    /// `planes`: one vector per channel, the block's first sample at index `first`. A block that
    /// does not continue the previous one (the mixer skipped ahead) forgets what came before.
    void write(int64_t first, const std::vector<std::vector<float>>& planes) {
        const size_t n = planes.empty() ? 0 : planes[0].size();
        if (!n) return;
        if (first != end_) valid_from_ = first;
        for (size_t ch = 0; ch < lanes_.size(); ++ch)
            for (size_t k = 0; k < n; ++k)
                lanes_[ch][slot(first + int64_t(k))] = ch < planes.size() && k < planes[ch].size() ? planes[ch][k] : 0.f;
        end_ = first + int64_t(n);
    }

    /// The `n` samples from index `first` into `out` (one vector per channel).
    void read(int64_t first, size_t n, std::vector<std::vector<float>>& out) const {
        out.resize(lanes_.size());
        const int64_t from = std::max(valid_from_, end_ - capacity_);
        for (size_t ch = 0; ch < lanes_.size(); ++ch) {
            out[ch].resize(n);
            for (size_t k = 0; k < n; ++k) {
                const int64_t i = first + int64_t(k);
                out[ch][k] = i >= from && i < end_ ? lanes_[ch][slot(i)] : 0.f;
            }
        }
    }

private:
    size_t slot(int64_t i) const { return size_t(((i % capacity_) + capacity_) % capacity_); }

    std::vector<std::vector<float>> lanes_;   // [channel], a ring indexed by sample % capacity_
    int64_t capacity_;
    int64_t end_ = 0;          // one past the last sample written
    int64_t valid_from_ = 0;   // the first sample of the current unbroken run
};

struct DuckerParams {
    bool enabled = true;
    float threshold_db = -40.f;   // on the sidechain's peak, dBFS
    float depth_db = -20.f;       // the gain while ducked; 0 or below
    float attack_ms = 10.f;       // time to reach the full depth
    float hold_ms = 300.f;        // ducked this long after the sidechain falls under the threshold
    float release_ms = 500.f;     // time to come back from the full depth
};

/// A sidechain ducker: lowers its targets by `depth_db` while the sidechain's peak is at or above
/// the threshold, and for `hold_ms` after. The gain moves linearly in dB, the full depth in
/// attack_ms down and release_ms up, so a change of depth or a disable glides instead of clicking.
class Ducker {
public:
    explicit Ducker(int sample_rate) : sample_rate_(sample_rate) { setParams(DuckerParams{}); }

    const DuckerParams& params() const { return params_; }
    void setParams(const DuckerParams& p) {
        params_ = p;
        params_.depth_db = std::min(0.f, p.depth_db);
        threshold_ = std::pow(10.f, p.threshold_db / 20.f);
        hold_ = int64_t(std::llround(std::max(0.f, p.hold_ms) * 1e-3 * sample_rate_));
        // A shallower depth set mid-duck still comes back within release_ms.
        const float range = std::max({1.f, -params_.depth_db, -gain_db_});
        down_ = range / std::max(1.f, p.attack_ms * 1e-3f * float(sample_rate_));
        up_ = range / std::max(1.f, p.release_ms * 1e-3f * float(sample_rate_));
    }

    /// The gain now, in dB: 0 when not ducking.
    float gainDb() const { return gain_db_; }

    /// `key`: the sidechain's peak magnitude per sample, `n` of them; `gain`: the targets' linear
    /// gain per sample, written.
    void process(const float* key, size_t n, float* gain) {
        for (size_t k = 0; k < n; ++k) {
            if (!params_.enabled) held_ = 0;
            else if (key[k] >= threshold_) held_ = hold_ + 1;
            const bool ducking = held_ > 0;
            if (held_ > 0) --held_;
            const float target = ducking ? params_.depth_db : 0.f;
            if (gain_db_ > target) gain_db_ = std::max(target, gain_db_ - down_);
            else if (gain_db_ < target) gain_db_ = std::min(target, gain_db_ + up_);
            gain[k] = gain_db_ < 0.f ? std::pow(10.f, gain_db_ / 20.f) : 1.f;
        }
    }

private:
    int sample_rate_;
    DuckerParams params_;
    float threshold_ = 0.f, down_ = 0.f, up_ = 0.f;
    int64_t hold_ = 0, held_ = 0;
    float gain_db_ = 0.f;
};

}  // namespace avp::audio_mixer
