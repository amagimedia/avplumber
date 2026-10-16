#include "nodes/audio_mixer_core.hpp"
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    using namespace avp::audio_mixer;

    std::vector<float> gains;
    float program_gain = 0.f;
    gainsForProgress("fade", 0.5, {0, 1}, {1, 2}, 3, gains, program_gain, false, false);
    assert(gains == std::vector<float>({0.5f, 1.f, 0.5f}));

    gainsForProgress("fade", 0.25, {}, {}, 0, gains, program_gain, true, true);
    assert(program_gain == 1.f);

    gainsForProgress("dip", 0.25, {0}, {0}, 1, gains, program_gain, false, false);
    assert(gains == std::vector<float>({1.f}));
    gainsForProgress("dip", 0.25, {0, 1}, {1, 2}, 3, gains, program_gain, true, true);
    assert(gains == std::vector<float>({0.5f, 1.f, 0.f}));
    assert(program_gain == 1.f);

    constexpr int sample_rate = 48000;
    constexpr size_t frame_samples = 960;
    constexpr int64_t block_start = 123000000000LL;
    assert(eventSampleOffset(block_start + 10000000, block_start, 0,
                             sample_rate, frame_samples, true) == 480);
    assert(eventSampleOffset(block_start + 1, block_start, 0,
                             sample_rate, frame_samples, true) == 1);
    assert(eventSampleOffset(10000000, 0, 0,
                             sample_rate, frame_samples, false) == 480);

    std::vector<float> out(4, 0.f);
    const std::vector<float> positive(4, 0.8f);
    const std::vector<float> negative(4, -0.8f);
    addSamples(out.data(), positive.data(), 0, out.size(), 1.f);
    addSamples(out.data(), positive.data(), 0, out.size(), 1.f);
    for (float sample : out) assert(sample > 1.f);
    addSamples(out.data(), negative.data(), 0, out.size(), 1.f);
    for (float sample : out) assert(std::fabs(sample - 0.8f) < 1e-6f);
}
