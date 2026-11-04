#include "nodes/audio_mixer_core.hpp"
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

using namespace avp::audio_mixer;

using Planes = std::vector<std::vector<float>>;

static Planes constant(size_t n, float value) {
    return Planes(1, std::vector<float>(n, value));
}

/// Gains off a trigonometric curve are not exact binary fractions, and an end of one lands on
/// cos/sin's own rounding (1e-17, far under the mixer's 1e-6 gain epsilon) rather than on zero.
static bool near(float value, float expected) {
    return std::abs(value - expected) < 1e-6f;
}

static void tickMath() {
    // 25 fps: 1920 samples a tick; 29.97: no common period, ticks of 1601 or 1602 samples.
    assert(tickSample(1, 48000, 25, 1) == 1920);
    assert(tickSample(1, 48000, 30000, 1001) == 1601);
    assert(tickSample(2, 48000, 30000, 1001) == 3203);
    assert(tickSample(5, 48000, 30000, 1001) == 8008);
    for (const auto& [num, den] : std::vector<std::pair<int64_t, int64_t>>{{25, 1}, {30000, 1001}, {60, 1}}) {
        for (int64_t tick = 0; tick < 200; ++tick) {
            const int64_t first = tickSample(tick, 48000, num, den);
            assert(sampleTick(first, 48000, num, den) == tick);
            assert(sampleTick(first - 1, 48000, num, den) == tick - 1);
        }
    }
    // A 20 ms block's last sample at 25 fps: blocks 0 and 1 end in tick 0, block 2 in tick 1.
    assert(sampleTick(959, 48000, 25, 1) == 0);
    assert(sampleTick(1919, 48000, 25, 1) == 0);
    assert(sampleTick(2879, 48000, 25, 1) == 1);
    assert(sampleAt(1000000000, 48000) == 48000);
    assert(sampleAt(1, 48000) == 1);
}

static void continuousFramesReadBack() {
    SampleQueue q(1, 1000, 48, 4, 48000);
    q.write(1000, constant(6, 0.5f));
    q.write(1006 + 3, constant(6, 0.5f));   // rounding of its timestamp: still continues the first
    Planes out;
    q.read(12, out);
    assert(out[0][0] < 0.5f);             // the stream starts out of silence: ramped in
    for (size_t i = 4; i < 12; ++i) assert(out[0][i] == 0.5f);
    assert(q.counts().missing == 0 && q.counts().jumps == 0 && q.counts().dropped == 0);
}

static void stoppedSourceFadesToSilence() {
    SampleQueue q(1, 0, 48, 4, 48000);
    q.write(0, constant(4, 1.f));
    Planes out;
    q.read(10, out);
    // Ramped in from silence, then faded out over the ramp, then silence.
    const std::vector<float> expected{0.2f, 0.4f, 0.6f, 0.8f, 0.64f, 0.48f, 0.32f, 0.16f, 0.f, 0.f};
    for (size_t i = 0; i < expected.size(); ++i) assert(std::fabs(out[0][i] - expected[i]) < 1e-6f);
    assert(q.counts().missing == 6);
}

static void lateFrameDroppedThenRampsIn() {
    SampleQueue q(1, 0, 0, 2, 48000);
    Planes out;
    q.read(8, out);                       // nothing arrived: silence
    q.write(4, constant(8, 1.f));         // half of it is for samples already out
    assert(q.counts().dropped == 4);
    q.read(4, out);
    const std::vector<float> expected{1.f / 3, 2.f / 3, 1.f, 1.f};
    for (size_t i = 0; i < expected.size(); ++i) assert(std::fabs(out[0][i] - expected[i]) < 1e-6f);
}

static void gapInsideTheQueueIsFilled() {
    SampleQueue q(1, 0, 1, 2, 48000);
    q.write(0, constant(4, 1.f));
    q.write(10, constant(4, 1.f));        // a jump past the snap: silence between
    assert(q.counts().jumps == 1);
    Planes out;
    q.read(14, out);
    assert(std::fabs(out[0][4] - 2.f / 3) < 1e-6f && out[0][6] == 0.f && out[0][9] == 0.f);
    assert(std::fabs(out[0][10] - 1.f / 3) < 1e-6f && out[0][13] == 1.f);
    assert(q.counts().missing == 0);
}

static void farFutureAndSkip() {
    SampleQueue q(1, 0, 0, 2, 100);
    q.write(200, constant(4, 1.f));       // more than max_ahead early
    assert(q.counts().dropped == 4 && q.end() == 0);
    q.write(0, constant(10, 1.f));
    q.skipTo(6);
    Planes out;
    q.read(4, out);
    for (float v : out[0]) assert(v == 1.f);
    assert(q.counts().missing == 0);
}

static void historyReadsBackAtAnyDelay() {
    // Blocks of 960 written in order; a delay of 1601 samples (one 29.97 fps frame) is no
    // multiple of the block, and a read spans two of them.
    SampleHistory h(1, 9600);
    for (int64_t block = 0; block < 6; ++block) {
        Planes planes(1, std::vector<float>(960));
        for (size_t k = 0; k < 960; ++k) planes[0][k] = float(block * 960 + int64_t(k));
        h.write(10000 + block * 960, planes);
    }
    Planes out;
    h.read(10000 + 5 * 960 - 1601, 960, out);
    for (size_t k = 0; k < 960; ++k) assert(out[0][k] == float(5 * 960 - 1601 + int64_t(k)));
    // Delay 0: the block just written.
    h.read(10000 + 5 * 960, 960, out);
    for (size_t k = 0; k < 960; ++k) assert(out[0][k] == float(5 * 960 + int64_t(k)));
    // Before the first block written, and further back than the capacity: silence.
    h.read(10000 - 10, 20, out);
    for (size_t k = 0; k < 10; ++k) assert(out[0][k] == 0.f);
    assert(out[0][10] == 0.f && out[0][11] == 1.f);
    for (int64_t block = 6; block < 16; ++block) h.write(10000 + block * 960, constant(960, 7.f));
    h.read(10000, 960, out);
    for (float v : out[0]) assert(v == 0.f);
}

static void historyForgetsAcrossASkip() {
    SampleHistory h(1, 4800);
    h.write(0, constant(960, 1.f));
    h.write(960, constant(960, 1.f));
    h.write(100000, constant(960, 2.f));   // the mixer skipped ahead: what came before is gone
    Planes out;
    h.read(100000 - 960, 1920, out);
    for (size_t k = 0; k < 960; ++k) assert(out[0][k] == 0.f);
    for (size_t k = 960; k < 1920; ++k) assert(out[0][k] == 2.f);
}

int main() {
    std::vector<float> gains;
    float program_gain = 0.f;
    // Halfway through a fade, what leaves and what arrives are each at cos/sin of a quarter turn,
    // not at 0.5: their powers add, so equal amplitudes would sag 3 dB. Source 1, named on both
    // sides, stays up throughout and is not part of either ramp.
    gainsForProgress("fade", 0.5, {0, 1}, {1, 2}, 3, gains, program_gain, false, false);
    assert(near(gains.at(0), 0.70710678f) && gains.at(1) == 1.f && near(gains.at(2), 0.70710678f));

    // The property that law is for: one source out, a different one in, constant power all the
    // way across -- and the ends are still exactly one source at full gain.
    for (int step = 0; step <= 20; ++step) {
        const double t = step / 20.0;
        gainsForProgress("fade", t, {0}, {1}, 2, gains, program_gain, false, false);
        const double power = double(gains.at(0)) * gains.at(0) + double(gains.at(1)) * gains.at(1);
        assert(std::abs(power - 1.0) < 1e-6);
    }
    gainsForProgress("fade", 0.0, {0}, {1}, 2, gains, program_gain, false, false);
    assert(gains.at(0) == 1.f && near(gains.at(1), 0.f));
    gainsForProgress("fade", 1.0, {0}, {1}, 2, gains, program_gain, false, false);
    assert(near(gains.at(0), 0.f) && gains.at(1) == 1.f);

    // A wipe is heard the same way: both sides are on air together, so it shares the fade's law.
    gainsForProgress("wipe", 0.5, {0}, {1}, 2, gains, program_gain, false, false);
    assert(near(gains.at(0), 0.70710678f) && near(gains.at(1), 0.70710678f));

    // An AUX bus crossfading into or out of the program follows it too.
    gainsForProgress("fade", 0.5, {}, {}, 0, gains, program_gain, true, false);
    assert(near(program_gain, 0.70710678f));
    gainsForProgress("fade", 0.5, {}, {}, 0, gains, program_gain, false, true);
    assert(near(program_gain, 0.70710678f));

    gainsForProgress("fade", 0.25, {}, {}, 0, gains, program_gain, true, true);
    assert(program_gain == 1.f);

    gainsForProgress("dip", 0.25, {0}, {0}, 1, gains, program_gain, false, false);
    assert(gains == std::vector<float>({1.f}));
    gainsForProgress("dip", 0.25, {0, 1}, {1, 2}, 3, gains, program_gain, true, true);
    assert(gains == std::vector<float>({0.5f, 1.f, 0.f}));
    assert(program_gain == 1.f);

    std::vector<float> out(4, 0.f);
    const std::vector<float> positive(4, 0.8f);
    const std::vector<float> negative(4, -0.8f);
    addSamples(out.data(), positive.data(), 0, out.size(), 1.f);
    addSamples(out.data(), positive.data(), 0, out.size(), 1.f);
    for (float sample : out) assert(sample > 1.f);
    addSamples(out.data(), negative.data(), 0, out.size(), 1.f);
    for (float sample : out) assert(std::fabs(sample - 0.8f) < 1e-6f);

    tickMath();
    continuousFramesReadBack();
    stoppedSourceFadesToSilence();
    lateFrameDroppedThenRampsIn();
    gapInsideTheQueueIsFilled();
    farFutureAndSkip();
    historyReadsBackAtAnyDelay();
    historyForgetsAcrossASkip();
}
