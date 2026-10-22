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

int main() {
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
}
