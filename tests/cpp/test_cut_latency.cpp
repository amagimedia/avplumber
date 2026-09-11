#include "mixer/CutLatency.hpp"
#include "CommandTiming.hpp"
#include <cassert>
#include <cmath>
#include <thread>

using avp::mixer::CutLatency;
using namespace std::chrono_literals;

int main() {
    CutLatency meter;
    const auto start = CutLatency::Clock::now();
    meter.begin("B", false, 1, start);
    assert(meter.tokenForInput(1, false) == 0);
    meter.arm();
    auto token = meter.tokenForInput(1, false);
    assert(token && !meter.tokenForInput(0, false) && !meter.tokenForInput(1, true));
    meter.encoderOutput(10, start + 1ms); // Old encoded picture cannot complete a cut.
    meter.encoderInput(token - 1, 11);
    meter.encoderOutput(11, start + 2ms);
    assert(!meter.snapshot(start).direct.milliseconds);
    meter.encoderInput(token, 12);
    meter.encoderInput(token, 13);
    meter.encoderOutput(13, start + 133250us); // Reordering/dropped first frame.
    auto sample = meter.snapshot(start).direct;
    assert(sample.state == "measured" && std::abs(*sample.milliseconds - 133.25) < 1e-8);
    assert(sample.encoded_pts == 13);
    meter.encoderOutput(12, start + 150ms);
    assert(meter.snapshot(start).direct.milliseconds == sample.milliseconds);

    meter.begin("A", true, 0, start);
    meter.arm();
    auto previewed = meter.tokenForInput(0, false);
    assert(previewed != token);
    meter.encoderInput(previewed, 20);
    meter.encoderOutput(20, start + 33250us);
    assert(*meter.snapshot(start).previewed.milliseconds == 33.25);
    assert(meter.snapshot(start).direct.milliseconds == sample.milliseconds);

    meter.begin("B", false, 1, start);
    meter.arm();
    token = meter.tokenForInput(1, false);
    meter.encoderInput(token, 30);
    meter.cancel();
    meter.encoderOutput(30, start + 50ms);
    assert(meter.snapshot(start).direct.state == "interrupted");
    assert(!meter.snapshot(start).direct.milliseconds);
    meter.begin("A", true, 0, start);
    meter.arm();
    meter.encoderInput(token, 31); // Stale generation.
    meter.encoderOutput(31, start + 50ms);
    assert(!meter.snapshot(start).previewed.milliseconds);
    assert(meter.snapshot(start + 31s).previewed.state == "timeout");

    meter.begin("A", true, 0, start);
    meter.arm();
    token = meter.tokenForInput(0, false);
    meter.encoderInput(token, 40);
    meter.encoderInput(token, 40);
    assert(meter.snapshot(start).previewed.state == "unmatched");
    meter.begin("B", false, 1, start);
    meter.arm();
    token = meter.tokenForInput(1, false);
    for (int i = 100; i <= 356; ++i) meter.encoderInput(token, i);
    assert(meter.snapshot(start).direct.state == "unmatched");

    CommandTiming outer(start);
    { CommandTiming inner(start + 1s); assert(CommandTiming::received() == start + 1s); }
    assert(CommandTiming::received() == start);
    std::thread other([&] { CommandTiming timing(start + 2s); assert(CommandTiming::received() == start + 2s); });
    other.join();
    assert(CommandTiming::received() == start);
}
