#include "EventLoop.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

// The engine defines these in avutils.cpp, which needs most of avcpp to link.
constexpr AVRational Wallclock::time_base;
Wallclock wallclock;

static bool within(std::chrono::milliseconds limit, const std::function<bool()> &condition) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!condition()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

int main() {
    using namespace std::chrono_literals;
    EventLoop loop;

    // A handler that waits for its event again from inside the loop, as every non-blocking
    // node does after each frame: the loop passes once per signal. Arming the wait from the
    // loop's own thread needs no wake-up, the loop is about to look at its waits anyway.
    Event event;
    std::atomic<int> runs{0};
    std::function<void(EventLoop &)> handler = [&](EventLoop &evl) {
        ++runs;
        evl.asyncWaitAndExecute(event, handler);
    };
    loop.asyncWaitAndExecute(event, handler);
    std::this_thread::sleep_for(50ms);
    const int signals = 200;
    const uint64_t before = loop.passes();
    for (int i = 0; i < signals; ++i) {
        event.signal();
        assert(within(1000ms, [&] { return runs == i + 1; }));
    }
    std::this_thread::sleep_for(50ms);
    const uint64_t passes = loop.passes() - before;
    assert(runs == signals);
    assert(passes >= uint64_t(signals));
    assert(passes <= uint64_t(signals) + 5);

    // From another thread a wait still wakes the loop: it may be asleep with no timeout.
    Event other;
    std::atomic<bool> ran{false};
    loop.asyncWaitAndExecute(other, [&](EventLoop &) { ran = true; });
    other.signal();
    assert(within(1000ms, [&] { return ran.load(); }));

    // So does work handed over from another thread.
    std::atomic<bool> executed{false};
    loop.execute([&](EventLoop &) { executed = true; });
    assert(within(1000ms, [&] { return executed.load(); }));

    // Work a handler hands to its own loop runs without a wake-up, in the same pass.
    std::atomic<int> chained{0};
    const uint64_t before_chain = loop.passes();
    loop.execute([&](EventLoop &evl) {
        evl.execute([&](EventLoop &inner) {
            ++chained;
            inner.execute([&](EventLoop &) { ++chained; });
        });
    });
    assert(within(1000ms, [&] { return chained == 2; }));
    std::this_thread::sleep_for(20ms);
    assert(loop.passes() - before_chain <= 2);

    // A timer set from inside the loop, sooner than any other, fires on time without one.
    std::atomic<int64_t> fired_after_ms{-1};
    loop.execute([&](EventLoop &evl) {
        const auto set = std::chrono::steady_clock::now();
        evl.sleepAndExecute(30, [&, set](EventLoop &) {
            fired_after_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - set).count();
        });
    });
    assert(within(1000ms, [&] { return fired_after_ms >= 0; }));
    assert(fired_after_ms >= 29 && fired_after_ms <= 60);

    // And from another thread, while the loop sleeps with no timeout.
    std::atomic<bool> timed{false};
    const auto set = std::chrono::steady_clock::now();
    loop.sleepAndExecute(30, [&](EventLoop &) { timed = true; });
    assert(within(1000ms, [&] { return timed.load(); }));
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - set).count();
    assert(waited >= 29 && waited <= 60);
    return 0;
}
