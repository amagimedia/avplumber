#include "EventLoop.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

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

    // More waits than the loop takes from the kernel in one go, all signalled at once: every
    // handler runs, once.
    {
        const int many = 300;
        std::vector<Event> events(many);
        std::vector<std::atomic<int>> ran_each(many);
        for (int i = 0; i < many; ++i) loop.asyncWaitAndExecute(events[i], [&ran_each, i](EventLoop &) { ++ran_each[i]; });
        for (auto &e : events) e.signal();
        assert(within(2000ms, [&] { for (auto &r : ran_each) if (r != 1) return false; return true; }));
        std::this_thread::sleep_for(20ms);
        for (auto &r : ran_each) assert(r == 1);
    }

    // An event signalled before anyone waits for it runs its handler as soon as one does; two
    // signals before the handler runs are one run, and nothing runs again without a new wait.
    {
        Event early;
        std::atomic<int> early_runs{0};
        early.signal();
        early.signal();
        loop.asyncWaitAndExecute(early, [&](EventLoop &) { ++early_runs; });
        assert(within(1000ms, [&] { return early_runs == 1; }));
        early.signal();
        std::this_thread::sleep_for(20ms);
        assert(early_runs == 1);
        loop.asyncWaitAndExecute(early, [&](EventLoop &) { ++early_runs; });
        assert(within(1000ms, [&] { return early_runs == 2; }));
    }

    // An event that goes away while waited for leaves nothing behind that a later event, which
    // may well get the same descriptor, would trip over.
    for (int round = 0; round < 50; ++round) {
        std::atomic<bool> reused{false};
        {
            Event gone;
            loop.asyncWaitAndExecute(gone, [](EventLoop &) { assert(!"a handler ran for an event nobody signalled"); });
        }
        Event next;
        loop.asyncWaitAndExecute(next, [&](EventLoop &) { reused = true; });
        next.signal();
        assert(within(1000ms, [&] { return reused.load(); }));
    }

    // Timers fire in the order of their times, those set for the same time in the order set.
    {
        std::mutex order_busy;
        std::vector<int> order;
        const int delays[] = {40, 10, 30, 10, 20, 40, 10};
        loop.execute([&](EventLoop &evl) {   // set from one pass of the loop: one common starting time
            for (int i = 0; i < 7; ++i)
                evl.sleepAndExecute(delays[i], [&, i](EventLoop &) { std::lock_guard<std::mutex> lock(order_busy); order.push_back(i); });
        });
        assert(within(1000ms, [&] { std::lock_guard<std::mutex> lock(order_busy); return order.size() == 7; }));
        assert((order == std::vector<int>{1, 3, 6, 4, 2, 0, 5}));
    }

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
