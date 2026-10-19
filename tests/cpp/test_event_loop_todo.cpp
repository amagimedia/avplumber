#include "EventLoop.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
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

// Microseconds per loop pass: a handler that waits for its event again, signalled from here once per run.
static double usPerPass(EventLoop &loop, int passes) {
    Event event;
    std::atomic<int> runs{0};
    std::function<void(EventLoop &)> handler = [&](EventLoop &evl) {
        ++runs;
        evl.asyncWaitAndExecute(event, handler);
    };
    loop.asyncWaitAndExecute(event, handler);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 1; i <= passes; ++i) {
        event.signal();
        while (runs.load() < i) { }
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    std::atomic<bool> parked{false};
    loop.asyncWaitAndExecute(event, [&](EventLoop &) { parked = true; });   // replaces the handler
    event.signal();
    assert(within(std::chrono::milliseconds(1000), [&] { return parked.load(); }));
    return us / passes;
}

int main() {
    using namespace std::chrono_literals;

    // Work handed to the loop from many threads at once all runs, each piece once.
    {
        EventLoop loop;
        const int threads = 300, each = 10;
        std::vector<std::atomic<int>> ran(threads * each);
        std::vector<std::thread> posters;
        for (int t = 0; t < threads; ++t)
            posters.emplace_back([&, t] {
                for (int i = 0; i < each; ++i) loop.execute([&ran, t, i, each](EventLoop &) { ++ran[t * each + i]; });
            });
        for (auto &p : posters) p.join();
        assert(within(2000ms, [&] { for (auto &r : ran) if (r != 1) return false; return true; }));
        std::this_thread::sleep_for(20ms);
        for (auto &r : ran) assert(r == 1);

        // Work handed over from the loop's own thread runs too, without a wake-up.
        std::atomic<int> chained{0};
        loop.execute([&](EventLoop &evl) { evl.execute([&](EventLoop &) { ++chained; }); });
        assert(within(1000ms, [&] { return chained.load() == 1; }));
    }

    // A pass costs the same whatever number of threads ever handed the loop work: a show's
    // loops have heard from hundreds, and look for work after every pass.
    {
        EventLoop quiet, busy;
        std::vector<std::thread> posters;
        std::atomic<int> ran{0};
        std::promise<void> done;
        std::shared_future<void> released = done.get_future().share();
        // The threads stay alive, as a show's source and decoder threads do.
        for (int t = 0; t < 400; ++t)
            posters.emplace_back([&] {
                busy.execute([&](EventLoop &) { ++ran; });
                released.wait();
            });
        assert(within(2000ms, [&] { return ran.load() == 400; }));
        usPerPass(quiet, 2000); usPerPass(busy, 2000);   // warm up
        double q = 1e9, b = 1e9;
        for (int round = 0; round < 3; ++round) {
            q = std::min(q, usPerPass(quiet, 20000));
            b = std::min(b, usPerPass(busy, 20000));
        }
        done.set_value();
        for (auto &p : posters) p.join();
        std::fprintf(stderr, "us per pass: loop that heard from 1 thread %.2f, from 400 threads %.2f\n", q, b);
        assert(b < q * 1.5 + 1.0);
    }

    // A handler that hands itself back to the loop (a node yielding after each frame) costs the
    // same whatever number of other threads have handed the loop work and are still alive.
    {
        auto usPerYield = [](EventLoop &loop, int yields) {
            std::atomic<int> left{yields};
            std::function<void(EventLoop &)> step = [&](EventLoop &evl) {
                if (--left > 0) evl.execute(step);
            };
            const auto start = std::chrono::steady_clock::now();
            loop.execute(step);
            while (left.load() > 0) { }
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / yields;
        };
        EventLoop quiet, busy;
        std::vector<std::thread> posters;
        std::atomic<int> ran{0};
        std::promise<void> done;
        std::shared_future<void> released = done.get_future().share();
        for (int t = 0; t < 400; ++t)
            posters.emplace_back([&] {
                busy.execute([&](EventLoop &) { ++ran; });
                released.wait();
            });
        assert(within(2000ms, [&] { return ran.load() == 400; }));
        usPerYield(quiet, 2000); usPerYield(busy, 2000);   // warm up
        double q = 1e9, b = 1e9;
        for (int round = 0; round < 3; ++round) {
            q = std::min(q, usPerYield(quiet, 50000));
            b = std::min(b, usPerYield(busy, 50000));
        }
        done.set_value();
        for (auto &p : posters) p.join();
        std::fprintf(stderr, "us per yield: loop that heard from 1 thread %.3f, from 400 threads %.3f\n", q, b);
        assert(b < q * 1.5 + 0.1);
    }

    std::puts("event loop work queue: ok");
    return 0;
}
