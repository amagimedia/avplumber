#include "EventLoop.hpp"
#include "MultiEventWait.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

// The engine defines these in avutils.cpp, which needs most of avcpp to link.
constexpr AVRational Wallclock::time_base;
Wallclock wallclock;

// The descriptor's count, read without waiting: how many writes reached it since the last read.
static uint64_t writes(Event &event) { return event.wait(0); }

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

    // A plain event writes on every signal, as it always did.
    Event plain;
    for (int i = 0; i < 100; ++i) plain.signal();
    assert(writes(plain) == 100);

    // A coalescing event writes once until a reader has read it: a queue's consumer that is not
    // waiting costs its producer no system call per frame.
    Event queue_event(Event::coalesce);
    for (int i = 0; i < 100; ++i) queue_event.signal();
    assert(writes(queue_event) == 1);
    // Read by Event::wait: the next signal writes again.
    queue_event.signal();
    assert(writes(queue_event) == 1);
    assert(writes(queue_event) == 0);

    // Read by MultiEventWait: the next signal writes again.
    {
        Event other(Event::coalesce);
        MultiEventWait wait({&queue_event, &other});
        queue_event.signal();
        queue_event.signal();
        assert(wait.wait(0));
        assert(!wait.wait(0));
        queue_event.signal();
        assert(wait.wait(0));
        assert(!wait.wait(0));
    }

    // Read by the event loop: a handler that waits again is run once per signal it was armed for.
    {
        // Declared before the loop, so that the loop's thread has stopped before they go.
        Event event(Event::coalesce);
        std::atomic<int> runs{0};
        std::function<void(EventLoop &)> handler = [&](EventLoop &evl) {
            ++runs;
            evl.asyncWaitAndExecute(event, handler);
        };
        EventLoop loop;
        loop.asyncWaitAndExecute(event, handler);
        for (int i = 1; i <= 50; ++i) {
            event.signal();
            assert(within(2000ms, [&] { return runs.load() == i; }));
        }
    }

    // A descriptor number reused by a new event starts with nothing written, whatever the
    // closed event's last signal left behind.
    {
        int fd;
        {
            Event closed(Event::coalesce);
            closed.signal();
            fd = closed.fd();
        }
        Event reused(Event::coalesce);
        assert(reused.fd() == fd);
        reused.signal();
        assert(writes(reused) == 1);
    }

    // No wake-up is lost: a producer that signals after each item and a consumer that sleeps
    // whenever it has caught up. A lost wake-up leaves the consumer asleep with items waiting,
    // and its wait times out.
    {
        Event event(Event::coalesce);
        std::atomic<uint64_t> produced{0};
        const uint64_t items = 300000;
        int timeouts = 0;
        std::thread producer([&] {
            for (uint64_t i = 1; i <= items; ++i) {
                produced.store(i, std::memory_order_release);
                event.signal();
                if (i % 997 == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
        uint64_t consumed = 0;
        while (consumed < items) {
            const uint64_t available = produced.load(std::memory_order_acquire);
            if (available > consumed) {
                consumed = available;
                continue;
            }
            if (event.wait(200) == 0 && produced.load(std::memory_order_acquire) > consumed) ++timeouts;
        }
        producer.join();
        assert(timeouts == 0);
    }

    std::puts("event coalescing: ok");
    return 0;
}
