#include "mixer/primitives/frame_subscription.hpp"
#include <atomic>
#include <cassert>
#include <thread>

void print_stack_trace() {}

struct TestPTS {
    int64_t ns;
    bool isValid() const { return ns >= 0; }
    int64_t timestamp(av::Rational) const { return ns; }
};
struct TestFrame { TestPTS stamp; TestPTS pts() const { return stamp; } };

int main() {
    avp::mixer::FrameSubscription subscription;
    subscription.configure({30, 1});
    int count = 0;
    auto publish = [&](int64_t ns) { subscription.publish(TestFrame{{ns}}, [&] { ++count; }); };
    publish(0);
    assert(count == 0);
    subscription.enable(true);
    for (int i = 0; i < 60; ++i) publish(avp::mixer::TickGrid({60, 1}).time(i));
    assert(count == 30);
    subscription.enable(false);
    publish(2000000000);
    assert(count == 30);
    subscription.enable(true);
    publish(2000000000);
    assert(count == 31);

    std::atomic<bool> running{true};
    std::atomic<int> delivered{0};
    std::thread producer([&] {
        while (running) subscription.publish(TestFrame{{-1}}, [&] { ++delivered; });
    });
    for (int i = 0; i < 1000; ++i) {
        subscription.enable(false);
        const int before = delivered;
        std::this_thread::yield();
        assert(delivered == before); // disable acknowledges all prior publishing
        subscription.enable(true);
    }
    subscription.close();
    const int before = delivered;
    subscription.enable(true); // a racing composition must not reopen a stopped consumer
    std::this_thread::yield();
    running = false;
    producer.join();
    assert(delivered == before);
}
