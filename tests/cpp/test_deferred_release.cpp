#include "nodes/hwaccel/DeferredRelease.hpp"
#include <cassert>
#include <future>
#include <stdexcept>

struct Resource {
    std::promise<void>& entered;
    std::shared_future<void> finish;
    std::thread::id& destruction_thread;
    Resource(std::promise<void>& entered, std::shared_future<void> finish,
             std::thread::id& destruction_thread, bool fail = false)
        : entered(entered), finish(std::move(finish)), destruction_thread(destruction_thread) {
        if (fail) throw std::runtime_error("allocation failed");
    }
    ~Resource() {
        destruction_thread = std::this_thread::get_id();
        entered.set_value();
        finish.wait();
    }
};

void lifetime_and_stop() {
    using namespace std::chrono_literals;
    auto cleanup = std::make_shared<DeferredRelease<Resource>>();
    auto budget = std::make_shared<DeferredRelease<Resource>::Budget>(1);
    std::promise<void> entered, finish, second_entered, second_finish;
    auto ready = entered.get_future();
    std::thread::id destroyed_on, second_destroyed_on;
    auto done = finish.get_future().share();
    // A failed construction must return its reservation.
    try { cleanup->tryMake(budget, entered, done, destroyed_on, true); assert(false); }
    catch (const std::runtime_error&) {}
    auto frame = cleanup->tryMake(budget, entered, done, destroyed_on);
    auto alias = frame;
    frame.reset();
    assert(ready.wait_for(20ms) == std::future_status::timeout);
    alias.reset(); // must return before the destructor is allowed to finish
    assert(ready.wait_for(1s) == std::future_status::ready);
    assert(destroyed_on != std::this_thread::get_id());
    auto second_done = second_finish.get_future().share();
    auto other_budget = std::make_shared<DeferredRelease<Resource>::Budget>(1);
    // Rejected admission must return even while cleanup is blocked in a destructor.
    // It must not construct a resource or consume another producer's reservation.
    assert(!cleanup->tryMake(other_budget, second_entered, second_done, second_destroyed_on));
    assert(cleanup->counts().first == 1 && cleanup->counts().second == 1);
    assert(cleanup->diagnostics(other_budget).declined == 1);
    finish.set_value();
    for (int i = 0; i < 1000 && cleanup->counts().first; ++i) std::this_thread::sleep_for(1ms);
    assert(cleanup->counts().first == 0);
    auto second = cleanup->tryMake(other_budget, second_entered, second_done, second_destroyed_on);
    assert(second);
    assert(cleanup->diagnostics(other_budget).released == 1);
    // A live frame also consumes its producer's budget, without pending cleanup.
    assert(!cleanup->tryMake(other_budget, entered, done, destroyed_on));
    assert(cleanup->diagnostics(other_budget).declined == 2);
    cleanup->close(other_budget);
    assert(!cleanup->tryMake(other_budget, entered, done, destroyed_on));
    // Instance teardown may release the service before edges release frames.
    // Neither that teardown nor the last frame release may wait for GPU cleanup.
    std::weak_ptr<DeferredRelease<Resource>> service = cleanup;
    cleanup.reset();
    second.reset();
    assert(second_entered.get_future().wait_for(1s) == std::future_status::ready);
    assert(second_destroyed_on == destroyed_on);
    second_finish.set_value();
    for (int i = 0; i < 1000 && !service.expired(); ++i) std::this_thread::sleep_for(1ms);
    assert(service.expired());
}

struct Stamp {
    using Clock = std::chrono::steady_clock;
    std::promise<Clock::time_point>& released;
    explicit Stamp(std::promise<Clock::time_point>& released): released(released) {}
    ~Stamp() { released.set_value(Clock::now()); }
};

void pacing() {
    using namespace std::chrono_literals;
    auto cleanup = std::make_shared<DeferredRelease<Stamp>>(50ms);
    auto budget = std::make_shared<DeferredRelease<Stamp>::Budget>(3);
    std::promise<Stamp::Clock::time_point> first, second, third;
    auto a = cleanup->tryMake(budget, first);
    auto b = cleanup->tryMake(budget, second);
    auto c = cleanup->tryMake(budget, third);
    a.reset();
    b.reset();
    auto start = first.get_future().get();
    auto end = second.get_future().get();
    assert(end - start >= 50ms);
    // Stop cancels pacing as well as blocked admissions.
    cleanup->close(budget);
    c.reset();
    assert(third.get_future().wait_for(1s) == std::future_status::ready);
    for (int i = 0; i < 1000 && cleanup->counts().first; ++i) std::this_thread::sleep_for(1ms);
    assert(cleanup->counts().first == 0);
}

int main() {
    lifetime_and_stop();
    pacing();
}
