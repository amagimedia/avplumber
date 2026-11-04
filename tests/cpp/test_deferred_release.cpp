#include "nodes/hwaccel/DeferredRelease.hpp"
#include <cassert>
#include <chrono>
#include <future>
#include <stdexcept>
#include <vector>

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
    // A producer's retired job, even blocked in its destructor, refuses no other producer.
    auto second = cleanup->tryMake(other_budget, second_entered, second_done, second_destroyed_on);
    assert(second);
    assert(cleanup->counts().first == 2 && cleanup->counts().second == 1);
    assert(cleanup->declined(other_budget) == 0);
    // Rejected admission must return while cleanup is blocked, and consume nothing.
    // Its own live frame counts against a producer's budget.
    assert(!cleanup->tryMake(other_budget, entered, done, destroyed_on));
    assert(cleanup->counts().first == 2);
    assert(cleanup->declined(other_budget) == 1);
    finish.set_value();
    for (int i = 0; i < 1000 && cleanup->counts().first > 1; ++i) std::this_thread::sleep_for(1ms);
    assert(cleanup->counts().first == 1);
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

void pace_rests_between_releases() {
    using namespace std::chrono_literals;
    using Clock = std::chrono::steady_clock;
    struct Stamped {
        std::vector<Clock::time_point> &times;
        explicit Stamped(std::vector<Clock::time_point> &times): times(times) {}
        ~Stamped() { times.push_back(Clock::now()); }
    };
    std::vector<Clock::time_point> times;
    auto cleanup = std::make_shared<DeferredRelease<Stamped>>(40ms);
    auto budget = std::make_shared<DeferredRelease<Stamped>::Budget>(3);
    auto first = cleanup->tryMake(budget, times);
    auto second = cleanup->tryMake(budget, times);
    assert(first && second);
    first.reset();
    second.reset();
    for (int i = 0; i < 1000 && cleanup->counts().first; ++i) std::this_thread::sleep_for(1ms);
    assert(cleanup->counts().first == 0 && times.size() == 2);
    assert(times[1] - times[0] >= 35ms);
    cleanup.reset();   // stopping does not wait out a rest
}

int main() {
    lifetime_and_stop();
    pace_rests_between_releases();
}
