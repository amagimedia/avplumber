#include "MultiEventWait.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <pthread.h>
#include <thread>
#include <vector>

void print_stack_trace() {}
static void ignore(int) {}

static bool signalled(Event &event) { return event.wait(0) != 0; }

int main() {
    // More events than one wait is likely to see fire: a compositor waits on one per source.
    std::vector<Event> events(200);
    Event stop, moved_from;
    Event taken(std::move(moved_from));   // moved_from now has no descriptor
    std::vector<Event *> list;
    for (auto &event : events) list.push_back(&event);
    list.push_back(&stop);
    list.push_back(&moved_from);          // tolerated, never fires
    list.push_back(&events[7]);           // listed twice, as wakeInputs() callers may build it
    MultiEventWait wait(list);

    // Nothing signalled: a zero timeout and a short one both report a timeout.
    assert(!wait.wait(0));
    assert(!wait.wait(5));

    // One signalled: reported, and drained so that the next wait sleeps again.
    events[7].signal();
    assert(wait.wait(0));
    assert(!wait.wait(0));
    assert(!signalled(events[7]));

    // Several signalled, one of them more than once: all drained by one wait.
    events[0].signal();
    events[150].signal();
    events[150].signal();
    events[199].signal();
    stop.signal();
    assert(wait.wait(-1));
    assert(!signalled(events[0]) && !signalled(events[150]) && !signalled(events[199]) && !signalled(stop));
    assert(!wait.wait(0));

    // Every event signalled at once is still one wake.
    for (auto &event : events) event.signal();
    assert(wait.wait(-1));
    for (auto &event : events) assert(!signalled(event));
    assert(!wait.wait(0));

    // A caller that asks is told which events fired, by their place in its list: a compositor
    // then looks at those sources only. An event listed twice is reported at its first place.
    std::vector<uint32_t> fired{99};
    assert(!wait.wait(0, &fired) && fired.empty());
    events[3].signal();
    events[150].signal();
    events[150].signal();
    stop.signal();
    assert(wait.wait(0, &fired));
    std::sort(fired.begin(), fired.end());
    assert((fired == std::vector<uint32_t>{3, 150, 200}));
    events[7].signal();
    assert(wait.wait(0, &fired) && (fired == std::vector<uint32_t>{7}));
    assert(!wait.wait(0, &fired) && fired.empty());

    // An event not in the list does not wake the wait and keeps its signal.
    taken.signal();
    assert(!wait.wait(0));
    assert(signalled(taken));

    // A signal from another thread ends a wait without a timeout.
    std::thread producer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        events[42].signal();
    });
    const auto started = std::chrono::steady_clock::now();
    assert(wait.wait(-1));
    assert(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(20));
    producer.join();
    assert(!wait.wait(0));

    // A timeout is honoured while nothing fires.
    const auto before = std::chrono::steady_clock::now();
    assert(!wait.wait(40));
    assert(std::chrono::steady_clock::now() - before >= std::chrono::milliseconds(35));

    // A signal during the wait (as when the process is stopped and resumed) makes epoll_wait fail
    // with EINTR: the wait goes on until its timeout, or until an event fires, instead of throwing.
    struct sigaction action = {};
    action.sa_handler = ignore;   // no SA_RESTART: epoll_wait is never restarted anyway
    sigaction(SIGUSR1, &action, nullptr);
    const pthread_t waiter = pthread_self();
    std::thread interrupter([waiter] {
        for (int i = 0; i < 5; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            pthread_kill(waiter, SIGUSR1);
        }
    });
    const auto interrupted = std::chrono::steady_clock::now();
    assert(!wait.wait(120));
    assert(std::chrono::steady_clock::now() - interrupted >= std::chrono::milliseconds(110));
    interrupter.join();
    std::thread late([&] {
        pthread_kill(waiter, SIGUSR1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        events[3].signal();
    });
    assert(wait.wait(-1));
    late.join();
    return 0;
}
