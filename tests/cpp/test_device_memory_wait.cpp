#include "nodes/hwaccel/device_memory_wait.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

int main() {
    int slept_ms = 0, attempts = 0;
    const auto sleep = [&](int ms) { slept_ms += ms; };

    // Memory there at once: no wait.
    assert(waitForDeviceMemory([&] { ++attempts; return true; }, sleep, 5000, [] { return false; }));
    assert(attempts == 1 && slept_ms == 0);

    // Memory there after three failed tries: the caller waited three steps and got its frame.
    attempts = 0;
    assert(waitForDeviceMemory([&] { return ++attempts > 3; }, sleep, 5000, [] { return false; }));
    assert(attempts == 4 && slept_ms == 3 * kDeviceMemoryWaitStepMs);

    // Never there: it gives up when the budget is spent, having tried throughout.
    attempts = 0; slept_ms = 0;
    assert(!waitForDeviceMemory([&] { ++attempts; return false; }, sleep, 200, [] { return false; }));
    assert(slept_ms == 200 && attempts == 200 / kDeviceMemoryWaitStepMs + 1);

    // A node asked to stop does not sit out the budget.
    attempts = 0; slept_ms = 0;
    assert(!waitForDeviceMemory([&] { ++attempts; return false; }, sleep, 5000, [&] { return attempts >= 2; }));
    assert(attempts == 2 && slept_ms == kDeviceMemoryWaitStepMs);

    // Anything that is not a shortage is the caller's error to handle, at once.
    attempts = 0; slept_ms = 0;
    try {
        waitForDeviceMemory([&]() -> bool { ++attempts; throw std::runtime_error("device lost"); }, sleep, 5000, [] { return false; });
        assert(!"an error other than a shortage was swallowed");
    } catch (const std::runtime_error &e) {
        assert(std::string(e.what()) == "device lost" && attempts == 1 && slept_ms == 0);
    }
    return 0;
}
