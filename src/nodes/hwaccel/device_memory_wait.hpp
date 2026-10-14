#pragma once
// A device short of memory for a moment (every source of a large show starting at once) is not
// a reason for a node to fail: it tries again for a bounded time.

constexpr int kDeviceMemoryWaitStepMs = 10;

/// Calls `attempt` (true: it got its memory, false: the device had none) until it succeeds,
/// `budget_ms` of waiting is spent or `stopping` says the node is going away; `sleep` waits the
/// given milliseconds between tries. Returns whether an attempt succeeded. Anything `attempt`
/// throws passes through.
template <typename Attempt, typename Sleep, typename Stopping>
bool waitForDeviceMemory(Attempt &&attempt, Sleep &&sleep, int budget_ms, Stopping &&stopping) {
    for (int waited_ms = 0;; waited_ms += kDeviceMemoryWaitStepMs) {
        if (attempt()) return true;
        if (waited_ms >= budget_ms || stopping()) return false;
        sleep(kDeviceMemoryWaitStepMs);
    }
}
