#pragma once

// Waiting for a DMA-BUF frame's acquire fence: a Linux sync_file fd whose poll(POLLIN) returns
// once the GPU work that produced the frame has completed. ipc_dmabuf_source waits on it before
// it forwards the frame, so nothing downstream reads pixels that are still being drawn. Nothing
// here depends on avplumber: tests/cpp/test_dmabuf_fence.cpp drives it with an eventfd in the
// fence's place.

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <poll.h>
#include <time.h>

enum class FenceWaitOutcome { Signalled, TimedOut, Interrupted, Invalid };

struct FenceWaitResult {
    FenceWaitOutcome outcome;
    int64_t waited_us;   // spent in the wait
    bool blocked;        // the fence had not signalled when the wait began
};

inline int64_t fenceMonotonicUs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

// Waits up to `timeout_ms` for `fence_fd`. `wake_fd` (-1 for none) is polled too: when it is
// readable, `on_wake()` runs (it should drain the descriptor) and ends the wait with Interrupted
// when it returns true. Only POLLIN proves completion: a descriptor that cannot be polled, or
// reports an error or hang-up without it, is Invalid, and the frame must not be forwarded.
template<class OnWake>
FenceWaitResult waitForFence(int fence_fd, int timeout_ms, int wake_fd, OnWake on_wake) {
    const int64_t started = fenceMonotonicUs();
    const auto done = [&](FenceWaitOutcome outcome, bool blocked) {
        return FenceWaitResult{outcome, fenceMonotonicUs() - started, blocked};
    };
    if (fence_fd < 0) return done(FenceWaitOutcome::Invalid, false);
    pollfd fds[2] = {{fence_fd, POLLIN, 0}, {wake_fd, POLLIN, 0}};
    const nfds_t count = wake_fd >= 0 ? 2 : 1;
    // A fence that has signalled already costs one poll and counts as no wait.
    bool blocked = false;
    for (;;) {
        const int64_t elapsed_ms = (fenceMonotonicUs() - started) / 1000;
        const int remaining_ms = !blocked ? 0 : timeout_ms > elapsed_ms ? int(timeout_ms - elapsed_ms) : 0;
        fds[0].revents = fds[1].revents = 0;
        const int ready = poll(fds, count, remaining_ms);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return done(FenceWaitOutcome::Invalid, blocked);
        }
        if (fds[0].revents & POLLIN) return done(FenceWaitOutcome::Signalled, blocked);
        if (fds[0].revents) return done(FenceWaitOutcome::Invalid, blocked);
        if (count == 2 && fds[1].revents && on_wake()) return done(FenceWaitOutcome::Interrupted, blocked);
        if (ready == 0 && blocked) return done(FenceWaitOutcome::TimedOut, blocked);
        blocked = true;
    }
}

inline FenceWaitResult waitForFence(int fence_fd, int timeout_ms) {
    return waitForFence(fence_fd, timeout_ms, -1, [] { return false; });
}

// Counters a receiver reports: how many frames came with a fence and without, how many waits
// found the fence still pending, the time spent waiting, and the frames dropped because the
// fence never signalled or could not be waited on.
struct FenceWaitStats {
    std::atomic<uint64_t> fenced{0}, plain{0}, blocked{0}, wait_us{0}, max_wait_us{0}, timeouts{0}, invalid{0};
    void record(const FenceWaitResult &r) {
        ++fenced;
        if (r.blocked) ++blocked;
        wait_us += uint64_t(r.waited_us);
        uint64_t seen = max_wait_us.load();
        while (uint64_t(r.waited_us) > seen && !max_wait_us.compare_exchange_weak(seen, uint64_t(r.waited_us))) {}
        if (r.outcome == FenceWaitOutcome::TimedOut) ++timeouts;
        else if (r.outcome == FenceWaitOutcome::Invalid) ++invalid;
    }
    void unfenced() { ++plain; }
};
