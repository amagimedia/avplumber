#pragma once
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>

class Event {
    friend class MultiEventWait;
private:
    int fd_;
    bool coalesce_ = false;
    // One flag per descriptor number: set by a coalescing signal() that wrote, cleared by the
    // reader after its read(). Numbers past the table are never coalesced.
    static constexpr int kFlagCount = 1 << 20;
    static std::atomic<uint8_t>* written() {
        static std::atomic<uint8_t> flags[kFlagCount];
        return flags;
    }
public:
    // A coalescing event writes its descriptor only when the last write has been read: a queue
    // that signals on every push and pop then costs a system call per wake of the other side,
    // not per frame. Whoever waits re-checks what it waits for after reading, so a signal that
    // found the last write unread is seen by that check.
    enum Coalesce { coalesce };
    Event(uint64_t initial = 0) {
        fd_ = eventfd(initial, EFD_NONBLOCK);
        if (fd_ >= 0 && fd_ < kFlagCount) written()[fd_].store(0, std::memory_order_relaxed);
    }
    Event(Coalesce, uint64_t initial = 0): Event(initial) {
        coalesce_ = true;
    }
    Event(const Event &copyfrom) = delete;
    Event(Event &&movefrom) {
        this->fd_ = movefrom.fd_;
        this->coalesce_ = movefrom.coalesce_;
        movefrom.fd_ = -1;
    }
    ~Event() {
        if (fd_<0) return;
        close(fd_);
        fd_ = -1;
    }
    // Every reader of an event's descriptor calls this after its read(): the next signal writes.
    static void consumed(int fd) {
        if (fd >= 0 && fd < kFlagCount) written()[fd].exchange(0, std::memory_order_acq_rel);
    }
    uint64_t wait(int timeout_ms = -1, bool poll_only = false) {
        struct pollfd pfd;
        int64_t val = 0;
        pfd.fd = this->fd_;
        pfd.events = POLLIN;
        pfd.revents = 0;
        //do {
        poll(&pfd, 1, timeout_ms);
        //} while(!(pfd.revents & POLLIN));
        if (!poll_only) {
            if (pfd.revents & POLLIN) {
                read(this->fd_, &val, sizeof val);
                consumed(this->fd_);
            }
        }
        return val;
    }
    void signal(uint64_t val = 1) {
        if (coalesce_ && fd_ >= 0 && fd_ < kFlagCount && written()[fd_].exchange(1, std::memory_order_acq_rel)) return;
        write(this->fd_, &val, sizeof val);
    }
    int fd() const {
        return fd_;
    }
};
