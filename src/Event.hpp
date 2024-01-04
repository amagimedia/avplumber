#pragma once
#include <sys/eventfd.h>
#include <poll.h>
#include <unistd.h>

class Event {
    friend class MultiEventWait;
private:
    int fd_;
public:
    Event(uint64_t initial = 0) {
        fd_ = eventfd(initial, EFD_NONBLOCK);
    }
    Event(const Event &copyfrom) = delete;
    Event(Event &&movefrom) {
        this->fd_ = movefrom.fd_;
        movefrom.fd_ = -1;
    }
    ~Event() {
        if (fd_<0) return;
        close(fd_);
        fd_ = -1;
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
            }
        }
        return val;
    }
    void signal(uint64_t val = 1) {
        write(this->fd_, &val, sizeof val);
    }
    int fd() const {
        return fd_;
    }
};
