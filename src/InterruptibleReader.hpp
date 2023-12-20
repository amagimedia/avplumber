#pragma once
#include "util.hpp"
#include "avutils.hpp"
#include <asm-generic/errno-base.h>
#include <string>
#include <fcntl.h>
#include <sys/eventfd.h>
#include <sys/poll.h>
#include <unistd.h>

class InterruptibleReader {
protected:
    std::string path_;
    int pipe_fd_ = -2;
    int event_fd_ = -2;
    struct pollfd pollfds_[2];
    static constexpr size_t PIPE_INDEX = 0;
    static constexpr size_t EVENT_INDEX = 1;
public:
    InterruptibleReader(const std::string path): path_(path) {
        event_fd_ = eventfd(1, 0);
        pollfds_[0].events = POLLIN;
        pollfds_[0].revents = 0;
        pollfds_[1].events = POLLIN;
        pollfds_[1].revents = 0;
        pollfds_[EVENT_INDEX].fd = event_fd_;
    }
    bool read(void* _buffer, size_t size) {
        uint8_t* buffer = (uint8_t*)_buffer;
        if (pipe_fd_ <= 0) {
            pipe_fd_ = open(path_.c_str(), O_RDONLY | O_NONBLOCK);
            if (pipe_fd_ <= 0) {
                if (errno==EAGAIN) {
                    logstream << "pipe " << path_ << " not ready to open";
                    wallclock.sleepms(500);
                    return false;
                }
                throw Error("pipe " + path_ + " open failed");
            }
            pollfds_[PIPE_INDEX].fd = pipe_fd_;
        }
        while (size > 0) {
            int ret = poll(pollfds_, 2, -1);
            if (ret<0) {
                throw Error("wait: poll error");
            }
            if (pollfds_[PIPE_INDEX].revents & POLLIN) {
                pollfds_[PIPE_INDEX].revents = 0;
                ret = ::read(pipe_fd_, buffer, size);
                if (ret > 0) {
                    size -= ret;
                    buffer += ret;
                } else if (ret < 0) {
                    logstream << "read failed, closing pipe";
                    close(pipe_fd_);
                    pipe_fd_ = -2;
                    return false;
                }
            } else if (pollfds_[EVENT_INDEX].revents & POLLIN) {
                pollfds_[EVENT_INDEX].revents = 0;
                int64_t blackhole;
                ::read(event_fd_, &blackhole, sizeof blackhole);
                return false;
            }
        }
        return true;
    }
    void interrupt() {
        uint64_t val = 1;
        write(event_fd_, &val, sizeof val);
    }
    ~InterruptibleReader() {
        if (pipe_fd_ > 0) {
            close(pipe_fd_);
        }
        close(event_fd_);
    }
};
