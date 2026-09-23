#pragma once

#include "../../../deps/dma-browser/addons/fdpass/dmabuf_ack.h"
#include <array>
#include <atomic>
#include <cerrno>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

// One queue owns one connection. Frames retain this queue after their receiver
// is stopped/reconnected, so neither the socket nor its ACKs can outlive/reach
// the wrong generation. Drained is sent only after the last frame lets go.
class DmabufReleaseAckQueue {
    int socket_fd_;
    int wake_fd_;
    mutable std::mutex mutex_;
    std::deque<std::array<uint8_t, DMABUF_RELEASE_ACK_BYTES>> pending_;
    size_t front_offset_ = 0;
    bool retired_ = false;
    std::atomic<bool> interrupted_{false};

    void append(uint64_t frame, DmabufAckKind kind) {
        pending_.emplace_back();
        dmabufEncodeReleaseAck(pending_.back().data(), frame, kind);
    }
    void wake() const {
        const uint64_t value = 1;
        (void)::write(wake_fd_, &value, sizeof(value));
    }

public:
    explicit DmabufReleaseAckQueue(int socket_fd = -1)
        : socket_fd_(socket_fd), wake_fd_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
        if (wake_fd_ < 0) {
            if (socket_fd_ >= 0) ::close(socket_fd_);
            throw std::runtime_error("eventfd() failed for DMA-BUF release acknowledgements");
        }
    }
    ~DmabufReleaseAckQueue() {
        if (socket_fd_ >= 0) {
            retire();
            append(0, DmabufAckKind::Drained);
            // Never block a frame destructor. If the peer cannot receive the
            // complete drain marker, EOF makes it quarantine instead of reuse.
            flush();
            ::close(socket_fd_);
        }
        ::close(wake_fd_);
    }
    int wakeFd() const { return wake_fd_; }
    bool interrupted() const { return interrupted_.load(); }
    void interrupt() { interrupted_.store(true); wake(); }
    void drainWakeFd() const {
        uint64_t value;
        while (::read(wake_fd_, &value, sizeof(value)) == sizeof(value)) {}
    }
    bool hasPending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !pending_.empty();
    }
    void enqueue(uint64_t frame) {
        bool retired;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            append(frame, DmabufAckKind::Frame);
            retired = retired_;
        }
        if (retired) flush(); // The receiver no longer services this connection.
        else wake();
    }
    void retire() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (retired_ || socket_fd_ < 0) return;
            retired_ = true;
            append(0, DmabufAckKind::Drain);
        }
        flush();
    }
    bool flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty()) {
            const auto &ack = pending_.front();
            const ssize_t sent = ::send(socket_fd_, ack.data() + front_offset_,
                ack.size() - front_offset_, MSG_DONTWAIT | MSG_NOSIGNAL);
            if (sent > 0) {
                front_offset_ += static_cast<size_t>(sent);
                if (front_offset_ == ack.size()) { pending_.pop_front(); front_offset_ = 0; }
            } else if (sent < 0 && errno == EINTR) {
                continue;
            } else {
                return sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
            }
        }
        return true;
    }
};
