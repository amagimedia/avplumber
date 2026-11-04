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
//
// With a hold of N, the sender always has N frames out with this consumer: a frame let go
// waits until more than N are out, which happens when the next frame arrives, so one buffer
// goes back for each one the sender hands out. A consumer that holds more for a while (a
// stall) takes the sender to its own limit, and the frames it then lets go in a bunch wait
// instead of going back together. The sender's buffer pool (Chromium's capture pool keeps two
// idle buffers and destroys the rest, and allocates when it has none) then stays on a fixed
// set of N plus its own few. Retiring a connection still returns everything at once.
class DmabufReleaseAckQueue {
    int socket_fd_;
    int wake_fd_;
    mutable std::mutex mutex_;
    std::deque<std::array<uint8_t, DMABUF_RELEASE_ACK_BYTES>> pending_;
    std::deque<uint64_t> queued_;   // let go by the consumer, waiting until more than hold_ are out
    size_t hold_ = 0;               // frames kept out with this consumer; 0 returns each frame at once
    size_t outstanding_ = 0;        // arrived and not yet acknowledged
    size_t front_offset_ = 0;
    bool retired_ = false;
    uint64_t released_frames_ = 0, sent_messages_ = 0;
    std::atomic<bool> interrupted_{false};

    void append(uint64_t frame, DmabufAckKind kind) {
        pending_.emplace_back();
        dmabufEncodeReleaseAck(pending_.back().data(), frame, kind);
    }
    // Acknowledge waiting frames while more than hold_ are out. Caller holds mutex_.
    bool returnBeyondHold() {
        bool sent = false;
        while (outstanding_ > hold_ && !queued_.empty()) {
            append(queued_.front(), DmabufAckKind::Frame);
            queued_.pop_front();
            --outstanding_;
            sent = true;
        }
        return sent;
    }
    void wake() const {
        const uint64_t value = 1;
        (void)::write(wake_fd_, &value, sizeof(value));
    }

public:
    explicit DmabufReleaseAckQueue(int socket_fd = -1, size_t hold = 0)
        : socket_fd_(socket_fd), wake_fd_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)), hold_(hold) {
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
    struct Stats { uint64_t released, sent; size_t pending, queued, outstanding; };
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {released_frames_, sent_messages_, pending_.size(), queued_.size(), outstanding_};
    }
    void enqueue(uint64_t frame) {
        bool retired, send = true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++released_frames_;
            retired = retired_;
            if (retired || !hold_) {
                append(frame, DmabufAckKind::Frame);
                if (outstanding_) --outstanding_;
            } else {
                queued_.push_back(frame);
                send = returnBeyondHold();
            }
        }
        if (retired) flush(); // The receiver no longer services this connection.
        else if (send) wake();
    }
    /// The sender handed out a buffer for a new frame: one more is out, so one waiting
    /// frame may go back. Called by the receiver, which then flushes.
    void frameArrived() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (retired_) return;
        ++outstanding_;
        returnBeyondHold();
    }
    void retire() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (retired_ || socket_fd_ < 0) return;
            retired_ = true;
            for (const uint64_t frame : queued_) append(frame, DmabufAckKind::Frame);
            queued_.clear();
            outstanding_ = 0;
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
                if (front_offset_ == ack.size()) {
                    pending_.pop_front(); front_offset_ = 0;
                    ++sent_messages_;
                }
            } else if (sent < 0 && errno == EINTR) {
                continue;
            } else {
                return sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
            }
        }
        return true;
    }
};
