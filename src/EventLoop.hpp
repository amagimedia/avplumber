#pragma once
#include "Event.hpp"
#include "util.hpp"
#include "avutils.hpp"
#include "instance_shared.hpp"
#include <concurrentqueue/concurrentqueue.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/epoll.h>
#include <unistd.h>
#include <avcpp/timestamp.h>

class EventLoop;

using Callable = std::function<void(EventLoop&)>;

class EventLoop: public InstanceShared<EventLoop> {
protected:
    std::thread delegated_execution_thread_;
    Event wakeup_;
    std::atomic_bool should_work_ {true};
    std::atomic<std::thread::id> loop_thread_ {};
    std::atomic<uint64_t> passes_ {0};
    // Work, a wait or a timer added by a handler running on the loop's own thread needs no
    // wake-up: the loop takes its work, its waits and its timers afresh before it sleeps.
    void wakeUnlessOnLoopThread() {
        if (std::this_thread::get_id() != loop_thread_.load(std::memory_order_relaxed)) wakeup_.signal();
    }
    moodycamel::ConcurrentQueue<Callable> todo_;
    // Work in todo_, counted after each enqueue: the queue keeps one sub-queue per thread that ever
    // handed the loop work, and finding it empty means looking at all of them, after every pass.
    std::atomic<size_t> todo_count_{0};
    // Work the loop hands itself (a node yielding after each frame): only its own thread touches it.
    std::deque<Callable> local_todo_;
    // What to run when a descriptor becomes readable, once: the kernel holds the waits
    // (EPOLLONESHOT), so a pass costs what is ready, not what is waited for.
    int epoll_fd_ = -1;
    std::unordered_map<int, Callable> todo_when_fd_readable_;
    std::mutex todo_when_fd_readable_busy_;
    std::multimap<AVTS, Callable> scheduled_;   // by time; equal times keep the order they were set in
    std::mutex scheduled_busy_;
    std::mutex busy_;
    bool debug_timing_ = false;
    AVTS debug_timing_tolerance_ = 2;
    void run(Callable &cb) {
        try {
            cb(*this);
        } catch (std::exception &e) {
            logstream << "error in event loop: " << e.what();
        }
    }
    // Counted after the enqueue: a loop that reads the count before it changes has a wake-up coming.
    void enqueueToDo(Callable cb) {
        todo_.enqueue(std::move(cb));
        todo_count_.fetch_add(1, std::memory_order_release);
    }
    bool executeSingleFromToDo() {
        if (!local_todo_.empty()) {
            Callable cb = std::move(local_todo_.front());
            local_todo_.pop_front();
            run(cb);
            return true;
        }
        if (todo_count_.load(std::memory_order_acquire) == 0) return false;
        Callable cb;
        if (!todo_.try_dequeue(cb)) {
            //logstream << "todo_ queue empty";
            return false;
        }
        todo_count_.fetch_sub(1, std::memory_order_relaxed);
        //logstream << "executing something from todo_ queue";
        run(cb);
        return true;
    }
    // Readable once more wakes the loop once more. A descriptor closed since its last wait has
    // left the kernel's set by itself, and its number may be a new event's by now.
    void arm(int fd) {
        struct epoll_event wait = {};
        wait.events = EPOLLIN | EPOLLONESHOT;
        wait.data.fd = fd;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &wait) == 0) return;
        if (errno == ENOENT && epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &wait) == 0) return;
        if (errno == EEXIST && epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &wait) == 0) return;
        throw Error("event loop: cannot wait for descriptor " + std::to_string(fd) + ": " + strerror(errno));
    }
    void threadFunction() {
        loop_thread_.store(std::this_thread::get_id(), std::memory_order_relaxed);
        std::vector<struct epoll_event> ready(64);
        std::vector<Callable> due;
        while (should_work_) {
            passes_.fetch_add(1, std::memory_order_relaxed);
            int timeout_ms = -1;
            {
                std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
                if (!scheduled_.empty()) {
                    timeout_ms = scheduled_.begin()->first - wallclock.pts();
                    if (timeout_ms<0) {
                        if (debug_timing_ && (timeout_ms <= -debug_timing_tolerance_)) {
                            logstream << "should wait " << timeout_ms << "ms for scheduled, changing to 0";
                        }
                        timeout_ms = 0;
                    }
                }
            }
            AVTS before_wait = wallclock.pts();
            int ret = epoll_wait(epoll_fd_, ready.data(), static_cast<int>(ready.size()), timeout_ms);
            AVTS diff = timeout_ms>=0 ? (wallclock.pts() - before_wait - timeout_ms) : 0;
            if ( debug_timing_ && ( (ret==0 && abs(diff)>=debug_timing_tolerance_) || (diff>=debug_timing_tolerance_) ) ) {
                logstream << "kernel is cheating on us! epoll_wait returned after ms diff " << diff << " timeout_ms " << timeout_ms;
            }

            if (!should_work_) {
                return;
            }
            if (ret<0) {
                if (errno == EINTR) continue;
                throw Error("epoll_wait error in event loop thread");
            }
            {
                std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
                AVTS now = wallclock.pts();
                while (!scheduled_.empty()) {
                    auto first = scheduled_.begin();
                    if (first->first > now) {
                        break;
                    }
                    AVTS diff = first->first - now;
                    if (debug_timing_ && (diff <= -debug_timing_tolerance_)) {
                        logstream << "got scheduled too late diff " << diff << "ms";
                    }
                    due.push_back(std::move(first->second));
                    scheduled_.erase(first);
                }
            }
            for (int i=0; i<ret; i++) {
                const int fd = ready[i].data.fd;
                int64_t blackhole;
                if (read(fd, &blackhole, sizeof blackhole) < 0 && errno != EAGAIN) continue;   // closed meanwhile
                Event::consumed(fd);
                if (fd == wakeup_.fd()) continue;
                std::lock_guard<decltype(todo_when_fd_readable_busy_)> lock(todo_when_fd_readable_busy_);
                auto cb_it = todo_when_fd_readable_.find(fd);
                if (cb_it != todo_when_fd_readable_.end()) {
                    due.push_back(std::move(cb_it->second));
                    todo_when_fd_readable_.erase(cb_it);
                }
            }
            {
                std::lock_guard<decltype(busy_)> lock(busy_);
                for (Callable &cb: due) {
                    if (!should_work_) break;
                    run(cb);
                }
                due.clear();
                while (executeSingleFromToDo() && should_work_) { };
            }
        }
    }
public:
    EventLoop() {
        const char* envstr = getenv("AVPLUMBER_WARN_BAD_TIMING");
        debug_timing_ = envstr && envstr[0];
        if (debug_timing_) {
            debug_timing_tolerance_ = atoi(envstr);
        }
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) throw Error("event loop: epoll_create1 error");
        struct epoll_event wake = {};
        wake.events = EPOLLIN;
        wake.data.fd = wakeup_.fd();
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wakeup_.fd(), &wake) != 0) {
            close(epoll_fd_);
            throw Error("event loop: epoll_ctl error");
        }
        delegated_execution_thread_ = start_thread("EventLoop", [this]() {
            threadFunction();
        });
    }
    ~EventLoop() {
        should_work_ = false;
        wakeup_.signal();
        delegated_execution_thread_.join();
        close(epoll_fd_);
        Callable blackhole;
        if (!local_todo_.empty() || todo_.try_dequeue(blackhole)) {
            logstream << "still have something in todo queue when shutting down event loop";
        }
        if (!todo_when_fd_readable_.empty()) {
            logstream << "still have " << todo_when_fd_readable_.size() << " events for fd wakeup when shutting down event loop";
        }
        if (!scheduled_.empty()) {
            logstream << "still have " << scheduled_.size() << " events scheduled for timed execution when shutting down event loop";
        }
    }
    /// How many times the loop has gone to wait: a measure of what waking it costs.
    uint64_t passes() const {
        return passes_.load(std::memory_order_relaxed);
    }
    void execute(Callable cb) {
        // From the loop's own thread the work runs in this pass, after what is running now.
        if (std::this_thread::get_id() == loop_thread_.load(std::memory_order_relaxed)) {
            local_todo_.push_back(std::move(cb));
            return;
        }
        enqueueToDo(std::move(cb));
        wakeUnlessOnLoopThread();
    }
    int fastExecute(av::Timestamp time_limit, Callable cb) {
        av::Timestamp deadline = addTS(wallclock.ts(), time_limit);
        if (busy_.try_lock()) {
            std::lock_guard<decltype(busy_)> lock(busy_, std::adopt_lock);
            enqueueToDo(std::move(cb));
            int diff = std::numeric_limits<int>::min();
            while (executeSingleFromToDo()) {
                diff = addTSSameTB(rescaleTS(wallclock.ts(), {1, 1000}), rescaleTS(negateTS(deadline), {1, 1000}));
                if (diff > 0) {
                    wakeup_.signal();
                    return diff;
                }
            }
            return diff;
        } else {
            execute(cb);
            return std::numeric_limits<int>::min();
        }
    }
    void asyncWaitAndExecute(Event &event, Callable cb) {
        // One handler per event: a second wait for the same event replaces the first.
        // The kernel wakes a sleeping loop for a wait added from any thread.
        std::lock_guard<decltype(todo_when_fd_readable_busy_)> lock(todo_when_fd_readable_busy_);
        todo_when_fd_readable_[event.fd()] = std::move(cb);
        arm(event.fd());
    }
    void schedule(av::Timestamp when, Callable cb) {
        AVTS ts = when.timestamp(wallclock.timeBase());
        std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
        const bool is_first = scheduled_.empty() || scheduled_.begin()->first > ts;
        if (debug_timing_) {
            AVTS diff = ts - wallclock.pts();
            if (diff <= -debug_timing_tolerance_) {
                logstream << "scheduling event from the past?! " << diff << " ms";
            }
        }
        scheduled_.emplace(ts, std::move(cb));
        if (is_first) {
            wakeUnlessOnLoopThread();
        }
    }
    void sleepAndExecute(int ms, Callable cb) {
        schedule(addTS(wallclock.ts(), av::Timestamp(ms, {1, 1000})), cb);
    }
};
