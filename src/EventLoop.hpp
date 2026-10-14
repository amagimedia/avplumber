#pragma once
#include "Event.hpp"
#include "util.hpp"
#include "avutils.hpp"
#include "instance_shared.hpp"
#include <concurrentqueue/concurrentqueue.h>
#include <atomic>
#include <deque>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <thread>
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
    std::map<int, Callable> todo_when_fd_readable_;
    std::mutex todo_when_fd_readable_busy_;
    std::list<std::pair<AVTS, Callable>> scheduled_;
    std::mutex scheduled_busy_;
    std::mutex busy_;
    bool debug_timing_ = false;
    AVTS debug_timing_tolerance_ = 2;
    bool executeSingleFromToDo() {
        Callable cb;
        if (!todo_.try_dequeue(cb)) {
            //logstream << "todo_ queue empty";
            return false;
        }
        //logstream << "executing something from todo_ queue";
        try {
            cb(*this);
        } catch (std::exception &e) {
            logstream << "error in event loop: " << e.what();
        }
        return true;
    }
    void threadFunction() {
        loop_thread_.store(std::this_thread::get_id(), std::memory_order_relaxed);
        while (should_work_) {
            passes_.fetch_add(1, std::memory_order_relaxed);
            struct pollfd *pfds = nullptr;
            size_t i = 0;
            size_t count = 0;
            {
                std::lock_guard<decltype(todo_when_fd_readable_busy_)> lock(todo_when_fd_readable_busy_);
                count = todo_when_fd_readable_.size()+1;
                pfds = new struct pollfd[count];
                for (auto &kv: todo_when_fd_readable_) {
                    pfds[i].fd = kv.first;
                    pfds[i].events = POLLIN;
                    pfds[i].revents = 0;
                    i++;
                }
            }
            pfds[i].fd = wakeup_.fd();
            pfds[i].events = POLLIN;
            pfds[i].revents = 0;
            int timeout_ms = -1;
            {
                std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
                if (!scheduled_.empty()) {
                    timeout_ms = scheduled_.front().first - wallclock.pts();
                    if (timeout_ms<0) {
                        if (debug_timing_ && (timeout_ms <= -debug_timing_tolerance_)) {
                            logstream << "should wait " << timeout_ms << "ms for scheduled, changing to 0";
                        }
                        timeout_ms = 0;
                    }
                }
            }
            AVTS before_poll = wallclock.pts();
            int ret = poll(pfds, count, timeout_ms);
            AVTS diff = timeout_ms>=0 ? (wallclock.pts() - before_poll - timeout_ms) : 0;
            if ( debug_timing_ && ( (ret==0 && abs(diff)>=debug_timing_tolerance_) || (diff>=debug_timing_tolerance_) ) ) {
                logstream << "kernel is cheating on us! poll returned after ms diff " << diff << " timeout_ms " << timeout_ms;
            }

            //logstream << "poll returned " << ret;
            if (!should_work_) {
                delete pfds;
                return;
            }
            {
                std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
                AVTS now = wallclock.pts();
                while (!scheduled_.empty()) {
                    if (scheduled_.front().first > now) {
                        break;
                    }
                    AVTS diff = scheduled_.front().first - now;
                    if (debug_timing_ && (diff <= -debug_timing_tolerance_)) {
                        logstream << "got scheduled too late diff " << diff << "ms";
                    }
                    todo_.enqueue(scheduled_.front().second);
                    scheduled_.pop_front();
                }
            }
            if (ret<0) {
                delete pfds;
                throw Error("poll error in event loop thread");
            } else if (ret>0) {
                int64_t blackhole;
                for (size_t i=0; i<(count-1); i++) {
                    if (pfds[i].revents & POLLIN) {
                        std::lock_guard<decltype(todo_when_fd_readable_busy_)> lock(todo_when_fd_readable_busy_);
                        auto cb_it = todo_when_fd_readable_.find(pfds[i].fd);
                        if (cb_it != todo_when_fd_readable_.end()) {
                            todo_.enqueue(cb_it->second);
                            todo_when_fd_readable_.erase(cb_it);
                            read(pfds[i].fd, &blackhole, sizeof blackhole);
                        }
                        pfds[i].revents = 0;
                    }
                }
                if (pfds[count-1].revents & POLLIN) {
                    // this is wakeup_
                    read(pfds[count-1].fd, &blackhole, sizeof blackhole);
                    pfds[count-1].revents = 0;
                }
            }
            delete pfds;
            {
                std::lock_guard<decltype(busy_)> lock(busy_);
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
        delegated_execution_thread_ = start_thread("EventLoop", [this]() {
            threadFunction();
        });
    }
    ~EventLoop() {
        should_work_ = false;
        wakeup_.signal();
        delegated_execution_thread_.join();
        Callable blackhole;
        if (todo_.try_dequeue(blackhole)) {
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
        todo_.enqueue(cb);
        wakeUnlessOnLoopThread();
    }
    int fastExecute(av::Timestamp time_limit, Callable cb) {
        av::Timestamp deadline = addTS(wallclock.ts(), time_limit);
        if (busy_.try_lock()) {
            std::lock_guard<decltype(busy_)> lock(busy_, std::adopt_lock);
            todo_.enqueue(cb);
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
        // TODO FIXME: this works only by accident!!!
        // it should be map of lists
        std::lock_guard<decltype(todo_when_fd_readable_busy_)> lock(todo_when_fd_readable_busy_);
        todo_when_fd_readable_[event.fd()] = cb;
        wakeUnlessOnLoopThread();
    }
    void schedule(av::Timestamp when, Callable cb) {
        AVTS ts = when.timestamp(wallclock.timeBase());
        std::lock_guard<decltype(scheduled_busy_)> lock(scheduled_busy_);
        auto it = scheduled_.begin();
        bool is_first = true;
        while (it != scheduled_.end()) {
            if (it->first > ts) {
                break;
            }
            is_first = false;
            it++;
        }
        if (debug_timing_) {
            AVTS diff = ts - wallclock.pts();
            if (diff <= -debug_timing_tolerance_) {
                logstream << "scheduling event from the past?! " << diff << " ms";
            }
        }
        scheduled_.insert(it, std::pair(ts, cb));
        if (is_first) {
            wakeUnlessOnLoopThread();
        }
    }
    void sleepAndExecute(int ms, Callable cb) {
        schedule(addTS(wallclock.ts(), av::Timestamp(ms, {1, 1000})), cb);
    }
};
