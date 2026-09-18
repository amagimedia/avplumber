#include "TransitionScheduler.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace avp::mixer {

struct TransitionScheduler::Impl {
    struct Task {
        std::chrono::steady_clock::time_point when;
        uint64_t seq = 0;
        std::string label;
        std::function<void()> run;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Task> tasks;
    bool stopping = false;
    uint64_t next_seq = 0;
    std::thread worker;

    static bool before(const Task& a, const Task& b) {
        if (a.when != b.when)
            return a.when < b.when;
        return a.seq < b.seq;
    }

    void workerLoop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            if (tasks.empty()) {
                if (stopping)
                    break;
                cv.wait(lock, [&] { return stopping || !tasks.empty(); });
                continue;
            }

            auto next = std::min_element(tasks.begin(), tasks.end(), before);
            auto now = std::chrono::steady_clock::now();
            if (next->when > now) {
                cv.wait_until(lock, next->when);
                continue;
            }

            Task task = std::move(*next);
            tasks.erase(next);
            lock.unlock();
            try {
                task.run();
            } catch (const std::exception& e) {
                logstream << "mixer transition task " << task.label << " failed: " << e.what();
            } catch (...) {
                logstream << "mixer transition task " << task.label << " failed with unknown exception";
            }
            lock.lock();
        }
    }
};

TransitionScheduler::TransitionScheduler()
    : impl_(std::make_unique<Impl>()) {
    impl_->worker = start_thread("mixer transitions", [impl = impl_.get()] {
        impl->workerLoop();
    });
}

TransitionScheduler::~TransitionScheduler() {
    shutdown();
}

void TransitionScheduler::post(std::string label, std::function<void()> task) {
    postAfter(std::move(label), 0, std::move(task));
}

void TransitionScheduler::postAfter(std::string label, int64_t delay_ms, std::function<void()> task) {
    if (!task)
        throw Error("mixer transition scheduler: empty task");

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping)
        throw Error("mixer transition scheduler is stopped");

    Impl::Task queued;
    queued.when = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(std::max<int64_t>(0, delay_ms));
    queued.seq = impl_->next_seq++;
    queued.label = std::move(label);
    queued.run = std::move(task);
    impl_->tasks.push_back(std::move(queued));
    impl_->cv.notify_one();
}

void TransitionScheduler::shutdown() {
    if (!impl_)
        return;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
        impl_->tasks.clear();
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable())
        impl_->worker.join();
}

}  // namespace avp::mixer
