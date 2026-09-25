#pragma once

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

// Reserve capacity on the producer before allocating. The last consumer only
// links an already allocated job. A producer denied capacity drops its frame
// instead of waiting with buffers retained. Budgets include live and retired jobs.
template<class T> class DeferredRelease : public std::enable_shared_from_this<DeferredRelease<T>> {
    using Clock = std::chrono::steady_clock;
    static double milliseconds(Clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    }
public:
    struct Budget {
        explicit Budget(size_t limit): limit(limit) {}
        const size_t limit;
        size_t used = 0; // protected by State::mutex
        bool closed = false;
        size_t declined = 0;
    };
private:
    struct Job {
        // On last release, ownership moves from the frame's deleter to its job.
        // Destruction order keeps the worker alive until T has been destroyed.
        std::shared_ptr<DeferredRelease> cleanup;
        T value;
        std::shared_ptr<Budget> budget;
        Job *next = nullptr;
        template<class... Args> Job(std::shared_ptr<Budget> budget, Args&&... args)
            : value(std::forward<Args>(args)...), budget(std::move(budget)) {}
    };
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        Job *head = nullptr, *tail = nullptr;
        size_t outstanding = 0;
        size_t pending = 0; // queued jobs plus the destructor currently running
        bool stopping = false;
        Clock::time_point releasing_since{};
        size_t released = 0;
        double release_ms = 0, max_release_ms = 0;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    const std::chrono::microseconds interval_;
    std::thread worker_;
public:
    explicit DeferredRelease(std::chrono::microseconds interval = std::chrono::microseconds(0))
        : interval_(interval), worker_([state = state_, interval] {
        auto next_release = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(state->mutex);
        for (;;) {
            state->changed.wait(lock, [&] { return state->head || (state->stopping && !state->outstanding); });
            if (!state->head) return;
            // Yield between complete releases, even when one release overran its
            // interval. Never catch up with a burst or delay a closing producer.
            if (interval.count() > 0) {
                state->changed.wait_until(lock, next_release, [&] {
                    return state->stopping || state->head->budget->closed;
                });
            }
            auto *job = state->head;
            state->head = job->next;
            if (!state->head) state->tail = nullptr;
            auto budget = job->budget;
            state->releasing_since = Clock::now();
            lock.unlock();
            delete job;
            lock.lock();
            const double elapsed = milliseconds(Clock::now() - state->releasing_since);
            state->release_ms += elapsed;
            state->max_release_ms = std::max(state->max_release_ms, elapsed);
            ++state->released;
            state->releasing_since = {};
            --budget->used;
            --state->outstanding;
            --state->pending;
            next_release = std::chrono::steady_clock::now() + interval;
            state->changed.notify_all();
        }
    }) {}
    std::chrono::microseconds interval() const { return interval_; }
    ~DeferredRelease() {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->stopping = true;
        }
        state_->changed.notify_all();
        // The final job can own the last service reference. Its thread keeps
        // State alive and exits after accounting for that completed job.
        if (worker_.get_id() == std::this_thread::get_id()) worker_.detach();
        else worker_.join();
    }
    void close(const std::shared_ptr<Budget>& budget) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            budget->closed = true;
        }
        state_->changed.notify_all();
    }
    std::pair<size_t, size_t> counts() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return {state_->outstanding, state_->pending};
    }
    struct Diagnostics {
        size_t declined;
        size_t released;
        double releasing_ms, release_ms, max_release_ms;
    };
    Diagnostics diagnostics(const std::shared_ptr<Budget>& budget) const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto now = Clock::now();
        auto age = [&](Clock::time_point since) { return since == Clock::time_point{} ? 0 : milliseconds(now - since); };
        return {budget->declined, state_->released, age(state_->releasing_since),
                state_->release_ms, state_->max_release_ms};
    }
    template<class... Args> std::shared_ptr<T> tryMake(const std::shared_ptr<Budget>& budget, Args&&... args) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (budget->closed) return {};
            // Keep memory admission conservative without retaining a realtime
            // input while another producer's retired allocations are released.
            if (state_->pending || budget->used >= budget->limit) {
                ++budget->declined;
                return {};
            }
            ++budget->used;
            ++state_->outstanding;
        }
        Job *job;
        try { job = new Job(budget, std::forward<Args>(args)...); }
        catch (...) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            --budget->used;
            --state_->outstanding;
            state_->changed.notify_all();
            throw;
        }
        return std::shared_ptr<T>(&job->value, [state = state_, cleanup = this->shared_from_this(), job](T*) noexcept {
            job->cleanup = cleanup;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->tail) state->tail->next = job;
                else state->head = job;
                state->tail = job;
                ++state->pending;
            }
            state->changed.notify_all();
        });
    }
};
