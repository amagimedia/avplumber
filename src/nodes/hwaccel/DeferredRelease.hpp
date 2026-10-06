#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

// Reserve capacity on the producer before allocating. The last consumer only
// links an already allocated job. A producer denied capacity drops its frame
// instead of waiting with buffers retained. Budgets include live and retired jobs.
template<class T> class DeferredRelease : public std::enable_shared_from_this<DeferredRelease<T>> {
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
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::thread worker_;
public:
    DeferredRelease(): worker_([state = state_] {
        std::unique_lock<std::mutex> lock(state->mutex);
        for (;;) {
            state->changed.wait(lock, [&] { return state->head || (state->stopping && !state->outstanding); });
            if (!state->head) return;
            auto *job = state->head;
            state->head = job->next;
            if (!state->head) state->tail = nullptr;
            auto budget = job->budget;
            lock.unlock();
            delete job;
            lock.lock();
            --budget->used;
            --state->outstanding;
            --state->pending;
            state->changed.notify_all();
        }
    }) {}
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
        std::lock_guard<std::mutex> lock(state_->mutex);
        budget->closed = true;
    }
    std::pair<size_t, size_t> counts() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return {state_->outstanding, state_->pending};
    }
    size_t declined(const std::shared_ptr<Budget>& budget) const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return budget->declined;
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
