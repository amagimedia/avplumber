#pragma once
#include <atomic>
#include <optional>
#include "instance_shared.hpp"
#include "graph_interfaces.hpp"

class PauseControlTeam: public InstanceShared<PauseControlTeam> {
protected:
    std::atomic_bool paused_ {false};
    std::optional<av::Timestamp> pause_at_;
    std::vector<std::weak_ptr<IInputReset>> nodes_to_resume_;
    std::mutex mutex_;
public:
    bool isPaused() {
        return paused_;
    }
    void pause() {
        paused_ = true;
    }
    void pause(const av::Timestamp& ts) {
        pause_at_ = ts;
    }
    bool checkPause(const av::Timestamp& ts) {
        if (pause_at_.has_value()) {
            if (ts >= pause_at_.value()) {
                pause_at_.reset();
                pause();
                return true;
            }
        }
        return false;
    }
    void resume() {
        paused_ = false;
        {
            std::lock_guard<decltype(mutex_)> lock(mutex_);
            for (auto &wptr: nodes_to_resume_) {
                std::shared_ptr<IInputReset> node = wptr.lock();
                if (node) {
                    node->resetInput();
                }
            }
        }
    }
    void addNode(std::weak_ptr<IInputReset> node) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        // TODO DRY: TickSource::add
        nodes_to_resume_.erase(std::remove_if(nodes_to_resume_.begin(), nodes_to_resume_.end(), [](std::weak_ptr<IInputReset> &p) { return p.expired(); }), nodes_to_resume_.end());
        nodes_to_resume_.push_back(std::weak_ptr<IInputReset>(node));
    }
};
