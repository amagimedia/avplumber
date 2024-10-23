#pragma once
#include <atomic>
#include <optional>
#include "instance_shared.hpp"
#include "graph_interfaces.hpp"

class PauseControlTeam: public InstanceShared<PauseControlTeam> {
protected:
    std::atomic_bool paused_ {false};
    std::optional<StreamTarget> pause_at_;
    std::vector<std::weak_ptr<IInputReset>> nodes_to_resume_;
    std::mutex mutex_;
    std::weak_ptr<IStreamsInput> input_;
    std::weak_ptr<IFlushAndSeek> sync_obj_;
public:
    bool isPaused() {
        return paused_;
    }
    void pause() {
        paused_ = true;

        auto obj = sync_obj_.lock();
        if (obj) {
            obj->flushAndSeek(StreamTarget::from_frames_relative(0));
        }
    }
    void pause(const StreamTarget& target) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        auto input = input_.lock();
        if (input) {
            StreamTarget ts = target;
            input->fixInputTimestamp(ts);
            pause_at_ = ts;
        } else {
            pause_at_ = target;
        }
    }
    bool checkPause(const av::Timestamp& ts) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        if (pause_at_.has_value()) {
            if (ts >= pause_at_.value().ts) {
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
    void setInputNode(std::weak_ptr<IStreamsInput>& node) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        input_ = node;
    }
    void setSyncObj(std::weak_ptr<IFlushAndSeek> obj) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        sync_obj_ = obj;
    }
};
