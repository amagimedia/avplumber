#pragma once
#include <atomic>
#include <optional>
#include <condition_variable>
#include "instance_shared.hpp"
#include "graph_interfaces.hpp"

class PauseControlTeam: public InstanceShared<PauseControlTeam>, public ILinkableTeam<PauseControlTeam> {
protected:
    std::atomic_bool paused_ {false};
    std::optional<StreamTarget> pause_at_;
    std::vector<std::weak_ptr<IInputReset>> nodes_to_resume_;
    std::mutex mutex_;
    std::weak_ptr<IPlaybackControl> playback_;
    std::vector<std::weak_ptr<IFlushAndSeek>> sync_objs_;
    std::condition_variable cv_;

    void pauseNonRecursive(bool synchonize_nodes = true) {
        paused_ = true;
        if (synchonize_nodes) {
            for (auto& p: sync_objs_) {
                auto obj = p.lock();
                if (obj) {
                    obj->flushAndSeek(StreamTarget::from_frames_relative(0));
                }
            }
        }
    }
    void pauseNonRecursive(const StreamTarget& target) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        auto playback = playback_.lock();
        if (playback) {
            StreamTarget ts = target;
            playback->convertStreamTarget(ts, StreamTarget::ETargetType::tt_Timestamp);
            pause_at_ = ts;
        }
    }
    void resumeNonRecursive() {
        paused_ = false;
        {
            std::lock_guard<decltype(mutex_)> lock(mutex_);
            for (auto &wptr: nodes_to_resume_) {
                std::shared_ptr<IInputReset> node = wptr.lock();
                if (node) {
                    node->resetInput();
                }
            }
            cv_.notify_all();
        }
    }
    bool checkPauseNonRecursive(const av::Timestamp& ts) {
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

public:
    bool isPaused() {
        return paused_;
    }
    
    void pause(bool synchonize_nodes = true) {
        pauseNonRecursive(synchonize_nodes);
        {
            for (auto team: getLinkedTeams()) {
                team->pauseNonRecursive(synchonize_nodes);
            }
        }
    }

    void pause(const StreamTarget& target) {
        pauseNonRecursive(target);
        {
            for (auto team: getLinkedTeams()) {
                team->pauseNonRecursive(target);
            }
        }
    }

    bool checkPause(const av::Timestamp& ts) {
        if (checkPauseNonRecursive(ts)) {
            return true;
        }
        {
            for (auto team: getLinkedTeams()) {
                if (team->checkPauseNonRecursive(ts)) {
                    return true;
                }
            }
        }
        return false;
    }
    void resume() {
        resumeNonRecursive();
        {
            for (auto team: getLinkedTeams()) {
                team->resumeNonRecursive();
            }
        }
    }
    bool waitForResume(int timeout_ms) {
        std::unique_lock<decltype(mutex_)> lock(mutex_);
        if (!paused_) {
            return true;
        }
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]{ return !paused_; });
        return !paused_;
    }
    void addNode(std::weak_ptr<IInputReset> node) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        // TODO DRY: TickSource::add
        nodes_to_resume_.erase(std::remove_if(nodes_to_resume_.begin(), nodes_to_resume_.end(), [](std::weak_ptr<IInputReset> &p) { return p.expired(); }), nodes_to_resume_.end());
        nodes_to_resume_.push_back(std::weak_ptr<IInputReset>(node));
    }
    void setPlaybackNode(std::weak_ptr<IPlaybackControl>& node) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        playback_ = node;
    }
    void addSyncObj(std::weak_ptr<IFlushAndSeek> obj) {
        std::lock_guard<decltype(mutex_)> lock(mutex_);
        sync_objs_.push_back(obj);
    }
};
