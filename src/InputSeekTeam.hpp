#pragma once
#include <atomic>
#include <optional>
#include "instance_shared.hpp"
#include "graph_interfaces.hpp"

class InputSeekTeam: public InstanceShared<InputSeekTeam>, public ISeekAt {
protected:
    std::mutex mutex_;
    std::list<std::weak_ptr<ISeekAt>> seek_targets_;
public:
    virtual void seekAtAdd(const StreamTarget& when, const StreamTarget& target) override {
        std::unique_lock<decltype(mutex_)>(mutex_);
        for (auto t: seek_targets_) {
            auto node = t.lock();
            if (node) {
                node->seekAtAdd(when, target);
            }
        }
    }

    virtual void seekAtClear() override {
        std::unique_lock<decltype(mutex_)>(mutex_);
        for (auto t: seek_targets_) {
            auto node = t.lock();
            if (node) {
                node->seekAtClear();
            }
        }
    }

    void addSeekTarget(std::weak_ptr<ISeekAt> target) {
        std::unique_lock<decltype(mutex_)>(mutex_);
        seek_targets_.push_back(target);
    }
};
