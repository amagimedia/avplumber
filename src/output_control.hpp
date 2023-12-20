#pragma once
#include <unordered_map>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <avcpp/timestamp.h>
#include "avutils.hpp"

class OutputControl {
public:
    enum State {
        Stopped,
        Waiting1,
        Waiting2,
        Started
    };
protected:
    static std::unordered_map<std::string, std::shared_ptr<OutputControl>> instances_;
    std::atomic<State> state_{Stopped};
    size_t tempcut_nodes_count_ = 0;
    size_t notcut_nodes_count_ = 0;
    std::mutex mutex_;
    std::unordered_set<void*> tempcut_pts_sources_;
    av::Timestamp tempcut_max_pts_ = NOTS;
    av::Timestamp start_pts_ = NOTS;
public:
    static std::shared_ptr<OutputControl> get(std::string name, bool create = true) {
        // TODO InstanceShared

        std::shared_ptr<OutputControl> &ptr = instances_[name];
        if (!ptr) {
            if (create) {
                ptr = std::make_shared<OutputControl>();
            } else {
                throw Error("Output control group not found");
            }
        }
        return ptr;
    }
    av::Timestamp minPts() { // MINimal for looking for video packet to emit
        return tempcut_max_pts_;
    }
    State setStartPts(av::Timestamp pts) {
        auto lock = getLock();
        start_pts_ = pts;
        State expected = Waiting2;
        if (state_.compare_exchange_strong(expected, Started, std::memory_order_release, std::memory_order_acquire)) {
            logstream << "output control: Started, start pts " << start_pts_;
            return Started;
        } else {
            logstream << "output control not changed to Waiting2 when setting start pts";
            return expected;
        }
    }
    av::Timestamp startPts() {
        return start_pts_;
    }
    void registerNode(bool is_temporally_cuttable) {
        if (is_temporally_cuttable) {
            tempcut_nodes_count_++;
        } else {
            notcut_nodes_count_++;
            if (notcut_nodes_count_>1) {
                throw Error("No more than 1 not-cuttable outputs supported");
            }
        }
    }
    void deregisterNode(bool is_temporally_cuttable) {
        if (is_temporally_cuttable) {
            assert(tempcut_nodes_count_>0);
            tempcut_nodes_count_--;
        } else {
            assert(notcut_nodes_count_>0);
            notcut_nodes_count_--;
        }
    }
    std::unique_lock<decltype(mutex_)> getLock() {
        return std::unique_lock<decltype(mutex_)>(mutex_);
    }
    State addTemporallyCuttablePTS(av::Timestamp pts, void* source) {
        auto lock = getLock();
        State state = state_.load(std::memory_order_relaxed);
        if (state == Waiting1) {
            if (tempcut_max_pts_.isNoPts() || (pts > tempcut_max_pts_)) {
                tempcut_max_pts_ = pts;
            }
            tempcut_pts_sources_.insert(source);
            size_t got_ptses = tempcut_pts_sources_.size();
            if (got_ptses >= tempcut_nodes_count_) {
                if (got_ptses > tempcut_nodes_count_) {
                    logstream << "BUG: got_ptses > tempcut_nodes_count_";
                }
                state = Waiting2;
                state_.store(state, std::memory_order_release);
                logstream << "output control: Waiting2, min next pts " << tempcut_max_pts_;
            }
        }
        return state;
    }
    void start() {
        auto lock = getLock();
        State expected = Stopped;
        tempcut_pts_sources_.clear();
        tempcut_max_pts_ = NOTS;
        State new_state = Waiting1;
        if (tempcut_nodes_count_==0) {
            logstream << "WARNING: Untested: Starting output without temporally cuttable nodes.";
            new_state = Waiting2;
        }

        if (state_.compare_exchange_strong(expected, new_state, std::memory_order_relaxed)) {
            logstream << "output control: Waiting";
        } else {
            logstream << "output control not started";
        }
    }
    void stop() {
        auto lock = getLock();
        state_.store(Stopped, std::memory_order_relaxed);
        logstream << "output control: Stopped";
    }
    State state() {
        return state_.load(std::memory_order_acquire);
    }
};


