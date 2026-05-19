#pragma once
#include "avplumber.hpp"
#include "instance_shared.hpp"
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

#ifdef EMBED_IN_OBS
struct obs_source;
typedef struct obs_source obs_source_t;
#endif


class InstanceData {
    friend class AVPlumber;
public:
    using ExceptionCallback = std::function<void(const std::string&, const std::string&, const std::string&)>;
#ifdef EMBED_IN_OBS
protected:
    std::atomic<obs_source_t*> obs_source_ {nullptr};
    std::atomic_int obs_source_used_by_ {0};
public:
    bool doWithObsSource(std::function<void(obs_source_t*)> cb) {
        if (!obs_source_.load()) {
            return false;
        }
        obs_source_used_by_++;
        obs_source_t* s = obs_source_.load();
        if (s) {
            try {
                cb(s);
            } catch(std::exception &e) {
                obs_source_used_by_--;
                throw e;
            }
        }
        obs_source_used_by_--;
        return s;
    }
#endif
private:
    std::mutex exception_callback_mutex_;
    ExceptionCallback exception_callback_;
public:
    InstanceData() {};
    InstanceData(const InstanceData &copyfrom) = delete;
    void setExceptionCallback(ExceptionCallback callback) {
        std::lock_guard<decltype(exception_callback_mutex_)> lock(exception_callback_mutex_);
        exception_callback_ = std::move(callback);
    }
    void notifyException(const std::string &node_name, const std::string &node_type, const std::string &message) {
        std::lock_guard<decltype(exception_callback_mutex_)> lock(exception_callback_mutex_);
        if (exception_callback_) {
            exception_callback_(node_name, node_type, message);
        }
    }
    ~InstanceData() {
        InstanceSharedObjectsDestructors::callDestructors(this);
    }
};
