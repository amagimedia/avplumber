#pragma once
#include "avplumber.hpp"
#include "instance_shared.hpp"
#include <atomic>
#include <functional>

#ifdef EMBED_IN_OBS
struct obs_source;
typedef struct obs_source obs_source_t;
#endif


class InstanceData {
    friend class AVPlumber;
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
public:
    InstanceData() {};
    InstanceData(const InstanceData &copyfrom) = delete;
    ~InstanceData() {
        InstanceSharedObjectsDestructors::callDestructors(this);
    }
};
