#pragma once
// Single worker thread that runs mixer transition steps at their scheduled
// wall-clock time, in submission order for equal times.
#include "../instance_shared.hpp"
#include "../graph_mgmt.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace avp::mixer {

class TransitionScheduler : public InstanceShared<TransitionScheduler>, public IShutdownable {
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    TransitionScheduler();
    ~TransitionScheduler();

    void post(std::string label, std::function<void()> task);
    void postAfter(std::string label, int64_t delay_ms, std::function<void()> task);
    void shutdown();
};

}  // namespace avp::mixer
