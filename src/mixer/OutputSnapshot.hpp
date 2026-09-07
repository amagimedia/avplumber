#pragma once

#include "Snapshot.hpp"
#include "../instance_shared.hpp"
#include "../avutils.hpp"
#include <mutex>

namespace avp::mixer {

struct OutputSnapshot : InstanceShared<OutputSnapshot> {
    std::mutex mutex;
    Snapshot<av::VideoFrame> frames;
    bool output_connected = false;
};

} // namespace avp::mixer
