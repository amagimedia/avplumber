#pragma once

#include "CudaVmmMemory.hpp"
#include "../../avbuffer.hpp"
#include "../../hwaccel.hpp"

extern "C" {
#include <libavutil/frame.h>
}

namespace avp::vmm {

class FramePool {
    struct State;
    std::shared_ptr<State> state_;

public:
    FramePool(std::shared_ptr<HWAccelDevice> device, AVPixelFormat format, int width, int height, unsigned slots);
    void requireLayout(const std::shared_ptr<HWAccelDevice> &device, AVPixelFormat format,
                       int width, int height, unsigned slots) const;
    av::VideoFrame acquire();
    uint64_t skipped() const;
};

}
