#pragma once

#include "../../hwaccel/EglImageFrame.hpp"
#include <avcpp/frame.h>
#include <memory>

class CudaEglImagePool;

// Converts a CUDA frame to a pooled RGBA EGL image. A published frame holds
// its slot until the last reader releases it.
class CudaEglFrameRenderer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    enum class Result { Ready, Busy, Invalid };
    enum class Sync { Async, Context, PrivateStream };
    CudaEglFrameRenderer(std::shared_ptr<CudaEglImagePool> pool, int size, int max_size, int grow_step, Sync sync);
    ~CudaEglFrameRenderer();
    Result render(av::VideoFrame &frame, EglImageFrame &out);
};
