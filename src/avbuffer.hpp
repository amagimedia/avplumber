#pragma once
#include <memory>

extern "C" {
#include <libavutil/buffer.h>
}

namespace avp {

// avcpp's SmartDeleter does not support AVBufferRef.
struct AvBufferUnref {
    void operator()(AVBufferRef* ref) const noexcept { av_buffer_unref(&ref); }
};

using AvBufferRef = std::unique_ptr<AVBufferRef, AvBufferUnref>;

}
