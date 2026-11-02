#pragma once
// Markers for an Nsight Systems timeline (doc/mixer.md, "Profiling"). Without a profiler each
// costs one pointer check; without the NVTX headers (deps/nvtx) they compile to nothing.
#include <cstdint>
#include <string>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define AVP_HAVE_NVTX 1
#endif

namespace avp::timeline {

/// An instant on this thread's row.
inline void mark([[maybe_unused]] const std::string &text) {
#ifdef AVP_HAVE_NVTX
    nvtxMarkA(text.c_str());
#endif
}

/// A range that may end on another thread, as a take does. 0: no profiler.
inline uint64_t start([[maybe_unused]] const std::string &name) {
#ifdef AVP_HAVE_NVTX
    return nvtxRangeStartA(name.c_str());
#else
    return 0;
#endif
}

inline void end([[maybe_unused]] uint64_t range) {
#ifdef AVP_HAVE_NVTX
    if (range) nvtxRangeEnd(range);
#endif
}

/// A range on this thread's row, until the end of the scope or close().
class Scope {
    bool open_ = true;

public:
    explicit Scope([[maybe_unused]] const char *name) {
#ifdef AVP_HAVE_NVTX
        nvtxRangePushA(name);
#endif
    }
    /// `tick`, with the output tick's index as its payload: outputs line up on it.
    explicit Scope([[maybe_unused]] int64_t tick) {
#ifdef AVP_HAVE_NVTX
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        attributes.payload.llValue = tick;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = "tick";
        nvtxRangePushEx(&attributes);
#endif
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    void close() {
#ifdef AVP_HAVE_NVTX
        if (open_) nvtxRangePop();
#endif
        open_ = false;
    }
    ~Scope() { close(); }
};

/// Names a CUDA stream's row (CUDA builds, timeline_cuda.cpp).
void nameCudaStream(void *stream, const std::string &name);

}  // namespace avp::timeline
