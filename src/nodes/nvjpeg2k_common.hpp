#pragma once
// Shared by the nvjpeg2k encoder and decoder nodes: SDK status reporting, the pixel formats both
// sides accept, and the CUDA context their GPU work runs in.
//
// `../cuda.hpp` comes before any FFmpeg CUDA header on purpose: it brings in the bundled
// driver-API loader the rest of the tree uses, which defines CUDA_VERSION, so
// libavutil/hwcontext_cuda.h does not pull the SDK's cuda.h in beside it.
#include "../cuda.hpp"
#include "../util.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <nvjpeg2k.h>

#include <string>

namespace avp::nvjpeg2k {

inline std::string statusString(nvjpeg2kStatus_t st) {
    switch (st) {
        case NVJPEG2K_STATUS_SUCCESS: return "success";
        case NVJPEG2K_STATUS_NOT_INITIALIZED: return "not initialized";
        case NVJPEG2K_STATUS_INVALID_PARAMETER: return "invalid parameter";
        case NVJPEG2K_STATUS_BAD_JPEG: return "bad jpeg2000";
        case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED: return "jpeg2000 not supported";
        case NVJPEG2K_STATUS_ALLOCATOR_FAILURE: return "allocator failure";
        case NVJPEG2K_STATUS_EXECUTION_FAILED: return "execution failed";
        case NVJPEG2K_STATUS_ARCH_MISMATCH: return "arch mismatch";
        case NVJPEG2K_STATUS_INTERNAL_ERROR: return "internal error";
        case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "implementation not supported";
#if NVJPEG2K_VER_MAJOR > 0 || NVJPEG2K_VER_MINOR >= 12
        case NVJPEG2K_STATUS_TILE_TOO_LARGE: return "tile too large";   // nvJPEG2000 0.12 and up
#endif
        default: return "status " + std::to_string(static_cast<int>(st));
    }
}

/// Throws naming the node (`who`) and the call that failed (`what`) on anything but success.
inline void check(nvjpeg2kStatus_t st, const char *who, const char *what) {
    if (st != NVJPEG2K_STATUS_SUCCESS) {
        throw Error(std::string(who) + ' ' + what + " failed: " + statusString(st));
    }
}

inline const char *pixFmtName(AVPixelFormat fmt) {
    const char *name = av_get_pix_fmt_name(fmt);
    return name ? name : "none";
}

/// The formats both nodes handle, with the codestream precision each carries. nvJPEG2000 works on
/// 3 components of equal size here, so these are the planar 4:4:4 ones; subsampled and alpha
/// formats are not supported by either node. Ordered by precision: formatForPrecision() takes the
/// first that is wide enough.
struct Format {
    AVPixelFormat pix_fmt;
    int precision;
};
inline constexpr Format kFormats[] = {
    {AV_PIX_FMT_YUV444P, 8},
    {AV_PIX_FMT_YUV444P10LE, 10},
    {AV_PIX_FMT_YUV444P12LE, 12},
    {AV_PIX_FMT_YUV444P16LE, 16},
};

/// The supported formats as FFmpeg spells them, for an error message.
inline std::string formatList() {
    std::string result;
    for (const Format &f : kFormats) {
        if (!result.empty()) result += ", ";
        result += pixFmtName(f.pix_fmt);
    }
    return result;
}

inline bool supportedFormat(AVPixelFormat fmt) {
    for (const Format &f : kFormats) {
        if (f.pix_fmt == fmt) return true;
    }
    return false;
}

/// The component precision of a supported format; throws for any other.
inline int componentPrecision(AVPixelFormat fmt, const char *who) {
    for (const Format &f : kFormats) {
        if (f.pix_fmt == fmt) return f.precision;
    }
    throw Error(std::string(who) + " unsupported pixel format " + pixFmtName(fmt) + ", supported: " +
                formatList());
}

/// The narrowest supported format that holds `precision` bits per component.
inline AVPixelFormat formatForPrecision(int precision, const char *who) {
    for (const Format &f : kFormats) {
        if (precision <= f.precision) return f.pix_fmt;
    }
    throw Error(std::string(who) + " unsupported component precision " + std::to_string(precision));
}

/// A supported format by the name FFmpeg gives it; throws for an unknown or unsupported one.
inline AVPixelFormat parsePixelFormat(const std::string &name, const char *who) {
    const AVPixelFormat fmt = av_get_pix_fmt(name.c_str());
    if (fmt == AV_PIX_FMT_NONE) {
        throw Error(std::string(who) + " unknown pixel format: " + name);
    }
    if (!supportedFormat(fmt)) {
        throw Error(std::string(who) + " pixel format " + name + " is not supported, supported: " +
                    formatList());
    }
    return fmt;
}

/// Makes a CUDA context current on the calling thread for as long as the guard lives.
///
/// FFmpeg's CUDA device context is a context of its own (`cuCtxCreate`, not the device's primary
/// one), and `av_hwdevice_ctx_create` pops it before returning, so a node's thread has no current
/// context. Without this, nvjpeg2k and the CUDA runtime bind the device's primary context lazily
/// on their first call, and that context cannot address the frame memory FFmpeg allocated in its
/// own: the encode or decode fails with an illegal address, or writes nothing anyone can read.
class ContextGuard {
    bool pushed_ = false;

public:
    /// `who` names the node in the error. Throws when the context cannot be made current.
    ContextGuard(CUcontext ctx, const char *who) {
        if (!ctx) throw Error(std::string(who) + " has no CUDA context");
        if (!cuCtxPushCurrent || !cuCtxPopCurrent) {
            throw Error(std::string(who) + " needs the CUDA driver API, which did not load");
        }
        if (CHECK_CU(cuCtxPushCurrent(ctx))) {
            throw Error(std::string(who) + " cannot enter its CUDA context");
        }
        pushed_ = true;
    }
    ~ContextGuard() {
        if (!pushed_) return;
        CUcontext previous = nullptr;
        CHECK_CU(cuCtxPopCurrent(&previous));
    }
    ContextGuard(const ContextGuard &) = delete;
    ContextGuard &operator=(const ContextGuard &) = delete;
};

/// A node-private CUDA stream for nvjpeg2k's work, ordered against the stream FFmpeg's frames
/// are produced and consumed on.
///
/// FFmpeg leaves `AVCUDADeviceContext::stream` NULL ("will make functions default to the legacy
/// default stream"), so every node's GPU work shares the legacy stream. Submitting there and then
/// synchronizing it once per frame is a process-wide barrier -- the legacy stream serializes with
/// every blocking stream in the process -- so each encoded or decoded frame stalls NVDEC,
/// scale_cuda and NVENC in the same graph. Running on a non-blocking stream of our own avoids
/// that, and the two orderings that matter are kept explicitly: waitFor() puts our work after
/// whatever last wrote or read the frame memory, and synchronize() holds the host until our work
/// is done, before the frame or packet goes downstream.
///
/// Every method needs the device's CUDA context current (see ContextGuard).
class WorkStream {
    CUstream stream_ = nullptr;
    CUevent frames_ready_ = nullptr;

public:
    WorkStream() = default;
    ~WorkStream() { destroy(); }
    WorkStream(const WorkStream &) = delete;
    WorkStream &operator=(const WorkStream &) = delete;

    /// Creates the stream and event on first call.
    void ensure(const char *who) {
        if (stream_) return;
        if (!cuStreamCreate || !cuEventCreate) {
            throw Error(std::string(who) + " needs the CUDA driver API, which did not load");
        }
        // The bundled dynamic-loader header omits CU_STREAM_NON_BLOCKING (0x1).
        if (CHECK_CU(cuStreamCreate(&stream_, 0x1))) {
            stream_ = nullptr;
            throw Error(std::string(who) + " cannot create its CUDA stream");
        }
        if (CHECK_CU(cuEventCreate(&frames_ready_, CU_EVENT_DISABLE_TIMING))) {
            frames_ready_ = nullptr;
            throw Error(std::string(who) + " cannot create its CUDA event");
        }
    }

    /// Frees them, reporting rather than throwing: this runs from a destructor.
    void destroy() noexcept {
        if (frames_ready_) {
            CHECK_CU(cuEventDestroy(frames_ready_));
            frames_ready_ = nullptr;
        }
        if (stream_) {
            CHECK_CU(cuStreamSynchronize(stream_));
            CHECK_CU(cuStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    cudaStream_t get() const { return reinterpret_cast<cudaStream_t>(stream_); }

    /// Orders our work after everything already queued on `frames`, the stream the hardware frames
    /// belong to. Recording an event is not a barrier, so this costs the producer nothing.
    void waitFor(CUstream frames, const char *who) {
        if (CHECK_CU(cuEventRecord(frames_ready_, frames)) ||
            CHECK_CU(cuStreamWaitEvent(stream_, frames_ready_, 0))) {
            throw Error(std::string(who) + " cannot order its CUDA stream after the frame's");
        }
    }

    void synchronize(const char *who) {
        if (CHECK_CU(cuStreamSynchronize(stream_))) {
            throw Error(std::string(who) + " cuStreamSynchronize failed");
        }
    }
};

/// The CUDA device context behind a hardware device, for ContextGuard; throws when the device is
/// not a usable CUDA one.
inline AVCUDADeviceContext *deviceContext(AVBufferRef *device_ref, const char *who) {
    if (!device_ref || !device_ref->data) throw Error(std::string(who) + " has no hwaccel device");
    auto *device = reinterpret_cast<AVHWDeviceContext *>(device_ref->data);
    if (device->type != AV_HWDEVICE_TYPE_CUDA || !device->hwctx) {
        throw Error(std::string(who) + " requires a cuda hwaccel device");
    }
    return reinterpret_cast<AVCUDADeviceContext *>(device->hwctx);
}

} // namespace avp::nvjpeg2k
