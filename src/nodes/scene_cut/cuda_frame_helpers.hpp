#pragma once

#include "../node_common.hpp"

#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <sstream>
#include <string>

namespace scene_cut_cuda {

inline int checkCuda(CUresult error, const char* node_name, const char* function) {
    if (error == CUDA_SUCCESS) return 0;

    const char* error_name = nullptr;
    const char* error_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(error, &error_name);
        cuGetErrorString(error, &error_string);
    }
    logstream << node_name << ": " << function << " failed: "
              << (error_name ? error_name : "?") << ": "
              << (error_string ? error_string : "?");
    return -1;
}

inline bool isSupportedLumaCudaFormat(AVPixelFormat format) {
    switch (format) {
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_GRAY8:
            return true;
        default:
            return false;
    }
}

inline std::string jsonStringEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00"
                        << hex[(static_cast<unsigned char>(c) >> 4) & 0x0f]
                        << hex[static_cast<unsigned char>(c) & 0x0f];
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

inline AVPixelFormat hwSwFormat(const av::VideoFrame& frame) {
    if (!frame.raw() || !frame.raw()->hw_frames_ctx || !frame.raw()->hw_frames_ctx->data) {
        return AV_PIX_FMT_NONE;
    }
    const AVHWFramesContext* context =
        reinterpret_cast<const AVHWFramesContext*>(frame.raw()->hw_frames_ctx->data);
    return context ? context->sw_format : AV_PIX_FMT_NONE;
}

inline bool initCudaContextFromFrame(
    const av::VideoFrame& frame,
    const char* node_name,
    AVCUDADeviceContext*& cuda_device_context,
    CUcontext& cuda_context,
    CUstream& stream,
    bool& owns_stream
) {
    if (cuda_context) return true;
    if (!frame.raw() || !frame.raw()->hw_frames_ctx || !frame.raw()->hw_frames_ctx->data) {
        logstream << node_name << ": missing hw_frames_ctx";
        return false;
    }

    AVHWFramesContext* frame_context =
        reinterpret_cast<AVHWFramesContext*>(frame.raw()->hw_frames_ctx->data);
    if (!frame_context || !frame_context->device_ctx || !frame_context->device_ctx->hwctx) {
        logstream << node_name << ": missing device_ctx/hwctx in frame";
        return false;
    }

    cuda_device_context =
        static_cast<AVCUDADeviceContext*>(frame_context->device_ctx->hwctx);
    if (!cuda_device_context || !cuda_device_context->cuda_ctx) {
        logstream << node_name << ": missing CUDA context in frame";
        return false;
    }

    cuda_context = cuda_device_context->cuda_ctx;
    if (checkCuda(cuCtxSetCurrent(cuda_context), node_name, "cuCtxSetCurrent(cu_ctx_)")) {
        return false;
    }
    stream = cuda_device_context->stream;
    if (!stream) {
        if (checkCuda(cuStreamCreate(&stream, 0), node_name, "cuStreamCreate(&stream_, 0)")) {
            stream = nullptr;
            return false;
        }
        owns_stream = true;
    }
    return true;
}

} // namespace scene_cut_cuda
