#pragma once
#include "cuda_rect_draw.hpp"

namespace avp::mixer {

// FFmpeg filters use the stream attached to their input frames' device. Give
// aux frames a device wrapper with a private stream and the same CUDA context.
inline std::shared_ptr<HWAccelDevice> makeCudaStreamDevice(const std::shared_ptr<HWAccelDevice> &parent) {
    AVBufferRef *ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
    if (!ref) throw Error("aux: cannot allocate CUDA device wrapper");
    auto *device = reinterpret_cast<AVHWDeviceContext *>(ref->data);
    auto *cuda = reinterpret_cast<AVCUDADeviceContext *>(device->hwctx);
    auto *original = reinterpret_cast<AVCUDADeviceContext *>(
        reinterpret_cast<AVHWDeviceContext *>(parent->deviceContext()->data)->hwctx);
    cuda->cuda_ctx = original->cuda_ctx;
    device->user_opaque = av_buffer_ref(parent->deviceContext());
    device->free = [](AVHWDeviceContext *ctx) {
        auto *c = reinterpret_cast<AVCUDADeviceContext *>(ctx->hwctx);
        cuCtxSetCurrent(c->cuda_ctx);
        if (c->stream) { cuStreamSynchronize(c->stream); cuStreamDestroy(c->stream); }
        auto *parent_ref = static_cast<AVBufferRef *>(ctx->user_opaque);
        av_buffer_unref(&parent_ref);
    };
    if (!device->user_opaque || AVP_CHECK_CU(cuCtxSetCurrent(cuda->cuda_ctx)) ||
        // The bundled dynamic-loader header omits CU_STREAM_NON_BLOCKING (0x1).
        AVP_CHECK_CU(cuStreamCreate(&cuda->stream, 0x1)) || av_hwdevice_ctx_init(ref) < 0) {
        av_buffer_unref(&ref);
        throw Error("aux: cannot initialize CUDA stream device");
    }
    return std::make_shared<HWAccelDevice>(ref);
}

}
