#pragma once
// NVDEC CUarray planes are sampled in place. A frames-context reference keeps its
// fixed array pool alive without retaining a decoder frame or its surface index.
#include "cuda_array.hpp"
#include "cuda_rect_sampler.h"
#include "../../util.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <memory>

extern "C" {
#include <libavutil/version.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace avp::cuda {

template<unsigned PlaneCount, int (*Check)(CUresult, const char *)>
class ArrayTextures {
    static_assert(PlaneCount == 1 || PlaneCount == 2);
public:
    // Called only after the previous draw's stream synchronization.
    void begin() {
        for (auto &[key, pool] : pools_) pool->used = false;
    }

    void prune() {
        for (auto it = pools_.begin(); it != pools_.end();)
            if (!it->second->used) it = pools_.erase(it);
            else ++it;
    }

    const std::array<CUtexObject, PlaneCount> &get(const AVFrame *frame, CUcontext context) {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(61, 5, 100)
        if (!frame->hw_frames_ctx || !frame->hw_frames_ctx->data || !frame->data[0])
            throw Error("CUDA array textures: CUarray frame has no frames context or array");
        auto *frames = reinterpret_cast<AVHWFramesContext *>(frame->hw_frames_ctx->data);
        if (!frames->device_ctx || frames->device_ctx->type != AV_HWDEVICE_TYPE_CUDA)
            throw Error("CUDA array textures: CUarray frame has no CUDA device");
        auto *device = static_cast<AVCUDADeviceContext *>(frames->device_ctx->hwctx);
        if (!device || device->cuda_ctx != context)
            throw Error("CUDA array textures: CUarray source and compositor must share a CUDA context");
        const auto *array_frames = static_cast<const AVCUDAFramesContext *>(frames->hwctx);
        const CUarray array = reinterpret_cast<CUarray>(frame->data[0]);
        // Retaining a frames context guarantees lifetime only for its own pool.
        // Do not cache arbitrary external handles or pin decoded frame buffers.
        if (!array_frames || !array_frames->cuarray_surfaces || array_frames->cuarray_num_surfaces <= 0 ||
            std::find(array_frames->cuarray_surfaces,
                      array_frames->cuarray_surfaces + array_frames->cuarray_num_surfaces, array) ==
                      array_frames->cuarray_surfaces + array_frames->cuarray_num_surfaces)
            throw Error("CUDA array textures: CUarray source must use an FFmpeg fixed surface pool");

        auto it = pools_.find(frames);
        if (it == pools_.end())
            it = pools_.emplace(frames, std::make_unique<Pool>(frame->hw_frames_ctx, device)).first;
        Pool &pool = *it->second;
        pool.used = true;
        auto &textures = pool.textures[array];
        if (!textures) textures = std::make_unique<Planes>(array, frame, frames->sw_format);
        return textures->handles;
#else
        throw Error("CUDA array textures: CUarray requires FFmpeg libavutil 61.5 or newer");
#endif
    }

private:
    struct Planes {
        std::array<CUtexObject, PlaneCount> handles{};
        Planes(CUarray array, const AVFrame *frame, AVPixelFormat format) {
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(format);
            if (!desc || desc->nb_components != 3 || av_pix_fmt_count_planes(format) != 2 ||
                (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_ALPHA)))
                throw Error("CUDA array textures: CUarray input must be semiplanar YUV");
            try {
                for (unsigned p = 0; p < handles.size(); ++p) {
                    CUarray plane = nullptr;
                    CUDA_ARRAY3D_DESCRIPTOR geometry{};
                    if (Check(arrayGetPlane(&plane, array, p), "cuArrayGetPlane") ||
                        Check(cuArray3DGetDescriptor(&geometry, plane), "cuArray3DGetDescriptor"))
                        throw Error("CUDA array textures: cannot inspect CUarray plane");
                    const unsigned width = AV_CEIL_RSHIFT(frame->width, p ? desc->log2_chroma_w : 0);
                    const unsigned height = AV_CEIL_RSHIFT(frame->height, p ? desc->log2_chroma_h : 0);
                    const CUarray_format expected = desc->comp[0].depth + desc->comp[0].shift <= 8 ? CU_AD_FORMAT_UNSIGNED_INT8
                                                                            : CU_AD_FORMAT_UNSIGNED_INT16;
                    if (geometry.Format != expected || geometry.NumChannels != (p ? 2u : 1u) ||
                        geometry.Width < width || geometry.Height < height)
                        throw Error("CUDA array textures: CUarray plane layout disagrees with sw_format");
                    CUDA_RESOURCE_DESC resource;
                    CUDA_TEXTURE_DESC texture;
                    avp::mixer::rectTextureDesc(plane, resource, texture);
                    if (Check(cuTexObjectCreate(&handles[p], &resource, &texture, nullptr), "cuTexObjectCreate"))
                        throw Error("CUDA array textures: cannot create CUarray plane texture");
                }
            } catch (...) {
                release();
                throw;
            }
        }
        ~Planes() { release(); }
        void release() {
            for (auto &handle : handles)
                if (handle) { Check(cuTexObjectDestroy(handle), "cuTexObjectDestroy"); handle = 0; }
        }
    };

    struct Pool {
        AVBufferRef *frames_ref = nullptr;
        CUcontext context = nullptr;
        bool used = false;
        std::map<CUarray, std::unique_ptr<Planes>> textures;

        Pool(AVBufferRef *frames, const AVCUDADeviceContext *device) : context(device->cuda_ctx) {
            frames_ref = av_buffer_ref(frames);
            if (!frames_ref) throw Error("CUDA array textures: cannot retain CUarray frames context");
        }
        ~Pool() {
            // A pool may outlive its decoder; frames_ref also owns the device.
            if (!Check(cuCtxPushCurrent(context), "cuCtxPushCurrent")) {
                textures.clear();
                CUcontext previous;
                Check(cuCtxPopCurrent(&previous), "cuCtxPopCurrent");
            } else textures.clear();
            av_buffer_unref(&frames_ref);
        }
    };

    std::map<AVHWFramesContext *, std::unique_ptr<Pool>> pools_;
};

} // namespace avp::cuda
