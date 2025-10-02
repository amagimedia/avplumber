#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
#include "../../../deps/cuda_loader/cuda.h"
#include <unistd.h>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/hwcontext_drm.h>
#include <libdrm/drm_fourcc.h>
}



class DRMPrimeToCUDA: public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef* hw_frames_ctx_ = nullptr;
    int width_ = 0;
    int height_ = 0;

    // CUDA state
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

    static inline const char* safe_str(const char* s) { return s ? s : ""; }

    static AVPixelFormat swfmt_from_fourcc(uint32_t fourcc) {
        return AV_PIX_FMT_0BGR32;
        switch (fourcc) {
            case DRM_FORMAT_ABGR8888: return AV_PIX_FMT_RGBA;
            case DRM_FORMAT_ARGB8888: return AV_PIX_FMT_BGRA;
            default: return AV_PIX_FMT_NONE;
        }
    }

    bool ensureCudaFramesCtx(int w, int h, AVPixelFormat swfmt) {
        if (!hwaccel_) return false;
        if (w <= 0 || h <= 0) return false;
        bool need = false;
        if (!hw_frames_ctx_) need = true;
        if (!need && (w != width_ || h != height_)) need = true;
        if (!need) return true;

        if (hw_frames_ctx_) {
            av_buffer_unref(&hw_frames_ctx_);
            hw_frames_ctx_ = nullptr;
        }
        hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
        if (!hw_frames_ctx_) {
            logstream << "drm2cuda: av_hwframe_ctx_alloc failed";
            return false;
        }
        AVHWFramesContext *frmctx = (AVHWFramesContext *)(hw_frames_ctx_->data);
        frmctx->format = AV_PIX_FMT_CUDA;
        frmctx->sw_format = swfmt;
        frmctx->width = w;
        frmctx->height = h;
        int r = av_hwframe_ctx_init(hw_frames_ctx_);
        if (r != 0) {
            logstream << "drm2cuda: av_hwframe_ctx_init failed: " << av::error2string(r);
            av_buffer_unref(&hw_frames_ctx_);
            hw_frames_ctx_ = nullptr;
            return false;
        }
        width_ = w;
        height_ = h;

        // Cache CUDA device ctx pointer for stream & context switches
        AVHWDeviceContext* devctx = (AVHWDeviceContext *)(hwaccel_->deviceContext()->data);
        cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
        return true;
    }

    bool import_one_plane_to_cuda(const AVDRMFrameDescriptor* desc, int layer_index, int plane_index,
                                  int width, int height, AVPixelFormat swfmt, av::VideoFrame &dst) {
        int cuda_error = 0;
        cuda_error |= CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (cuda_error) {
            logstream << "drm2cuda: cuCtxPushCurrent failed";
            return false;
        }

        const AVDRMLayerDescriptor &layer = desc->layers[layer_index];
        const AVDRMPlaneDescriptor &pl = layer.planes[plane_index];
        const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];

        logstream << "drm2cuda: import fd=" << obj.fd
                  << " size=" << (unsigned long long)obj.size
                  << " mod=0x" << std::hex << (unsigned long long)obj.format_modifier << std::dec
                  << " fourcc=0x" << std::hex << layer.format << std::dec
                  << " w=" << width << " h=" << height
                  << " pitch=" << pl.pitch << " offset=" << pl.offset;

        if (layer.format != DRM_FORMAT_ABGR8888 && layer.format != DRM_FORMAT_ARGB8888) {
            CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "drm2cuda: unsupported DRM format for external memory";
            return false;
        }

        const int bytes_per_pixel = 4; // ABGR/ARGB 8:8:8:8
        const uint64_t required_span = (uint64_t)pl.offset + (uint64_t)(pl.pitch) * (uint64_t)(height - 1) + (uint64_t)(width * bytes_per_pixel);
        uint64_t allocation_size = obj.size ? obj.size : required_span;

        int imported_fd = dup(obj.fd);
        if (imported_fd < 0) {
            CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "drm2cuda: dup(fd) failed";
            return false;
        }

        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc{};
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memDesc.handle.fd = imported_fd;
        memDesc.size = allocation_size;
        memDesc.flags = 0;

        CUexternalMemory extMem = nullptr;
        CUresult cr = cuImportExternalMemory(&extMem, &memDesc);
        if (cr != CUDA_SUCCESS) {
            // Retry with DEDICATED flag; close fd if still failing
            memDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
            cr = cuImportExternalMemory(&extMem, &memDesc);
            if (cr != CUDA_SUCCESS) {
                logstream << "drm2cuda: cuImportExternalMemory failed, size=" << (unsigned long long)memDesc.size
                          << " flags=" << memDesc.flags;
                close(imported_fd);
                CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
                return false;
            }
        }

        CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufDesc{};
        bufDesc.offset = 0;                       // map from base of allocation
        bufDesc.size = allocation_size;           // map whole allocation
        bufDesc.flags = 0;

        CUdeviceptr baseDevPtr = 0;
        cr = cuExternalMemoryGetMappedBuffer(&baseDevPtr, extMem, &bufDesc);
        if (cr != CUDA_SUCCESS) {
            cuDestroyExternalMemory(extMem);
            CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "drm2cuda: cuExternalMemoryGetMappedBuffer failed";
            return false;
        }

        CUdeviceptr srcDevPtr = baseDevPtr + (size_t)pl.offset;

        if (!hw_frames_ctx_) {
            if (!ensureCudaFramesCtx(width, height, swfmt)) {
                cuDestroyExternalMemory(extMem);
                CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
                return false;
            }
        }

        dst.setTimeBase({1, 1000000});
        dst.raw()->format = AV_PIX_FMT_CUDA;
        dst.raw()->width = width;
        dst.raw()->height = height;
        av_hwframe_get_buffer(hw_frames_ctx_, dst.raw(), 0);

        CUDA_MEMCPY2D cpy{};
        cpy.WidthInBytes = (size_t)width * bytes_per_pixel;
        cpy.Height = (size_t)height;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = (CUdeviceptr)reinterpret_cast<uint64_t>(dst.raw()->data[0]);
        cpy.dstPitch = (size_t)dst.raw()->linesize[0];

        cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.srcDevice = srcDevPtr;
        cpy.srcPitch = (size_t)pl.pitch;

        cuda_error |= CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream));
        cuda_error |= CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));

        cuDestroyExternalMemory(extMem);

        CUcontext dummy;
        cuda_error |= CHECK_CU(cuCtxPopCurrent(&dummy));

        if (cuda_error) {
            logstream << "drm2cuda: CUDA copy failed";
            return false;
        }
        return true;
    }

public:
    using NodeSISO::NodeSISO;
    virtual void process() {
        av::VideoFrame in = this->source_->get();
        if (!in) return;
        if (in.raw()->format != AV_PIX_FMT_DRM_PRIME) {
            // pass through if not DRM PRIME
            this->sink_->put(in);
            return;
        }

        const AVDRMFrameDescriptor *desc = (const AVDRMFrameDescriptor*)in.raw()->data[0];
        if (!desc) {
            logstream << "drm2cuda: missing DRM descriptor";
            return;
        }
        if (desc->nb_layers < 1 || desc->layers[0].nb_planes < 1) {
            logstream << "drm2cuda: unsupported layer/plane count";
            return;
        }

        // single-plane ABGR/ARGB only for now
        uint32_t fourcc = desc->layers[0].format;
        AVPixelFormat swfmt = swfmt_from_fourcc(fourcc);
        if (swfmt == AV_PIX_FMT_NONE) {
            logstream << "drm2cuda: unsupported DRM fourcc";
            return;
        }

        av::VideoFrame out;
        out.raw()->pts = in.raw()->pts;
        out.raw()->color_range = in.raw()->color_range;
        out.raw()->colorspace = in.raw()->colorspace;

        int w = in.width();
        int h = in.height();
        bool ok = import_one_plane_to_cuda(desc, 0, 0, w, h, swfmt, out);
        if (!ok) return;

        if (hw_frames_ctx_) {
            out.raw()->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
        }
        out.setComplete(true);
        this->sink_->put(out);
    }
    DRMPrimeToCUDA(std::unique_ptr<typename NodeSISO<av::VideoFrame,av::VideoFrame>::SourceType> &&source,
                   std::unique_ptr<typename NodeSISO<av::VideoFrame,av::VideoFrame>::SinkType> &&sink)
        : NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)) {}

    ~DRMPrimeToCUDA() {
        if (hw_frames_ctx_) {
            av_buffer_unref(&hw_frames_ctx_);
            hw_frames_ctx_ = nullptr;
        }
    }

    static std::shared_ptr<DRMPrimeToCUDA> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::VideoFrame>> dst = edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<DRMPrimeToCUDA>(make_unique<EdgeSource<av::VideoFrame>>(src), make_unique<EdgeSink<av::VideoFrame>>(dst));
        if (!params.count("hwaccel")) {
            throw Error("drm_prime_to_cuda requires hwaccel parameter (CUDA device)");
        }
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_) {
            throw Error("drm_prime_to_cuda: failed to get hwaccel");
        }
        AVHWDeviceContext* devctx = (AVHWDeviceContext *)(r->hwaccel_->deviceContext()->data);
        r->cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
        if (!r->cuda_dev_ctx_) {
            throw Error("drm_prime_to_cuda: CUDA device context missing");
        }
        return r;
    }
};

DECLNODE(drm_prime_to_cuda, DRMPrimeToCUDA);


