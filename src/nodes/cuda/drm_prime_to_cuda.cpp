#define EGL_EGLEXT_PROTOTYPES 1
#include <EGL/eglext.h>
#include <EGL/egl.h>

#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
#include "../../../deps/cuda_loader/cudaEGL.h"

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

    // EGL state
    EGLDisplay egl_dpy_ = EGL_NO_DISPLAY;
    bool have_dma_buf_import_ = false;
    bool have_mods_ = false;

    // CUDA state
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

    static inline const char* safe_str(const char* s) { return s ? s : ""; }

    bool ensureEGL() {
        if (egl_dpy_ != EGL_NO_DISPLAY) return true;
        egl_dpy_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_dpy_ == EGL_NO_DISPLAY) {
            logstream << "drm2cuda: eglGetDisplay failed";
            return false;
        }
        EGLint major=0, minor=0;
        if (!eglInitialize(egl_dpy_, &major, &minor)) {
            logstream << "drm2cuda: eglInitialize failed";
            egl_dpy_ = EGL_NO_DISPLAY;
            return false;
        }
        const char* exts = eglQueryString(egl_dpy_, EGL_EXTENSIONS);
        have_dma_buf_import_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import");
        have_mods_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import_modifiers");
        if (!have_dma_buf_import_) {
            logstream << "drm2cuda: EGL_EXT_image_dma_buf_import missing";
            return false;
        }
        return true;
    }

    static AVPixelFormat swfmt_from_fourcc(uint32_t fourcc) {
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
        if (!ensureEGL()) return false;

        const AVDRMLayerDescriptor &layer = desc->layers[layer_index];
        const AVDRMPlaneDescriptor &pl = layer.planes[plane_index];
        const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];

        EGLint attrs[64];
        int a = 0;
        attrs[a++] = EGL_WIDTH;  attrs[a++] = (EGLint)width;
        attrs[a++] = EGL_HEIGHT; attrs[a++] = (EGLint)height;
        attrs[a++] = EGL_LINUX_DRM_FOURCC_EXT; attrs[a++] = (EGLint)layer.format;
        attrs[a++] = EGL_DMA_BUF_PLANE0_FD_EXT; attrs[a++] = (EGLint)obj.fd;
        attrs[a++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; attrs[a++] = (EGLint)pl.pitch;
        attrs[a++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; attrs[a++] = (EGLint)pl.offset;
        if (have_mods_ && obj.format_modifier) {
            EGLint mod_lo = (EGLint)(obj.format_modifier & 0xFFFFFFFFu);
            EGLint mod_hi = (EGLint)(obj.format_modifier >> 32);
            attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; attrs[a++] = mod_lo;
            attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; attrs[a++] = mod_hi;
        }
        attrs[a++] = EGL_NONE;

        EGLImageKHR img = eglCreateImageKHR(egl_dpy_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer)NULL, attrs);
        if (img == EGL_NO_IMAGE_KHR) {
            logstream << "drm2cuda: eglCreateImageKHR failed";
            return false;
        }

        CUgraphicsResource gres = nullptr;
        CUresult cr = cuGraphicsEGLRegisterImage(&gres, img, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (cr != CUDA_SUCCESS) {
            logstream << "drm2cuda: cuGraphicsEGLRegisterImage failed";
            eglDestroyImageKHR(egl_dpy_, img);
            return false;
        }

        CUeglFrame eframe{};
        cr = cuGraphicsResourceGetMappedEglFrame(&eframe, gres, 0, 0);
        if (cr != CUDA_SUCCESS) {
            logstream << "drm2cuda: cuGraphicsResourceGetMappedEglFrame failed";
            cuGraphicsUnregisterResource(gres);
            eglDestroyImageKHR(egl_dpy_, img);
            return false;
        }

        int cuda_error = 0;
        cuda_error |= CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (cuda_error) {
            logstream << "drm2cuda: cuCtxPushCurrent failed";
            cuGraphicsUnregisterResource(gres);
            eglDestroyImageKHR(egl_dpy_, img);
            return false;
        }

        if (!hw_frames_ctx_) {
            if (!ensureCudaFramesCtx(width, height, swfmt)) {
                CUcontext dummy; CHECK_CU(cuCtxPopCurrent(&dummy));
                cuGraphicsUnregisterResource(gres);
                eglDestroyImageKHR(egl_dpy_, img);
                return false;
            }
        }

        dst.setTimeBase({1, 1000000});
        dst.raw()->format = AV_PIX_FMT_CUDA;
        dst.raw()->width = width;
        dst.raw()->height = height;
        av_hwframe_get_buffer(hw_frames_ctx_, dst.raw(), 0);

        CUDA_MEMCPY2D cpy{};
        cpy.WidthInBytes = (size_t)width * 4;
        cpy.Height = (size_t)height;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = (CUdeviceptr)reinterpret_cast<uint64_t>(dst.raw()->data[0]);
        cpy.dstPitch = (size_t)dst.raw()->linesize[0];

        if (eframe.frameType == 0 /*CU_EGL_FRAME_TYPE_ARRAY*/) {
            cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            cpy.srcArray = eframe.frame.pArray[0];
        } else {
            cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice = (CUdeviceptr)reinterpret_cast<uint64_t>(eframe.frame.pPitch[0]);
            cpy.srcPitch = (size_t)eframe.pitch;
        }

        cuda_error |= CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream));
        cuda_error |= CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));

        CUcontext dummy;
        cuda_error |= CHECK_CU(cuCtxPopCurrent(&dummy));

        cuGraphicsUnregisterResource(gres);
        eglDestroyImageKHR(egl_dpy_, img);

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
        if (egl_dpy_ != EGL_NO_DISPLAY) {
            eglTerminate(egl_dpy_);
            egl_dpy_ = EGL_NO_DISPLAY;
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


