#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>

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
    AVPixelFormat sw_fmt_ = AV_PIX_FMT_NONE;

    // EGL state
    EGLDisplay egl_dpy_ = EGL_NO_DISPLAY;
    EGLContext egl_ctx_ = EGL_NO_CONTEXT;
    EGLSurface egl_surf_ = EGL_NO_SURFACE;

    bool have_dma_buf_import_ = false;
    bool have_mods_ = false;
    PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR_ = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR_ = nullptr;

    // CUDA state
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

    // One EGL image + CUDA registration per physical DMA-BUF allocation. The
    // producer recycles a small pool of buffers, so after warm-up every frame
    // is one device copy from the mapped EGL frame instead of a fresh import
    // (EGL image, GL copy, glFinish, CUDA map/unmap, destroy) per frame.
    struct ImportKey {
        dev_t st_dev = 0;
        ino_t st_ino = 0;
        uint32_t width = 0, height = 0, pitch = 0, fourcc = 0;
        uint64_t modifier = 0, offset = 0;
        bool operator==(const ImportKey &o) const {
            return st_dev == o.st_dev && st_ino == o.st_ino && width == o.width && height == o.height &&
                   pitch == o.pitch && fourcc == o.fourcc && modifier == o.modifier && offset == o.offset;
        }
    };
    struct ImportEntry {
        ImportKey key;
        int dup_fd = -1;                    // keeps the allocation (and its inode) alive while cached
        EGLImageKHR image = EGL_NO_IMAGE_KHR;
        CUgraphicsResource resource = nullptr;
        CUeglFrame frame{};
        int64_t last_used_ms = 0;
    };
    std::vector<ImportEntry> imports_;
    size_t max_imports_ = 64;
    int64_t import_ttl_ms_ = 3000;
    int64_t last_purge_ms_ = 0;
    uint64_t cache_hits_ = 0, fresh_imports_ = 0;

    static inline const char* safe_str(const char* s) { return s ? s : ""; }

    bool ensureEGL() {
        if (egl_ctx_ != EGL_NO_CONTEXT) {
            // Re-bind on every call: cuCtxPushCurrent/PopCurrent each frame can
            // release the GL context on the current thread, so we must restore it.
            if (!eglMakeCurrent(egl_dpy_, egl_surf_, egl_surf_, egl_ctx_)) {
                logstream << "drm2cuda: eglMakeCurrent (re-bind) failed: " << eglGetError();
                return false;
            }
            return true;
        }

        EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy == EGL_NO_DISPLAY) {
            logstream << "drm2cuda: eglGetDisplay failed";
            return false;
        }
        EGLint major=0, minor=0;
        if (!eglInitialize(dpy, &major, &minor)) {
            logstream << "drm2cuda: eglInitialize failed";
            return false;
        }

        if (!eglBindAPI(EGL_OPENGL_API)) {
            logstream << "drm2cuda: eglBindAPI(EGL_OPENGL_API) failed: " << eglGetError();
            return false;
        }
        logstream << "drm2cuda: EGL vendor: " << safe_str(eglQueryString(dpy, EGL_VENDOR));

        static const EGLint ctx_config_attribs[] = {EGL_STENCIL_SIZE,
            0,
            EGL_DEPTH_SIZE,
            0,
            EGL_BUFFER_SIZE,
            32,
            EGL_ALPHA_SIZE,
            8,
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_NONE};
        
        EGLConfig cfg = nullptr;
        EGLint num = 0;
        if (!eglChooseConfig(dpy, ctx_config_attribs, &cfg, 1, &num) || num < 1) {
            logstream << "drm2cuda: eglChooseConfig failed";
            return false;
        }

        static int ctx_pbuffer_attribs[] = {EGL_WIDTH, 2, EGL_HEIGHT, 2, EGL_NONE};
        EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, ctx_pbuffer_attribs);
        if (surf == EGL_NO_SURFACE) {
            logstream << "drm2cuda: eglCreatePbufferSurface failed: " << eglGetError();
            return false;
        }
        
        static const int ctx_attribs[] = {
            #ifdef _DEBUG
                EGL_CONTEXT_OPENGL_DEBUG,
                EGL_TRUE,
            #endif
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_CONTEXT_MAJOR_VERSION,
                3,
                EGL_CONTEXT_MINOR_VERSION,
                3,
                EGL_NONE,
        };
        
        EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attribs);
        if (ctx == EGL_NO_CONTEXT) {
            logstream << "drm2cuda: eglCreateContext failed: " << eglGetError();
            eglDestroySurface(dpy, surf);
            return false;
        }

        const char* exts = eglQueryString(dpy, EGL_EXTENSIONS);
        have_dma_buf_import_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import");
        have_mods_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import_modifiers");
        if (!have_dma_buf_import_) {
            logstream << "drm2cuda: EGL_EXT_image_dma_buf_import missing";
            eglDestroyContext(dpy, ctx);
            eglDestroySurface(dpy, surf);
            return false;
        }
        // Resolve extension function pointers at runtime to avoid link-time deps
        if (!p_eglCreateImageKHR_) {
            p_eglCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
            if (!p_eglCreateImageKHR_) {
                // Try core symbol name as a fallback on some implementations
                p_eglCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImage");
            }
        }
        if (!p_eglDestroyImageKHR_) {
            p_eglDestroyImageKHR_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
            if (!p_eglDestroyImageKHR_) {
                p_eglDestroyImageKHR_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImage");
            }
        }
        if (!p_eglCreateImageKHR_ || !p_eglDestroyImageKHR_) {
            logstream << "drm2cuda: failed to load eglCreateImageKHR/eglDestroyImageKHR";
            eglDestroyContext(dpy, ctx);
            eglDestroySurface(dpy, surf);
            return false;
        }
        if (!eglMakeCurrent(dpy, surf, surf, ctx)) {
            logstream << "drm2cuda: eglMakeCurrent failed: " << eglGetError();
            eglDestroyContext(dpy, ctx);
            eglDestroySurface(dpy, surf);
            return false;
        }

        // Commit — only store members after full successful init
        egl_dpy_ = dpy;
        egl_surf_ = surf;
        egl_ctx_ = ctx;

        auto get_string = [](GLenum key) {
            const char* s = (const char*)glGetString(key);
            return s ? std::string(s) : std::string("null");
        };
        logstream << "gl: " << get_string(GL_VENDOR) << " / " << get_string(GL_RENDERER) << " / " << get_string(GL_VERSION);

        return true;
    }

    static AVPixelFormat swfmt_from_fourcc(uint32_t fourcc) {
        // The fourth byte is copied but never meaningful downstream, so advertise
        // the no-alpha variant in the buffer's own byte order.
        switch (fourcc) {
            case DRM_FORMAT_ABGR8888: case DRM_FORMAT_XBGR8888: return AV_PIX_FMT_RGB0;
            case DRM_FORMAT_ARGB8888: case DRM_FORMAT_XRGB8888: return AV_PIX_FMT_BGR0;
            default: return AV_PIX_FMT_NONE;
        }
    }

    bool ensureCudaFramesCtx(int w, int h, AVPixelFormat swfmt) {
        if (!hwaccel_) return false;
        if (w <= 0 || h <= 0) return false;
        bool need = false;
        if (!hw_frames_ctx_) need = true;
        if (!need && (w != width_ || h != height_ || swfmt != sw_fmt_)) need = true;
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
        sw_fmt_ = swfmt;

        // Cache CUDA device ctx pointer for stream & context switches
        AVHWDeviceContext* devctx = (AVHWDeviceContext *)(hwaccel_->deviceContext()->data);
        cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
        return true;
    }

    void releaseEntry(ImportEntry &e) {
        if (e.resource) {
            cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx);
            cuGraphicsUnregisterResource(e.resource);
            CUcontext dummy; cuCtxPopCurrent(&dummy);
            e.resource = nullptr;
        }
        if (e.image != EGL_NO_IMAGE_KHR && egl_dpy_ != EGL_NO_DISPLAY && p_eglDestroyImageKHR_)
            p_eglDestroyImageKHR_(egl_dpy_, e.image);
        e.image = EGL_NO_IMAGE_KHR;
        if (e.dup_fd >= 0) close(e.dup_fd);
        e.dup_fd = -1;
    }

    void purgeImports(int64_t now_ms, bool all) {
        if (!all && last_purge_ms_ && now_ms - last_purge_ms_ < 1000) return;
        last_purge_ms_ = now_ms;
        for (auto it = imports_.begin(); it != imports_.end();) {
            if (all || now_ms - it->last_used_ms >= import_ttl_ms_) {
                releaseEntry(*it);
                it = imports_.erase(it);
            } else {
                ++it;
            }
        }
    }

    ImportEntry *findOrImport(const AVDRMFrameDescriptor *desc, int width, int height, int64_t now_ms) {
        const AVDRMLayerDescriptor &layer = desc->layers[0];
        const AVDRMPlaneDescriptor &pl = layer.planes[0];
        if (pl.object_index < 0 || pl.object_index >= desc->nb_objects) return nullptr;
        const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];
        struct stat st{};
        if (fstat(obj.fd, &st) != 0) {
            logstream << "drm2cuda: fstat(fd) failed: " << std::strerror(errno);
            return nullptr;
        }
        ImportKey key{st.st_dev, st.st_ino, (uint32_t)width, (uint32_t)height, (uint32_t)pl.pitch,
                      layer.format, obj.format_modifier, (uint64_t)pl.offset};
        purgeImports(now_ms, false);
        for (auto &e : imports_) {
            if (e.key == key) {
                e.last_used_ms = now_ms;
                ++cache_hits_;
                return &e;
            }
        }
        if (!ensureEGL()) return nullptr;

        ImportEntry e;
        e.key = key;
        e.last_used_ms = now_ms;
        e.dup_fd = dup(obj.fd);
        if (e.dup_fd < 0) {
            logstream << "drm2cuda: dup(fd) failed: " << std::strerror(errno);
            return nullptr;
        }
        EGLAttrib attrs[32];
        int a = 0;
        attrs[a++] = EGL_WIDTH;  attrs[a++] = (EGLint)width;
        attrs[a++] = EGL_HEIGHT; attrs[a++] = (EGLint)height;
        attrs[a++] = EGL_LINUX_DRM_FOURCC_EXT; attrs[a++] = (EGLint)layer.format;
        attrs[a++] = EGL_DMA_BUF_PLANE0_FD_EXT; attrs[a++] = e.dup_fd;
        attrs[a++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; attrs[a++] = (EGLint)pl.pitch;
        attrs[a++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; attrs[a++] = (EGLint)pl.offset;
        if (have_mods_ && obj.format_modifier) {
            attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; attrs[a++] = (EGLint)(obj.format_modifier & 0xFFFFFFFFu);
            attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; attrs[a++] = (EGLint)(obj.format_modifier >> 32);
        }
        attrs[a++] = EGL_NONE;
        e.image = eglCreateImage(egl_dpy_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer)NULL, attrs);
        if (e.image == EGL_NO_IMAGE_KHR) {
            logstream << "drm2cuda: eglCreateImage failed width=" << width << " height=" << height
                      << " EGL error=" << eglGetError();
            releaseEntry(e);
            return nullptr;
        }

        bool ok = !CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (ok) {
            ok = !CHECK_CU(cuGraphicsEGLRegisterImage(&e.resource, e.image, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)) &&
                 !CHECK_CU(cuGraphicsResourceGetMappedEglFrame(&e.frame, e.resource, 0, 0));
            CUcontext dummy; cuCtxPopCurrent(&dummy);
        }
        if (ok && (e.frame.planeCount != 1 || e.frame.width != (unsigned)width || e.frame.height != (unsigned)height ||
                   e.frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8 || e.frame.numChannels != 4 ||
                   (e.frame.frameType != CU_EGL_FRAME_TYPE_ARRAY && e.frame.frameType != CU_EGL_FRAME_TYPE_PITCH))) {
            logstream << "drm2cuda: unsupported EGL frame planes=" << e.frame.planeCount << " size=" << e.frame.width
                      << "x" << e.frame.height << " channels=" << e.frame.numChannels
                      << " type=" << (int)e.frame.frameType;
            ok = false;
        }
        if (!ok) {
            releaseEntry(e);
            return nullptr;
        }
        if (imports_.size() >= max_imports_) {
            auto oldest = std::min_element(imports_.begin(), imports_.end(),
                [](const ImportEntry &l, const ImportEntry &r) { return l.last_used_ms < r.last_used_ms; });
            releaseEntry(*oldest);
            imports_.erase(oldest);
        }
        ++fresh_imports_;
        if (fresh_imports_ <= 2 || fresh_imports_ % 64 == 0)
            logstream << "drm2cuda: imported allocation " << width << "x" << height << " type="
                      << (e.frame.frameType == CU_EGL_FRAME_TYPE_PITCH ? "pitch" : "array")
                      << " cached=" << imports_.size() + 1 << " hits=" << cache_hits_ << " imports=" << fresh_imports_;
        imports_.push_back(e);
        return &imports_.back();
    }

    bool import_to_cuda(const AVDRMFrameDescriptor* desc, int width, int height, AVPixelFormat swfmt,
                        av::VideoFrame &dst) {
        if (!ensureCudaFramesCtx(width, height, swfmt)) return false;
        ImportEntry *e = findOrImport(desc, width, height, wallclock.pts());
        if (!e) return false;

        int cuda_error = CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (cuda_error) return false;
        dst.raw()->format = AV_PIX_FMT_CUDA;
        dst.raw()->width = width;
        dst.raw()->height = height;
        int r = av_hwframe_get_buffer(hw_frames_ctx_, dst.raw(), 0);
        if (r < 0) {
            logstream << "drm2cuda: av_hwframe_get_buffer failed: " << av::error2string(r);
            CUcontext dummy; cuCtxPopCurrent(&dummy);
            return false;
        }
        CUDA_MEMCPY2D cpy{};
        cpy.WidthInBytes = (size_t)width * 4;
        cpy.Height = (size_t)height;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = (CUdeviceptr)reinterpret_cast<uintptr_t>(dst.raw()->data[0]);
        cpy.dstPitch = (size_t)dst.raw()->linesize[0];
        if (e->frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            cpy.srcArray = e->frame.frame.pArray[0];
        } else {
            cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            cpy.srcDevice = (CUdeviceptr)reinterpret_cast<uintptr_t>(e->frame.frame.pPitch[0]);
            cpy.srcPitch = (size_t)e->frame.pitch;
        }
        cuda_error |= CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream));
        // The producer may overwrite the buffer once the input frame is released,
        // so the copy must have landed before this call returns.
        cuda_error |= CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));
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

        AVPixelFormat swfmt = swfmt_from_fourcc(desc->layers[0].format);
        if (swfmt == AV_PIX_FMT_NONE) {
            logstream << "drm2cuda: unsupported DRM fourcc " << desc->layers[0].format;
            return;
        }

        av::VideoFrame out;
        out.setTimeBase(in.timeBase());
        out.raw()->pts = in.raw()->pts;
        out.raw()->color_range = in.raw()->color_range;
        out.raw()->colorspace = in.raw()->colorspace;

        int w = in.width();
        int h = in.height();
        bool ok = import_to_cuda(desc, w, h, swfmt, out);
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
        if (cuda_dev_ctx_) purgeImports(0, true);
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

