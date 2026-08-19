#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"

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
    AVPixelFormat sw_format_ = AV_PIX_FMT_NONE;
    bool drop_alpha_ = true;

    // EGL state
    EGLDisplay egl_dpy_ = EGL_NO_DISPLAY;
    EGLContext egl_ctx_ = EGL_NO_CONTEXT;
    EGLSurface egl_surf_ = EGL_NO_SURFACE;

    GLuint src_tex_ = 0;

    bool have_dma_buf_import_ = false;
    bool have_mods_ = false;
    PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR_ = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR_ = nullptr;

    // CUDA state
    AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

    struct GresEntry {
        int width = 0;
        int height = 0;
        AVPixelFormat swfmt = AV_PIX_FMT_NONE;
        GLuint dst_tex = 0;
        CUgraphicsResource gres = nullptr;
    };
    std::vector<GresEntry> gres_cache_;

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

    void createTextures() {
        if (src_tex_ != 0) {
            glDeleteTextures(1, &src_tex_);
            src_tex_ = 0;
        }
        if (width_ <= 0 || height_ <= 0) return;
        
        glGenTextures(1, &src_tex_);
        glBindTexture(GL_TEXTURE_2D, src_tex_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Find a cache entry matching (w, h, swfmt), or allocate a new GL texture and add one.
    // Must be called with the EGL/GL context current and NO CUDA context pushed.
    int findOrCreateEntryIdx(int w, int h, AVPixelFormat swfmt) {
        for (int i = 0; i < (int)gres_cache_.size(); ++i) {
            const auto& e = gres_cache_[i];
            if (e.width == w && e.height == h && e.swfmt == swfmt) return i;
        }
        GresEntry e;
        e.width = w; e.height = h; e.swfmt = swfmt;
        glGenTextures(1, &e.dst_tex);
        glBindTexture(GL_TEXTURE_2D, e.dst_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        gres_cache_.push_back(std::move(e));
        return (int)gres_cache_.size() - 1;
    }

    static AVPixelFormat swfmt_from_fourcc(uint32_t fourcc) {
        switch (fourcc) {
            case DRM_FORMAT_ABGR8888: return AV_PIX_FMT_RGBA;
            case DRM_FORMAT_ARGB8888: return AV_PIX_FMT_BGRA;
            default: return AV_PIX_FMT_NONE;
        }
        // use AV_PIX_FMT_0BGR32 for hwdownload not supporting transparency
    }

    bool ensureCudaFramesCtxAndTextures(int w, int h, AVPixelFormat swfmt) {
        if (!hwaccel_) return false;
        if (w <= 0 || h <= 0) return false;
        bool need = false;
        if (!hw_frames_ctx_) need = true;
        if (!need && (w != width_ || h != height_)) need = true;
        if (!need && swfmt != sw_format_) need = true;
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
        sw_format_ = swfmt;
        createTextures();

        // Cache CUDA device ctx pointer for stream & context switches
        AVHWDeviceContext* devctx = (AVHWDeviceContext *)(hwaccel_->deviceContext()->data);
        cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
        return true;
    }

    bool import_one_plane_to_cuda(const AVDRMFrameDescriptor* desc, int layer_index, int plane_index,
                                  int width, int height, AVPixelFormat swfmt, av::VideoFrame &dst) {

        // Ensure GL context is current BEFORE any GL work (including texture creation).
        // Do NOT push CUDA context yet — on NVIDIA, having the CUDA context current
        // while calling glGenTextures / eglMakeCurrent can prevent the GL context from
        // becoming usable, leaving all glGen* calls returning 0 (invalid).
        if (!ensureEGL()) return false;

        const AVDRMLayerDescriptor &layer = desc->layers[layer_index];
        const AVDRMPlaneDescriptor &pl = layer.planes[plane_index];
        const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];

        if (!ensureCudaFramesCtxAndTextures(width, height, swfmt)) {
            return false;
        }

        // Resolve (or allocate) the dst_tex+gres cache entry for this frame's dimensions.
        // Must happen before cuCtxPushCurrent to keep GL and CUDA contexts separate.
        int entry_idx = findOrCreateEntryIdx(width, height, swfmt);

        EGLAttrib attrs[64];
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

        EGLImageKHR img = eglCreateImage(egl_dpy_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer)NULL, attrs);

        if (img == EGL_NO_IMAGE_KHR) {
            logstream << "drm2cuda: eglCreateImage failed width=" << width << " height=" << height;
            return false;
        }

        glBindTexture(GL_TEXTURE_2D, src_tex_);
        glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
        
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            logstream << "drm2cuda: glEGLImageTargetTexture2DOES failed: " << err;
            return false;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
        
        glCopyImageSubData(src_tex_, GL_TEXTURE_2D, 0, 0, 0, 0,
            gres_cache_[entry_idx].dst_tex, GL_TEXTURE_2D, 0, 0, 0, 0,
            width, height, 1);

        err = glGetError();
        if (err != GL_NO_ERROR) {
            logstream << "drm2cuda: glCopyImageSubData failed: " << err;
            return false;
        }

        glFinish();

        // All GL work is done. Now push the CUDA context for CUDA-GL interop.
        int cuda_error = 0;
        cuda_error |= CHECK_CU(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx));
        if (cuda_error) {
            logstream << "drm2cuda: cuCtxPushCurrent failed";
            p_eglDestroyImageKHR_(egl_dpy_, img);
            return false;
        }

        // Register the dst_tex with CUDA once per cache entry (lazy, first use only).
        GresEntry& entry = gres_cache_[entry_idx];
        if (!entry.gres) {
            CUresult cr = cuGraphicsGLRegisterImage(&entry.gres, entry.dst_tex, GL_TEXTURE_2D, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            if (cr != CUDA_SUCCESS) {
                logstream << "drm2cuda: cuGraphicsGLRegisterImage failed: " << cr;
                entry.gres = nullptr;
                CUcontext dummy; cuCtxPopCurrent(&dummy);
                p_eglDestroyImageKHR_(egl_dpy_, img);
                return false;
            }
        }
        CUgraphicsResource gres = entry.gres;
        cuGraphicsMapResources(1, &gres, 0);
        CUarray garr = nullptr;
        cuGraphicsSubResourceGetMappedArray(&garr, gres, 0, 0);

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

        cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        cpy.srcArray = garr;

        cuda_error |= CHECK_CU(cuMemcpy2DAsync(&cpy, cuda_dev_ctx_->stream));
        cuda_error |= CHECK_CU(cuStreamSynchronize(cuda_dev_ctx_->stream));

        cuGraphicsUnmapResources(1, &gres, 0);
        p_eglDestroyImageKHR_(egl_dpy_, img);

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

        // single-plane ABGR/ARGB only for now. Keep RGB0 as the default for
        // existing encoder pipelines, but let compositors retain browser alpha.
        const uint32_t fourcc = desc->layers[0].format;
        const AVPixelFormat source_swfmt = swfmt_from_fourcc(fourcc);
        if (source_swfmt == AV_PIX_FMT_NONE) {
            logstream << "drm2cuda: unsupported DRM fourcc";
            return;
        }
        const AVPixelFormat swfmt = drop_alpha_ ? AV_PIX_FMT_RGB0 : source_swfmt;

        av::VideoFrame out;
        out.setTimeBase(in.timeBase());
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
        if (cuda_dev_ctx_ && !gres_cache_.empty()) {
            cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx);
            for (auto& e : gres_cache_) {
                if (e.gres) {
                    cuGraphicsUnregisterResource(e.gres);
                    e.gres = nullptr;
                }
            }
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
        }
        if (egl_dpy_ != EGL_NO_DISPLAY) {
            eglTerminate(egl_dpy_);
            egl_dpy_ = EGL_NO_DISPLAY;
        }
        if (src_tex_ != 0) {
            glDeleteTextures(1, &src_tex_);
            src_tex_ = 0;
        }
        for (auto& e : gres_cache_) {
            if (e.dst_tex) {
                glDeleteTextures(1, &e.dst_tex);
                e.dst_tex = 0;
            }
        }
        gres_cache_.clear();
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
        if (params.count("drop_alpha")) {
            r->drop_alpha_ = params["drop_alpha"].get<bool>();
        }
        return r;
    }
};

DECLNODE(drm_prime_to_cuda, DRMPrimeToCUDA);
