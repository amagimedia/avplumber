#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"
#include "cuda_rect_texture.h"

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>

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
    bool drop_alpha_ = true;
    // zero_copy: hand the mapped EGL frame to consumers instead of copying it into a pool
    // frame. Pitch-linear imports become ordinary device-pointer frames; tiled (array)
    // imports carry a texture object (cuda_rect_texture.h) that only the compositor reads.
    // The output frame keeps the DRM input frame alive, so the producer's release ack goes
    // out when the last consumer lets go, not when the copy would have landed.
    bool zero_copy_ = false;
    bool zero_copy_warned_ = false;

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
        CUtexObject tex = 0;                // zero_copy on an array frame: sampling handle
        int64_t last_used_ms = 0;
        // Release happens when the cache AND every zero-copy frame referencing the entry are gone.
        CUcontext cuda_ctx = nullptr;
        EGLDisplay egl_dpy = EGL_NO_DISPLAY;
        PFNEGLDESTROYIMAGEKHRPROC destroy_image = nullptr;
        ~ImportEntry() {
            if (resource || tex) {
                cuCtxPushCurrent(cuda_ctx);
                if (tex) cuTexObjectDestroy(tex);
                if (resource) cuGraphicsUnregisterResource(resource);
                CUcontext dummy; cuCtxPopCurrent(&dummy);
            }
            if (image != EGL_NO_IMAGE_KHR && egl_dpy != EGL_NO_DISPLAY && destroy_image)
                destroy_image(egl_dpy, image);
            if (dup_fd >= 0) close(dup_fd);
        }
    };
    std::vector<std::shared_ptr<ImportEntry>> imports_;
    // Owner of a zero-copy output frame's buffer: pins the import entry and the DRM input frame.
    struct ZeroCopyOwner {
        std::shared_ptr<ImportEntry> entry;
        AVBufferRef *input = nullptr;
    };
    static void freeZeroCopyOwner(void *opaque, uint8_t *) {
        auto *owner = static_cast<ZeroCopyOwner *>(opaque);
        av_buffer_unref(&owner->input);   // closes the fds and queues the release ack
        delete owner;                     // may be the entry's last reference
    }
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

    static AVPixelFormat swfmt_from_fourcc(uint32_t fourcc, bool drop_alpha) {
        // Alpha is opt-in; X formats have padding rather than alpha even when
        // requested. The copy preserves all four bytes in their DRM byte order.
        switch (fourcc) {
            case DRM_FORMAT_ABGR8888: return drop_alpha ? AV_PIX_FMT_RGB0 : AV_PIX_FMT_RGBA;
            case DRM_FORMAT_ARGB8888: return drop_alpha ? AV_PIX_FMT_BGR0 : AV_PIX_FMT_BGRA;
            case DRM_FORMAT_XBGR8888: return AV_PIX_FMT_RGB0;
            case DRM_FORMAT_XRGB8888: return AV_PIX_FMT_BGR0;
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

    void purgeImports(int64_t now_ms, bool all) {
        if (!all && last_purge_ms_ && now_ms - last_purge_ms_ < 1000) return;
        last_purge_ms_ = now_ms;
        for (auto it = imports_.begin(); it != imports_.end();) {
            if (all || now_ms - (*it)->last_used_ms >= import_ttl_ms_)
                it = imports_.erase(it);   // released now, or when the last zero-copy frame dies
            else
                ++it;
        }
    }

    std::shared_ptr<ImportEntry> findOrImport(const AVDRMFrameDescriptor *desc, int width, int height, int64_t now_ms) {
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
            if (e->key == key) {
                e->last_used_ms = now_ms;
                ++cache_hits_;
                return e;
            }
        }
        if (!ensureEGL()) return nullptr;

        auto entry = std::make_shared<ImportEntry>();
        ImportEntry &e = *entry;
        e.key = key;
        e.last_used_ms = now_ms;
        e.cuda_ctx = cuda_dev_ctx_->cuda_ctx;
        e.egl_dpy = egl_dpy_;
        e.destroy_image = p_eglDestroyImageKHR_;
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
        if (ok && zero_copy_ && e.frame.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            // Sampling handle for the compositor: exact texels, unnormalized coordinates.
            CUDA_RESOURCE_DESC res{};
            res.resType = CU_RESOURCE_TYPE_ARRAY;
            res.res.array.hArray = e.frame.frame.pArray[0];
            CUDA_TEXTURE_DESC td{};
            td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
            td.filterMode = CU_TR_FILTER_MODE_POINT;
            td.flags = 0;   // element reads, integer coordinates
            cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx);
            const bool made = !CHECK_CU(cuTexObjectCreate(&e.tex, &res, &td, nullptr));
            CUcontext dummy; cuCtxPopCurrent(&dummy);
            if (!made) {
                e.tex = 0;
                if (!zero_copy_warned_) {
                    zero_copy_warned_ = true;
                    logstream << "drm2cuda: texture object creation failed; copying this allocation instead";
                }
            }
        }
        if (!ok)
            return nullptr;
        if (imports_.size() >= max_imports_) {
            auto oldest = std::min_element(imports_.begin(), imports_.end(),
                [](const std::shared_ptr<ImportEntry> &l, const std::shared_ptr<ImportEntry> &r) {
                    return l->last_used_ms < r->last_used_ms; });
            imports_.erase(oldest);
        }
        ++fresh_imports_;
        if (fresh_imports_ <= 2 || fresh_imports_ % 64 == 0)
            logstream << "drm2cuda: imported allocation " << width << "x" << height << " type="
                      << (e.frame.frameType == CU_EGL_FRAME_TYPE_PITCH ? "pitch" : "array")
                      << (zero_copy_ ? " zero-copy" : "")
                      << " cached=" << imports_.size() + 1 << " hits=" << cache_hits_ << " imports=" << fresh_imports_;
        imports_.push_back(entry);
        return entry;
    }

    // Zero-copy output: the frame points at the mapped EGL frame and pins the input DRM frame.
    bool wrap_mapped(const std::shared_ptr<ImportEntry> &e, const av::VideoFrame &in, int width, int height,
                     av::VideoFrame &dst) {
        if (!in.raw()->buf[0]) return false;
        const bool pitch = e->frame.frameType == CU_EGL_FRAME_TYPE_PITCH;
        if (!pitch && !e->tex) return false;
        auto *owner = new ZeroCopyOwner{e, av_buffer_ref(in.raw()->buf[0])};
        if (!owner->input) { delete owner; return false; }
        AVFrame *f = dst.raw();
        f->buf[0] = av_buffer_create(reinterpret_cast<uint8_t *>(owner), 0, freeZeroCopyOwner, owner, AV_BUFFER_FLAG_READONLY);
        if (!f->buf[0]) { freeZeroCopyOwner(owner, nullptr); return false; }
        f->format = AV_PIX_FMT_CUDA;
        f->width = width;
        f->height = height;
        if (pitch) {
            f->data[0] = reinterpret_cast<uint8_t *>(e->frame.frame.pPitch[0]);
            f->linesize[0] = (int)e->frame.pitch;
            return true;
        }
        f->opaque_ref = av_buffer_alloc(sizeof(avp::mixer::TextureFrameDesc));
        if (!f->opaque_ref) return false;
        auto *d = reinterpret_cast<avp::mixer::TextureFrameDesc *>(f->opaque_ref->data);
        *d = avp::mixer::TextureFrameDesc{};
        d->tex = e->tex;
        d->width = width;
        d->height = height;
        f->data[0] = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(e->tex));   // handle, not memory
        f->linesize[0] = width * 4;
        return true;
    }

    bool import_to_cuda(const AVDRMFrameDescriptor* desc, int width, int height, AVPixelFormat swfmt,
                        const av::VideoFrame &in, av::VideoFrame &dst) {
        if (!ensureCudaFramesCtx(width, height, swfmt)) return false;
        std::shared_ptr<ImportEntry> e = findOrImport(desc, width, height, wallclock.pts());
        if (!e) return false;
        if (zero_copy_ && wrap_mapped(e, in, width, height, dst))
            return true;

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

        AVPixelFormat swfmt = swfmt_from_fourcc(desc->layers[0].format, drop_alpha_);
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
        bool ok = import_to_cuda(desc, w, h, swfmt, in, out);
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
        r->drop_alpha_ = params.value("drop_alpha", true);
        r->zero_copy_ = params.value("zero_copy", false);
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
