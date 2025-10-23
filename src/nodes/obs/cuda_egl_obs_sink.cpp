#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1

#include "../node_common.hpp"

#include <atomic>
#include <mutex>
#include <unordered_map>

#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include <obs-module.h>
#include <graphics/graphics.h>

#ifndef HAVE_CUDA
#define HAVE_CUDA 0
#endif

#if HAVE_CUDA

#include <GL/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/hwcontext_cuda.h>

// CUDA runtime symbols (for the kernel launcher)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>

extern "C" cudaError_t yuv444p_to_rgba8_709lim_surface(
    const uint8_t* dY, const uint8_t* dU, const uint8_t* dV,
    int pitchY, int pitchU, int pitchV,
    cudaSurfaceObject_t surfOut,
    int W, int H,
    cudaStream_t stream);

// minimal subset from OBS internals used here
struct gs_device { struct gl_platform *plat; };
typedef struct gs_device gs_device_t;

struct gl_platform {
    Display *xdisplay;
    EGLDisplay edisplay;
    EGLConfig config;
    EGLContext context;
    EGLSurface pbuffer;
    bool close_xdisplay;
};

static inline bool gl_success(const char *funcname)
{
    GLenum errorcode = glGetError();
    if (errorcode != GL_NO_ERROR) {
        int attempts = 8;
        do {
            logstream << funcname << " failed, glGetError returned 0x" << std::hex << errorcode;
            errorcode = glGetError();
            --attempts;
            if (attempts == 0) {
                logstream << "Too many GL errors, moving on";
                break;
            }
        } while (errorcode != GL_NO_ERROR);
        return false;
    }
    return true;
}

static inline bool gl_tex_param_i(GLenum target, GLenum param, GLint val)
{
    glTexParameteri(target, param, val);
    return gl_success("glTexParameteri");
}

static inline bool gl_bind_texture(GLenum target, GLuint texture)
{
    glBindTexture(target, texture);
    return gl_success("glBindTexture");
}

static CudaFunctions* egl_global_cu = nullptr;

static int check_cu(CUresult err, const char *func)
{
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (egl_global_cu) {
        egl_global_cu->cuGetErrorName(err, &err_name);
        egl_global_cu->cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: " << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}

#define CHECK_CU(x) check_cu((x), #x)

// Ensure CUDA driver API is loaded and initialized
__attribute__((constructor)) static void init_cuda_cuda_egl_obs_sink(void)
{
    if (!cuda_load_functions(&egl_global_cu, nullptr)) {
        auto cu = egl_global_cu;
        if (!CHECK_CU(cu->cuInit(0))) {
            logstream << "cuda_egl_obs_sink: CUDA initialized";
        } else {
            logstream << "cuda_egl_obs_sink: failed to initialize CUDA";
            egl_global_cu = nullptr;
        }
    } else {
        egl_global_cu = nullptr;
        logstream << "cuda_egl_obs_sink: failed to load CUDA functions";
    }
}

class CudaEglObsSink: public NodeSingleInput<av::VideoFrame>, public NonBlockingNode<CudaEglObsSink>, public IFlushable {
protected:
    InstanceData& app_instance_;

    // OBS frame state
    struct obs_source_frame obs_frame_ = {0};
    enum AVPixelFormat cur_sw_pix_fmt_ = AV_PIX_FMT_NONE;
    AVTS prev_timestamp_ = 0;
    signed int timeout_ms_ = -1;
    AVTS last_frame_emitted_at_ = 0;
    bool unbuffered_ = false;
    bool debug_timing_ = false;

    // EGL/GL objects for our private context
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLConfig egl_config_ = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC EGLImageTargetTexture2DOES_ = nullptr;
    PFNEGLCREATEIMAGEKHRPROC EGLCreateImageKHR_ = nullptr;

    // GL texture we render into (in our context) and its EGLImage
    GLuint gl_tex_rgba_ = 0;
    EGLImageKHR egl_image_ = EGL_NO_IMAGE_KHR;
    int tex_w_ = 0;
    int tex_h_ = 0;

    // CUDA interop for the GL texture
    CUcontext cu_ctx_ = nullptr;
    CUgraphicsResource cu_tex_res_ = nullptr;

    // OBS HW buffer callbacks
    struct obs_hw_buffer obs_hw_;

    // Helpers
    av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm)
    {
        if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (ctx == nullptr) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }

    void destroy_cuda_gl_resources()
    {
        if (cu_tex_res_) {
            auto cu = egl_global_cu;
            CHECK_CU(cu->cuGraphicsUnregisterResource(cu_tex_res_));
            cu_tex_res_ = nullptr;
        }
        if (egl_image_ != EGL_NO_IMAGE_KHR) {
            eglDestroyImageKHR(egl_display_, egl_image_);
            egl_image_ = EGL_NO_IMAGE_KHR;
        }
        if (gl_tex_rgba_) {
            glDeleteTextures(1, &gl_tex_rgba_);
            gl_tex_rgba_ = 0;
        }
        tex_w_ = tex_h_ = 0;
    }

    bool ensure_egl_context()
    {
        if (egl_context_ != EGL_NO_CONTEXT) return true;

        graphics_t* graphics = gs_get_context();
        if (!graphics || !graphics->device || !graphics->device->plat) {
            logstream << "cuda_egl_obs_sink: OBS graphics device not available";
            return false;
        }
        gl_platform* plat = graphics->device->plat;
        egl_display_ = plat->edisplay;
        egl_config_ = plat->config;
        EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
        // create a shared context? we keep it unshared; not required to share with OBS
        egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctx_attr);
        if (egl_context_ == EGL_NO_CONTEXT) {
            logstream << "cuda_egl_obs_sink: eglCreateContext failed";
            return false;
        }
        EGLint pbuf_attr[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
        egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pbuf_attr);
        if (egl_surface_ == EGL_NO_SURFACE) {
            logstream << "cuda_egl_obs_sink: eglCreatePbufferSurface failed";
            return false;
        }
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
            logstream << "cuda_egl_obs_sink: eglMakeCurrent failed";
            return false;
        }

        EGLCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
        EGLImageTargetTexture2DOES_ = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
        if (!EGLCreateImageKHR_ || !EGLImageTargetTexture2DOES_) {
            logstream << "cuda_egl_obs_sink: failed to load EGL/GL image functions";
            return false;
        }

        return true;
    }

    bool ensure_cuda_context()
    {
        if (cu_ctx_) return true;
        if (!egl_global_cu) return false;
        auto cu = egl_global_cu;

        CUdevice display_dev;
        unsigned int device_count = 0;
        obs_enter_graphics();
        int rc = CHECK_CU(cu->cuGLGetDevices(&device_count, &display_dev, 1, CU_GL_DEVICE_LIST_ALL));
        obs_leave_graphics();
        if (rc) {
            logstream << "cuda_egl_obs_sink: cuGLGetDevices failed";
            return false;
        }
        if (CHECK_CU(cu->cuCtxCreate(&cu_ctx_, CU_CTX_SCHED_BLOCKING_SYNC, display_dev))) {
            logstream << "cuda_egl_obs_sink: cuCtxCreate failed";
            cu_ctx_ = nullptr;
            return false;
        }
        CUcontext dummy;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return true;
    }

    bool ensure_texture_and_registration(int W, int H)
    {
        if (!ensure_egl_context()) return false;
        if (!ensure_cuda_context()) return false;

        if (W == tex_w_ && H == tex_h_ && gl_tex_rgba_ && egl_image_ && cu_tex_res_) return true;

        // recreate
        destroy_cuda_gl_resources();

        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
            logstream << "cuda_egl_obs_sink: eglMakeCurrent before GL create failed";
            return false;
        }

        glGenTextures(1, &gl_tex_rgba_);
        gl_bind_texture(GL_TEXTURE_2D, gl_tex_rgba_);
        gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl_bind_texture(GL_TEXTURE_2D, 0);

        EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, EGL_NONE };
        egl_image_ = EGLCreateImageKHR_(egl_display_, egl_context_, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)gl_tex_rgba_, img_attrs);
        if (egl_image_ == EGL_NO_IMAGE_KHR) {
            logstream << "cuda_egl_obs_sink: eglCreateImageKHR failed";
            destroy_cuda_gl_resources();
            return false;
        }

        auto cu = egl_global_cu;
        CHECK_CU(cu->cuCtxPushCurrent(cu_ctx_));
        if (CHECK_CU(cu->cuGraphicsGLRegisterImage(&cu_tex_res_, gl_tex_rgba_, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST))) {
            logstream << "cuda_egl_obs_sink: cuGraphicsGLRegisterImage failed";
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            destroy_cuda_gl_resources();
            return false;
        }
        CUcontext dummy;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));

        tex_w_ = W;
        tex_h_ = H;
        return true;
    }

    bool run_conversion_to_texture(const av::VideoFrame &frm)
    {
        if (!egl_global_cu || !cu_ctx_ || !cu_tex_res_) return false;
        auto cu = egl_global_cu;

        CHECK_CU(cu->cuCtxPushCurrent(cu_ctx_));

        if (CHECK_CU(cu->cuGraphicsMapResources(1, &cu_tex_res_, 0))) {
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: cuGraphicsMapResources failed";
            return false;
        }

        CUarray cu_arr = nullptr;
        if (CHECK_CU(cu->cuGraphicsSubResourceGetMappedArray(&cu_arr, cu_tex_res_, 0, 0))) {
            CHECK_CU(cu->cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: cuGraphicsSubResourceGetMappedArray failed";
            return false;
        }

        cudaResourceDesc rdesc; memset(&rdesc, 0, sizeof(rdesc));
        rdesc.resType = cudaResourceTypeArray;
        rdesc.res.array.array = (cudaArray_t)cu_arr;
        cudaSurfaceObject_t surf = 0;
        cudaError_t cerr = cudaCreateSurfaceObject(&surf, &rdesc);
        if (cerr != cudaSuccess) {
            logstream << "cuda_egl_obs_sink: cudaCreateSurfaceObject failed: " << (int)cerr;
            CHECK_CU(cu->cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            return false;
        }

        const uint8_t* dY = (const uint8_t*)frm.raw()->data[0];
        const uint8_t* dU = (const uint8_t*)frm.raw()->data[1];
        const uint8_t* dV = (const uint8_t*)frm.raw()->data[2];
        int pitchY = frm.raw()->linesize[0];
        int pitchU = frm.raw()->linesize[1];
        int pitchV = frm.raw()->linesize[2];

        cudaStream_t stream = 0;
        cerr = yuv444p_to_rgba8_709lim_surface(dY, dU, dV, pitchY, pitchU, pitchV, surf, tex_w_, tex_h_, stream);
        if (cerr != cudaSuccess) {
            logstream << "cuda_egl_obs_sink: kernel launch failed: " << (int)cerr;
            cudaDestroySurfaceObject(surf);
            CHECK_CU(cu->cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            return false;
        }
        // Ensure completion before unmap so OBS sees complete frame
        //cudaDeviceSynchronize();
        cudaDestroySurfaceObject(surf);

        CHECK_CU(cu->cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
        CUcontext dummy;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return true;
    }

    void outputFrame()
    {
        app_instance_.doWithObsSource([this](obs_source_t* s) {
            obs_source_output_video(s, &obs_frame_);
        });
    }

public:
    using NodeSingleInput::NodeSingleInput;

    virtual void start()
    {
        app_instance_.doWithObsSource([this](obs_source_t* s) {
            obs_source_set_async_unbuffered(s, unbuffered_);
        });
    }

    virtual void processNonBlocking(EventLoop& evl, bool ticks)
    {
        av::VideoFrame *pfrm = this->source_->peek(0);
        if (pfrm==nullptr) {
            bool timelimit = (timeout_ms_>=0);
            if (timelimit && !ticks) {
                this->sleepAndProcess(timeout_ms_);
            }
            if (!ticks) {
                this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
            }
            if ((!timelimit) || (wallclock.pts() < last_frame_emitted_at_ + timeout_ms_)) {
                return;
            }
        }

        // reset frame
        obs_frame_ = {0};
        obs_frame_.format = VIDEO_FORMAT_RGBA;
        obs_frame_.full_range = true;
        obs_frame_.hw = &obs_hw_;

        if (pfrm && *pfrm) {
            av::VideoFrame frm = *pfrm;
            if (ticks) {
                while (this->source_->pop()) {};
            } else {
                this->source_->pop();
            }

            av::PixelFormat swpf = getHwSwPixelFormat(frm);
            if (!(frm.pixelFormat().get()==AV_PIX_FMT_CUDA && swpf==AV_PIX_FMT_YUV444P)) {
                logstream << "cuda_egl_obs_sink: unsupported frame format, expected CUDA/YUV444P";
                // fall back to empty frame
                obs_frame_.width = 0;
                obs_frame_.height = 0;
                obs_frame_.timestamp = prev_timestamp_ + 1;
                prev_timestamp_ = obs_frame_.timestamp;
                outputFrame();
                if (!ticks) this->yieldAndProcess();
                return;
            }

            int W = frm.width();
            int H = frm.height();
            if (!ensure_texture_and_registration(W, H)) {
                logstream << "cuda_egl_obs_sink: ensure_texture_and_registration failed";
                // empty frame
                obs_frame_.width = 0; obs_frame_.height = 0;
                obs_frame_.timestamp = prev_timestamp_ + 1;
                prev_timestamp_ = obs_frame_.timestamp;
                outputFrame();
                if (!ticks) this->yieldAndProcess();
                return;
            }

            if (!run_conversion_to_texture(frm)) {
                logstream << "cuda_egl_obs_sink: conversion failed";
                obs_frame_.width = 0; obs_frame_.height = 0;
                obs_frame_.timestamp = prev_timestamp_ + 1;
            } else {
                obs_frame_.width = W;
                obs_frame_.height = H;
                // carry EGLImage to callback via data[0]
                obs_frame_.data[0] = (uint8_t*)egl_image_;
                obs_frame_.linesize[0] = 0;
                obs_frame_.timestamp = rescaleTS(frm.pts(), av::Rational(1, 1000000000)).timestamp();
                last_frame_emitted_at_ = wallclock.pts();
            }
        } else {
            // timeout - send empty
            obs_frame_.width = 0; obs_frame_.height = 0;
            obs_frame_.timestamp = prev_timestamp_ + 1;
        }

        prev_timestamp_ = obs_frame_.timestamp;
        outputFrame();
        if (!ticks) this->yieldAndProcess();
    }

    virtual void flush()
    {
        this->prohibitProcessNonBlocking();
        obs_frame_ = {0};
        obs_frame_.format = VIDEO_FORMAT_RGBA;
        obs_frame_.timestamp = prev_timestamp_ + 1;
        outputFrame();
    }

    CudaEglObsSink(std::unique_ptr<SourceType> &&source, InstanceData& app_instance): NodeSingleInput(std::move(source)), app_instance_(app_instance)
    {
        std::fill(reinterpret_cast<uint8_t*>(&obs_hw_), reinterpret_cast<uint8_t*>(&obs_hw_)+sizeof(obs_hw_), 0);
        obs_hw_.borrows_frames = true;
        // buffer_to_texture: bind EGLImage to OBS texture
        obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
            CudaEglObsSink &self = *reinterpret_cast<CudaEglObsSink*>(opaque);
            if (!self.EGLImageTargetTexture2DOES_) return;
            EGLImage image = (EGLImage)buf; // passed in obs_frame_.data[0]
            const GLuint gltex = *(GLuint *)gs_texture_get_obj(tex);
            gl_bind_texture(GL_TEXTURE_2D, gltex);
            gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            self.EGLImageTargetTexture2DOES_(GL_TEXTURE_2D, image);
            gl_bind_texture(GL_TEXTURE_2D, 0);
        };
        // free_buffer: nothing to free on OBS thread side
        obs_hw_.free_buffer = [](void* opaque, void* buf) {
            (void)opaque; (void)buf;
        };
    }

    static std::shared_ptr<CudaEglObsSink> create(NodeCreationInfo &nci)
    {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> edge = edges.find<av::VideoFrame>(params["src"]);
        auto r = std::make_shared<CudaEglObsSink>(make_unique<EdgeSource<av::VideoFrame>>(edge), nci.instance);
        if (params.count("max_freeze_duration")) {
            r->timeout_ms_ = params["max_freeze_duration"].get<float>() * 1000 + 0.5;
        }
        if (params.count("unbuffered")) {
            r->unbuffered_ = params["unbuffered"];
        }
        const char* debug_timing = getenv("AVPLUMBER_DEBUG_TIMING");
        r->debug_timing_ = debug_timing && debug_timing[0];
        // Initialize EGL function pointers early
        r->ensure_egl_context();
        // Ensure CUDA context ready
        r->ensure_cuda_context();
        // Pass 'this' as opaque to callbacks
        r->obs_frame_.hw_opaque = r.get();
        return r;
    }
};

DECLNODE(cuda_egl_obs_sink, CudaEglObsSink);

#endif // HAVE_CUDA


