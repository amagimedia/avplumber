#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1

#include "../node_common.hpp"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <dlfcn.h>

#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include <obs-module.h>
#include <graphics/graphics.h>

#ifndef HAVE_CUDA
#define HAVE_CUDA 0
#endif

#include <GL/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <X11/Xlib.h>

//#include <ffnvcodec/dynlink_loader.h>
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"
#include <libavutil/hwcontext_cuda.h>
#include "../../cuda.hpp"

// CU_GL_DEVICE_LIST_ALL constant (not in dynlink header)
#ifndef CU_GL_DEVICE_LIST_ALL
#define CU_GL_DEVICE_LIST_ALL 0x03
#endif

// PTX blob for the conversion kernels (generated at build time)
// Supports YUV444p and YUV420p -> RGBA8 (BT.709 limited)
#include "../../../objs/src/nodes/hwaccel/yuv_to_rgba_709lim_surface.ptx.h"

// Note: we launch the kernel via CUDA Driver API (dynlink), not the runtime.
// We create our own independent EGL/OpenGL context, separate from OBS.

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


static int check_cu(CUresult err, const char *func)
{
    if (err == CUDA_SUCCESS) return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda function: " << func << " failed: " << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
    return -1;
}

#define CHECK_CU(x) check_cu((x), #x)

__attribute__((constructor)) static void init_cuda_cuda_egl_obs_sink(void)
{
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
    CUmodule cu_module_ = nullptr;
    CUfunction cu_kernel_444_ = nullptr;
    CUfunction cu_kernel_420_ = nullptr;
    CUfunction cu_kernel_nv12_ = nullptr;

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
            CHECK_CU(cuGraphicsUnregisterResource(cu_tex_res_));
            cu_tex_res_ = nullptr;
        }
        if (egl_image_ != EGL_NO_IMAGE_KHR && egl_display_ != EGL_NO_DISPLAY) {
            eglDestroyImageKHR(egl_display_, egl_image_);
            egl_image_ = EGL_NO_IMAGE_KHR;
        }
        if (gl_tex_rgba_ && egl_context_ != EGL_NO_CONTEXT) {
            // Make context current to delete texture
            eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
            glDeleteTextures(1, &gl_tex_rgba_);
            gl_tex_rgba_ = 0;
        }
        tex_w_ = tex_h_ = 0;
    }

    void destroy_egl_context()
    {
        destroy_cuda_gl_resources();
        
        if (egl_context_ != EGL_NO_CONTEXT) {
            eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
        }
        if (egl_surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(egl_display_, egl_surface_);
            egl_surface_ = EGL_NO_SURFACE;
        }
        if (egl_display_ != EGL_NO_DISPLAY) {
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
        }
    }

    bool ensure_kernel_loaded()
    {
        if (cu_module_ && cu_kernel_444_ && cu_kernel_420_) return true;
        if (!cu_ctx_) return false;
        CHECK_CU(cuCtxPushCurrent(cu_ctx_));
        if (!cu_module_) {
            if (CHECK_CU(cuModuleLoadData(&cu_module_, (const void*)avpl_yuv_rgba709lim_ptx))) {
                CUcontext dummy;
                CHECK_CU(cuCtxPopCurrent(&dummy));
                logstream << "cuda_egl_obs_sink: cuModuleLoadData failed";
                return false;
            }
            logstream << "cuda_egl_obs_sink: CUDA module loaded from PTX";
        }
        bool ok = true;
        if (!cu_kernel_444_) {
            if (CHECK_CU(cuModuleGetFunction(&cu_kernel_444_, cu_module_, "kYUV444p_to_RGBA8_709lim_surface"))) {
                ok = false;
                logstream << "cuda_egl_obs_sink: cuModuleGetFunction failed for kYUV444p_to_RGBA8_709lim_surface";
            } else {
                logstream << "cuda_egl_obs_sink: CUDA kernel loaded (kYUV444p_to_RGBA8_709lim_surface)";
            }
        }
        if (!cu_kernel_420_) {
            if (CHECK_CU(cuModuleGetFunction(&cu_kernel_420_, cu_module_, "kYUV420p_to_RGBA8_709lim_surface"))) {
                ok = false;
                logstream << "cuda_egl_obs_sink: cuModuleGetFunction failed for kYUV420p_to_RGBA8_709lim_surface";
            } else {
                logstream << "cuda_egl_obs_sink: CUDA kernel loaded (kYUV420p_to_RGBA8_709lim_surface)";
            }
        }
        if (!cu_kernel_nv12_) {
            if (CHECK_CU(cuModuleGetFunction(&cu_kernel_nv12_, cu_module_, "kNV12_to_RGBA8_709lim_surface"))) {
                ok = false;
                logstream << "cuda_egl_obs_sink: cuModuleGetFunction failed for kNV12_to_RGBA8_709lim_surface";
            } else {
                logstream << "cuda_egl_obs_sink: CUDA kernel loaded (kNV12_to_RGBA8_709lim_surface)";
            }
        }
        CUcontext dummy;
        CHECK_CU(cuCtxPopCurrent(&dummy));
        return ok;
    }

    bool ensure_egl_context()
    {
        if (egl_context_ != EGL_NO_CONTEXT) return true;

        // Create our own independent EGL display
        egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_display_ == EGL_NO_DISPLAY) {
            logstream << "cuda_egl_obs_sink: eglGetDisplay failed";
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL display created";

        EGLint major, minor;
        if (!eglInitialize(egl_display_, &major, &minor)) {
            logstream << "cuda_egl_obs_sink: eglInitialize failed";
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL initialized (version " << major << "." << minor << ")";

        // Bind desktop OpenGL API (avoid ES/OpenGL API mismatch on this thread)
        if (!eglBindAPI(EGL_OPENGL_API)) {
            logstream << "cuda_egl_obs_sink: eglBindAPI(EGL_OPENGL_API) failed";
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL OpenGL API bound";

        // Choose our own config (RGBA8, OpenGL core compatible)
        EGLint config_attrs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE,   EGL_PBUFFER_BIT,
            EGL_RED_SIZE,       8,
            EGL_GREEN_SIZE,     8,
            EGL_BLUE_SIZE,      8,
            EGL_ALPHA_SIZE,     8,
            EGL_NONE
        };
        EGLint num_configs;
        if (!eglChooseConfig(egl_display_, config_attrs, &egl_config_, 1, &num_configs) || num_configs == 0) {
            logstream << "cuda_egl_obs_sink: eglChooseConfig failed";
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL config selected";

        // Create our own independent OpenGL context (core 3.3 if supported)
        EGLint ctx_attr[] = {
            #ifdef EGL_CONTEXT_OPENGL_PROFILE_MASK
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            #endif
            #ifdef EGL_CONTEXT_MAJOR_VERSION
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 3,
            #endif
            EGL_NONE
        };
        egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctx_attr);
        if (egl_context_ == EGL_NO_CONTEXT) {
            EGLint error = eglGetError();
            logstream << "cuda_egl_obs_sink: eglCreateContext failed, error: 0x" << std::hex << error;
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL context created";

        // Create pbuffer surface for context
        EGLint pbuf_attr[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
        egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pbuf_attr);
        if (egl_surface_ == EGL_NO_SURFACE) {
            EGLint error = eglGetError();
            logstream << "cuda_egl_obs_sink: eglCreatePbufferSurface failed, error: 0x" << std::hex << error;
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL pbuffer surface created";

        // Make context current to initialize OpenGL
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
            EGLint error = eglGetError();
            logstream << "cuda_egl_obs_sink: eglMakeCurrent failed, error: 0x" << std::hex << error;
            eglDestroySurface(egl_display_, egl_surface_);
            egl_surface_ = EGL_NO_SURFACE;
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL context made current";

        // Load EGL/GL extension functions
        EGLCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
        EGLImageTargetTexture2DOES_ = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
        if (!EGLCreateImageKHR_ || !EGLImageTargetTexture2DOES_) {
            logstream << "cuda_egl_obs_sink: failed to load EGL/GL image functions";
            eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroySurface(egl_display_, egl_surface_);
            egl_surface_ = EGL_NO_SURFACE;
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL/GL extension functions loaded";

        // Verify OpenGL is initialized by checking a simple call
        GLuint test_tex;
        glGenTextures(1, &test_tex);
        if (glGetError() != GL_NO_ERROR) {
            logstream << "cuda_egl_obs_sink: OpenGL initialization failed";
            eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroySurface(egl_display_, egl_surface_);
            egl_surface_ = EGL_NO_SURFACE;
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
            eglTerminate(egl_display_);
            egl_display_ = EGL_NO_DISPLAY;
            return false;
        }
        glDeleteTextures(1, &test_tex);
        logstream << "cuda_egl_obs_sink: OpenGL verified and ready";

        return true;
    }

    bool ensure_cuda_context()
    {
        if (cu_ctx_) return true;
        if (!ensure_egl_context()) return false;
        if (global_cuda.has_errors) {
            logstream << "cuda_egl_obs_sink: global CUDA not initialized";
        } else {
            logstream << "cuda_egl_obs_sink: global CUDA initialized";
        }
        return !global_cuda.has_errors;
    }

    bool ensure_texture_and_registration(int W, int H)
    {
        if (!ensure_egl_context()) return false;
        if (!ensure_cuda_context()) return false;

        if (W == tex_w_ && H == tex_h_ && gl_tex_rgba_ && egl_image_ && cu_tex_res_) return true;

        // recreate
        destroy_cuda_gl_resources();

        // Re-bind API to avoid mismatch if other code changed it on this thread
        eglBindAPI(EGL_OPENGL_API);
        // If a different context is current on this thread, switch to ours
        if (eglGetCurrentContext() != egl_context_ || eglGetCurrentDisplay() != egl_display_ || eglGetCurrentSurface(EGL_DRAW) != egl_surface_) {
            eglBindAPI(EGL_OPENGL_API);
            if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
                EGLint error = eglGetError();
                logstream << "cuda_egl_obs_sink: eglMakeCurrent before GL create failed, error: 0x" << std::hex << error;
                // Attempt to recreate pbuffer surface and retry
                if (egl_surface_ != EGL_NO_SURFACE) {
                    eglDestroySurface(egl_display_, egl_surface_);
                    egl_surface_ = EGL_NO_SURFACE;
                }
                EGLint pbuf_attr2[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
                egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pbuf_attr2);
                if (egl_surface_ == EGL_NO_SURFACE) {
                    EGLint e2 = eglGetError();
                    logstream << "cuda_egl_obs_sink: recreate eglCreatePbufferSurface failed, error: 0x" << std::hex << e2;
                    return false;
                }
                if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
                    EGLint e3 = eglGetError();
                    logstream << "cuda_egl_obs_sink: eglMakeCurrent retry failed, error: 0x" << std::hex << e3;
                    return false;
                }
                logstream << "cuda_egl_obs_sink: eglMakeCurrent succeeded after pbuffer recreate";
            }
        }

        glGenTextures(1, &gl_tex_rgba_);
        gl_bind_texture(GL_TEXTURE_2D, gl_tex_rgba_);
        gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl_bind_texture(GL_TEXTURE_2D, 0);
        logstream << "cuda_egl_obs_sink: OpenGL texture created (" << W << "x" << H << ")";

        EGLint img_attrs[] = { EGL_GL_TEXTURE_LEVEL_KHR, 0, EGL_NONE };
        egl_image_ = EGLCreateImageKHR_(egl_display_, egl_context_, EGL_GL_TEXTURE_2D_KHR, (EGLClientBuffer)(uintptr_t)gl_tex_rgba_, img_attrs);
        if (egl_image_ == EGL_NO_IMAGE_KHR) {
            logstream << "cuda_egl_obs_sink: eglCreateImageKHR failed";
            destroy_cuda_gl_resources();
            return false;
        }
        logstream << "cuda_egl_obs_sink: EGL image created from texture";

        CHECK_CU(cuCtxPushCurrent(cu_ctx_));
        if (CHECK_CU(cuGraphicsGLRegisterImage(&cu_tex_res_, gl_tex_rgba_, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD | CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST))) {
            logstream << "cuda_egl_obs_sink: cuGraphicsGLRegisterImage failed";
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            destroy_cuda_gl_resources();
            return false;
        }
        CUcontext dummy;
        CHECK_CU(cuCtxPopCurrent(&dummy));
        logstream << "cuda_egl_obs_sink: CUDA-GL interop resource registered";

        tex_w_ = W;
        tex_h_ = H;
        return true;
    }

    bool run_conversion_to_texture(const av::VideoFrame &frm)
    {
        if (!cu_ctx_ || !cu_tex_res_) return false;
        // Validate input planes depending on subsampling
        if (!frm.raw()->data[0] || frm.raw()->linesize[0] <= 0) {
            logstream << "cuda_egl_obs_sink: invalid Y plane";
            return false;
        }
        if (cur_sw_pix_fmt_ == AV_PIX_FMT_YUV444P || cur_sw_pix_fmt_ == AV_PIX_FMT_YUV420P) {
            if (!frm.raw()->data[1] || !frm.raw()->data[2] ||
                frm.raw()->linesize[1] <= 0 || frm.raw()->linesize[2] <= 0) {
                logstream << "cuda_egl_obs_sink: invalid U/V planes for YUV planar format";
                return false;
            }
        } else if (cur_sw_pix_fmt_ == AV_PIX_FMT_NV12) {
            if (!frm.raw()->data[1] || frm.raw()->linesize[1] <= 0) {
                logstream << "cuda_egl_obs_sink: invalid UV plane for NV12";
                return false;
            }
        } else {
            logstream << "cuda_egl_obs_sink: unsupported SW pixel format in validation";
            return false;
        }
        CHECK_CU(cuCtxPushCurrent(cu_ctx_));

        if (CHECK_CU(cuGraphicsMapResources(1, &cu_tex_res_, 0))) {
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: cuGraphicsMapResources failed";
            return false;
        }

        CUarray cu_arr = nullptr;
        if (CHECK_CU(cuGraphicsSubResourceGetMappedArray(&cu_arr, cu_tex_res_, 0, 0))) {
            CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: cuGraphicsSubResourceGetMappedArray failed";
            return false;
        }

        if (!cuSurfObjectCreate || !cuSurfObjectDestroy) {
            CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: surface object functions not available";
            return false;
        }

        CUDA_RESOURCE_DESC rdesc; memset(&rdesc, 0, sizeof(rdesc));
        rdesc.resType = CU_RESOURCE_TYPE_ARRAY;
        rdesc.res.array.hArray = cu_arr;
        CUsurfObject surf = 0;
        if (CHECK_CU(cuSurfObjectCreate(&surf, &rdesc))) {
            CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            logstream << "cuda_egl_obs_sink: cuSurfObjectCreate failed";
            return false;
        }

        // Device pointers/pitches must match kernel signature types exactly
        CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
        CUdeviceptr dU = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
        CUdeviceptr dV = (CUdeviceptr)(uintptr_t)frm.raw()->data[2];
        size_t pitchY = (size_t)frm.raw()->linesize[0];
        size_t pitchU = (size_t)frm.raw()->linesize[1];
        size_t pitchV = (size_t)frm.raw()->linesize[2];

        if (!ensure_kernel_loaded()) {
            cuSurfObjectDestroy(surf);
            CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            return false;
        }

        // Setup launch parameters
        void* args[] = {
            // Must match kYUV444p_to_RGBA8_709lim_surface parameter order:
            // (Y, pitchY, U, pitchU, V, pitchV, surfOut, W, H)
            (void*)&dY,
            (void*)&pitchY,
            (void*)&dU,
            (void*)&pitchU,
            (void*)&dV,
            (void*)&pitchV,
            (void*)&surf,
            (void*)&tex_w_,
            (void*)&tex_h_
        };

        unsigned int blockX = 32, blockY = 8;
        unsigned int gridX = (tex_w_ + blockX - 1) / blockX;
        unsigned int gridY = (tex_h_ + blockY - 1) / blockY;

        CUfunction kfun = cu_kernel_444_;
        if (cur_sw_pix_fmt_ == AV_PIX_FMT_YUV420P) kfun = cu_kernel_420_;
        else if (cur_sw_pix_fmt_ == AV_PIX_FMT_NV12) kfun = cu_kernel_nv12_;
        if (CHECK_CU(cuLaunchKernel(kfun,
                                        gridX, gridY, 1,
                                        blockX, blockY, 1,
                                        0, 0, args, nullptr))) {
            logstream << "cuda_egl_obs_sink: cuLaunchKernel failed";
            cuSurfObjectDestroy(surf);
            CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
            CUcontext dummy;
            CHECK_CU(cuCtxPopCurrent(&dummy));
            return false;
        }

        // Ensure completion before unmap so OBS sees complete frame
        CHECK_CU(cuCtxSynchronize());
        cuSurfObjectDestroy(surf);

        CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res_, 0));
        CUcontext dummy;
        CHECK_CU(cuCtxPopCurrent(&dummy));
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
        obs_frame_.hw_opaque = this;

        if (pfrm && *pfrm) {
            av::VideoFrame frm = *pfrm;
            if (ticks) {
                while (this->source_->pop()) {};
            } else {
                this->source_->pop();
            }

            av::PixelFormat swpf = getHwSwPixelFormat(frm);
            bool supported = (frm.pixelFormat().get() == AV_PIX_FMT_CUDA) && (swpf == AV_PIX_FMT_YUV444P || swpf == AV_PIX_FMT_YUV420P || swpf == AV_PIX_FMT_NV12);
            if (!supported) {
                logstream << "cuda_egl_obs_sink: unsupported frame format, expected CUDA/YUV444P or CUDA/YUV420P";
                // fall back to empty frame
                obs_frame_.width = 0;
                obs_frame_.height = 0;
                obs_frame_.timestamp = prev_timestamp_ + 1;
                prev_timestamp_ = obs_frame_.timestamp;
                outputFrame();
                if (!ticks) this->yieldAndProcess();
                return;
            }

            // Remember current SW pixel format for kernel selection
            cur_sw_pix_fmt_ = (AVPixelFormat)swpf;

            // On first frame, adopt decoder's CUDA context to avoid cross-context device pointers
            if (!cu_ctx_) {
                if (frm.raw()->hw_frames_ctx && frm.raw()->hw_frames_ctx->data) {
                    AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
                    if (fctx && fctx->device_ctx && fctx->device_ctx->hwctx) {
                        AVCUDADeviceContext* dc = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
                        cu_ctx_ = dc->cuda_ctx;
                        if (!cu_ctx_) {
                            logstream << "cuda_egl_obs_sink: frame's CUDA context is null";
                        } else {
                            logstream << "cuda_egl_obs_sink: adopted decoder CUDA context";
                        }
                    }
                }
                if (!cu_ctx_) {
                    logstream << "cuda_egl_obs_sink: cannot adopt CUDA context from frame";
                }
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
        // Defer EGL/CUDA initialization to processNonBlocking thread

        return r;
    }
};

DECLNODE(cuda_egl_obs_sink, CudaEglObsSink);
