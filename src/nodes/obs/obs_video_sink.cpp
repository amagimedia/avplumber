#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1

#include "../node_common.hpp"

#include <atomic>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <mutex>
#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <graphics/graphics.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "../../hwaccel/EglImagePoolToken.hpp"
#include "../../hwaccel/EglImageFrame.hpp"

#ifndef HAVE_CUDA
#define HAVE_CUDA 0
#endif

#ifndef HAVE_VAAPI
#define HAVE_VAAPI 0
#endif

#if (HAVE_CUDA || HAVE_VAAPI)

#include <GL/gl.h>
#include <GL/glext.h>

struct gs_device {
  struct gl_platform *plat;
};
typedef struct gs_device gs_device_t;

struct fbo_info;

struct gs_texture {
  gs_device_t *device;
  enum gs_texture_type type;
  enum gs_color_format format;
  GLenum gl_format;
  GLenum gl_target;
  GLenum gl_internal_format;
  GLenum gl_type;
  GLuint texture;
  uint32_t levels;
  bool is_dynamic;
  bool is_render_target;
  bool is_dummy;
  bool gen_mipmaps;
  
  gs_samplerstate_t *cur_sampler;
  struct fbo_info *fbo;
  
    void (*on_destroy_callback)(struct gs_texture *itself);
};

struct gs_texture_2d {
  struct gs_texture base;

  uint32_t width;
  uint32_t height;
  bool gen_mipmaps;
  GLuint unpack_buffer;
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

#if (HAVE_CUDA || HAVE_VAAPI)
// Cache glEGLImageTargetTexture2DOES globally; never resolve it in hot path
static void (*g_EGLImageTargetTexture2DOES)(GLenum, void*) = nullptr;
__attribute__((constructor)) static void init_gl_eglimage_fn(void) {
  g_EGLImageTargetTexture2DOES = (void (*)(GLenum, void*))eglGetProcAddress("glEGLImageTargetTexture2DOES");
}
#endif


#endif


#if HAVE_CUDA
#include <ffnvcodec/dynlink_loader.h>
#include <libavutil/hwcontext_cuda.h>

CudaFunctions* global_cu;

int check_cu(CUresult err, const char *func, CudaFunctions *cu) {
    const char *err_name;
    const char *err_string;
    if (err == CUDA_SUCCESS)
        return 0;

    cu->cuGetErrorName(err, &err_name);
    cu->cuGetErrorString(err, &err_string);

    logstream << "cuda function: " << func << " failed: " << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");

    return -1;
}

#define CHECK_CU(x) check_cu((x), #x, (cu))

// we need to do it in module constructor - if we do it in object's constructor, bad things happen (probably race condition)
__attribute__((constructor)) void init(void) {
    if (!cuda_load_functions(&global_cu, nullptr)) {
        logstream << "constructor loaded CUDA";
        auto cu = global_cu;
        if (!CHECK_CU(cu->cuInit(0))) {
            logstream << "constructor initialized CUDA";
        } else {
            logstream << "constructor failed to initialize CUDA";
            global_cu = nullptr;
        }
    } else {
        global_cu = nullptr;
        logstream << "constructor failed to load CUDA";
    }
}

#endif // HAVE_CUDA

#if HAVE_VAAPI
#include <va/va.h>
#include <va/va_drmcommon.h>
#include <drm/drm_fourcc.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <X11/Xlib.h>
#include <libavutil/hwcontext_vaapi.h>

// TODO include it properly from obs sources
struct gl_platform {
    Display *xdisplay;
    EGLDisplay edisplay;
    EGLConfig config;
    EGLContext context;
    EGLSurface pbuffer;
    bool close_xdisplay;
};

struct graphics_subsystem {
    void *module;
    gs_device_t *device;
};

#endif // HAVE_VAAPI

// various parts of this code adapted from OBS source code: deps/media-playback/media-playback/media.c
// Copyright (c) 2017 Lain Bailey <lain@obsproject.com>

static inline enum video_colorspace convert_color_space(enum AVColorSpace s)
{
	return s == AVCOL_SPC_BT709 ? VIDEO_CS_709 : VIDEO_CS_DEFAULT;
}

static inline enum video_range_type convert_color_range(enum AVColorRange r)
{
	return r == AVCOL_RANGE_JPEG ? VIDEO_RANGE_FULL : VIDEO_RANGE_DEFAULT;
}

static inline enum video_format convert_pixel_format(int f)
{
	switch (f) {
	case AV_PIX_FMT_NONE:
		return VIDEO_FORMAT_NONE;
	case AV_PIX_FMT_YUV420P:
		return VIDEO_FORMAT_I420;
	case AV_PIX_FMT_NV12:
		return VIDEO_FORMAT_NV12;
	case AV_PIX_FMT_YUYV422:
		return VIDEO_FORMAT_YUY2;
	case AV_PIX_FMT_YUV444P:
		return VIDEO_FORMAT_I444;
	case AV_PIX_FMT_UYVY422:
		return VIDEO_FORMAT_UYVY;
	case AV_PIX_FMT_RGBA:
		return VIDEO_FORMAT_RGBA;
	case AV_PIX_FMT_BGRA:
		return VIDEO_FORMAT_BGRA;
	case AV_PIX_FMT_BGR0:
		return VIDEO_FORMAT_BGRX;
	case AV_PIX_FMT_YUVA420P:
		return VIDEO_FORMAT_I40A;
	case AV_PIX_FMT_YUVA422P:
		return VIDEO_FORMAT_I42A;
	case AV_PIX_FMT_YUVA444P:
		return VIDEO_FORMAT_YUVA;
	default:;
	}

	return VIDEO_FORMAT_NONE;
}

// MediaSpecific traits for different input types
template<typename T> struct ObsSinkMediaSpecific;

// Forward declare unified sink
template<typename T> class ObsVideoSink;

// Helper to check if a frame is valid (works for both av::VideoFrame and EglImageFrame)
template<typename T> bool isFrameValid(const T& frm);
template<> inline bool isFrameValid<av::VideoFrame>(const av::VideoFrame& frm) {
    return static_cast<bool>(frm);
}
template<> inline bool isFrameValid<EglImageFrame>(const EglImageFrame& frm) {
    return frm.isComplete();
}

// av::VideoFrame specialization
template<> struct ObsSinkMediaSpecific<av::VideoFrame> {
    enum AVPixelFormat cur_pix_fmt_ = AV_PIX_FMT_NONE;
    enum AVColorSpace cur_colorspace_ = AVCOL_SPC_NB;
    enum AVColorRange cur_color_range_ = AVCOL_RANGE_NB;
    uint_fast8_t planes_count_ = 0;
    AVPixelFormat obs_hw_pixel_format_ = AV_PIX_FMT_NONE;
    AVBufferRef* have_hw_info_for_ = nullptr;
    #if HAVE_VAAPI
    //void (EGLAPIENTRY *EGLImageTargetTexture2DOES)(GLenum, GLeglImageOES) = nullptr;
    PFNEGLCREATEIMAGEKHRPROC EGLCreateImageKHR = nullptr;
    #endif
    struct FrameInfo {
        std::atomic<ObsVideoSink<av::VideoFrame>*> owner;
        av::VideoFrame frame;
        operator bool() const { return owner; }
        FrameInfo(): owner(nullptr), frame(av::VideoFrame::null()) {}
        FrameInfo(const FrameInfo &copyfrom): owner(copyfrom.owner.load()), frame(copyfrom.frame) {}
    };
    std::vector<FrameInfo> frames_;
    FrameInfo* findFreeFrame() {
        for (FrameInfo &frm: frames_) if (!frm) return &frm;
        return nullptr;
    }
    bool framesEmpty() {
        for (FrameInfo &frm: frames_) if (frm) return false;
        return true;
    }
    size_t occupiedFramesCount() {
        size_t result = 0;
        for (FrameInfo &frm: frames_) if (frm) result++;
        return result;
    }
    #if HAVE_CUDA
    static CUcontext global_cu_ctx;
    struct TextureInfo {
        CUgraphicsResource cu_res = nullptr;
        CUarray cu_arr = nullptr;
    };
    static std::unordered_map<gs_texture_t*, TextureInfo> global_textures;
    static std::mutex global_cu_ctx_create_mutex;
    #endif
    av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm) {
        if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (ctx == nullptr) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }
    void prepareHwInfo(ObsVideoSink<av::VideoFrame> &vsink, av::VideoFrame &frm);
    void initOnCreate(ObsVideoSink<av::VideoFrame> &vsink);
    void onStart(ObsVideoSink<av::VideoFrame> &) {}
    bool hasFormat() const { return planes_count_ != 0; }
    void composeObsFrame(ObsVideoSink<av::VideoFrame> &vsink, av::VideoFrame &frm, bool ticks);
};

// EglImageFrame specialization
template<> struct ObsSinkMediaSpecific<EglImageFrame> {
    std::atomic<size_t> outstanding_frames_{0};
    void initOnCreate(ObsVideoSink<EglImageFrame> &vsink);
    void onStart(ObsVideoSink<EglImageFrame> &vsink) {
    }
    bool hasFormat() const { return true; }
    bool framesEmpty() const { return outstanding_frames_.load(std::memory_order_acquire) == 0; }
    size_t occupiedFramesCount() const { return outstanding_frames_.load(std::memory_order_acquire); }
    void composeObsFrame(ObsVideoSink<EglImageFrame> &vsink, EglImageFrame &frm, bool /*ticks*/);
};

template<typename T> class ObsVideoSink: public NodeSingleInput<T>, public NonBlockingNode<ObsVideoSink<T>>, public IFlushable {
public:
    struct obs_source_frame obs_frame_ = {0};
    struct obs_hw_buffer obs_hw_;
    bool debug_timing_ = false;
    AVTS last_frame_emitted_at_ = 0;
    
    void prepareEmptyFrame() {
        obs_frame_.width = 0;
        obs_frame_.height = 0;
        obs_frame_.hw = nullptr;
        obs_frame_.hw_opaque = nullptr;
        obs_frame_.format = VIDEO_FORMAT_NONE;
        obs_frame_.timestamp = prev_timestamp_ + 1;
    }
    
    void sleepAndProcessPublic(int ms) {
        this->sleepAndProcess(ms);
    }
protected:
    InstanceData& app_instance_;
    AVTS prev_timestamp_ = 0;
    signed int timeout_ms_ = -1;
    bool unbuffered_ = false;
    ObsSinkMediaSpecific<T> mspec_;
    void outputFrame() {
        bool outputted = app_instance_.doWithObsSource([this](obs_source_t* s) {
            obs_source_output_video(s, &obs_frame_);
        });
        // if outputted==false, it means that source is being destroyed and obs_source_t is already set to nullptr
        // in that case, OBS won't have chance to call free_buffer, we need to do it ourselves:
        if (!outputted && obs_frame_.hw && obs_frame_.hw_opaque && obs_frame_.hw->borrows_frames) {
            obs_frame_.hw->free_buffer(obs_frame_.hw_opaque, obs_frame_.data[0]);
        }
    }
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual void start() {
        app_instance_.doWithObsSource([this](obs_source_t* s) {
            obs_source_set_async_unbuffered(s, unbuffered_);
        });
        mspec_.onStart(*this);
    }
    virtual void processNonBlocking(EventLoop& evl, bool ticks) {
        T *pfrm = this->source_->peek(0);
        if (pfrm==nullptr) {
            //logstream << "no frame";
            bool timelimit = mspec_.hasFormat() && (timeout_ms_>=0);
            if (timelimit && !ticks) {
                // retry after waiting
                this->sleepAndProcess(timeout_ms_);
            }
            if (!ticks) {
                // retry when we have packet in source queue
                this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
            }
            if ((!timelimit) || (wallclock.pts() < last_frame_emitted_at_ + timeout_ms_)) {
                // not waited enough yet for timeout - don't proceed to outputting empty frame
                //logstream << "not waited enough yet for timeout";
                return;
            }
        }
        T frm;
        if (pfrm && isFrameValid(*pfrm)) {
            //logstream << "have frame";
            frm = *pfrm;
            if (ticks && unbuffered_) {
                while (this->source_->pop()) {}; // remove outstanding buffered packets
            } else {
                this->source_->pop();
            }
            mspec_.composeObsFrame(*this, frm, ticks);
        } else {
            // timeout or frame empty
            prepareEmptyFrame();
        }
        prev_timestamp_ = obs_frame_.timestamp;
        outputFrame();
        if (!ticks) {
            // process next packet
            this->yieldAndProcess();
        }
    }
    virtual void flush() {
        this->prohibitProcessNonBlocking();
        prepareEmptyFrame();
        outputFrame();

        AVTS warn_at = wallclock.pts() + 2000;
        while(true) {
            if (mspec_.framesEmpty()) {
                break;
            }
            wallclock.sleepms(50);
            if (wallclock.pts() >= warn_at) {
                logstream << "WARNING: still have " << mspec_.occupiedFramesCount() << " frames in hold buffer, waiting";
                warn_at = wallclock.pts() + 2000;
            }
        }
    }
    ObsVideoSink(std::unique_ptr<typename NodeSingleInput<T>::SourceType> &&source, InstanceData& app_instance): NodeSingleInput<T>(std::move(source)), app_instance_(app_instance) {
    }
    static std::shared_ptr<ObsVideoSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<T>> edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<ObsVideoSink>(make_unique<EdgeSource<T>>(edge), nci.instance);
        if (params.count("max_freeze_duration")) {
            r->timeout_ms_ = params["max_freeze_duration"].get<float>() * 1000 + 0.5;
        }
        if (params.count("unbuffered")) {
            r->unbuffered_ = params["unbuffered"];
        }

        const char* debug_timing = getenv("AVPLUMBER_DEBUG_TIMING");
        if (!debug_timing) {
            debug_timing = getenv("MSE_DEBUG_TIMING");
            if (debug_timing) {
                logstream << "deprecated env var MSE_DEBUG_TIMING set, please use AVPLUMBER_DEBUG_TIMING";
            }
        }
        r->debug_timing_ = debug_timing && debug_timing[0];
        std::fill(reinterpret_cast<uint8_t*>(&r->obs_hw_), reinterpret_cast<uint8_t*>(&r->obs_hw_)+sizeof(obs_hw_), 0);
        r->mspec_.initOnCreate(*r);
        return r;
    }
};

// Register generic OBS sink over specific types using unified class
DECLNODE_ATD_TYPES(obs_video_sink, ObsVideoSink, av::VideoFrame, EglImageFrame);

#if HAVE_CUDA
CUcontext ObsSinkMediaSpecific<av::VideoFrame>::global_cu_ctx = nullptr;
std::unordered_map<gs_texture_t*, ObsSinkMediaSpecific<av::VideoFrame>::TextureInfo> ObsSinkMediaSpecific<av::VideoFrame>::global_textures;
std::mutex ObsSinkMediaSpecific<av::VideoFrame>::global_cu_ctx_create_mutex;
#endif

// Trait method implementations
void ObsSinkMediaSpecific<av::VideoFrame>::initOnCreate(ObsVideoSink<av::VideoFrame> &vsink)
{
    #if HAVE_CUDA
    if (global_cu) {
        // match maximum delay for live source
        frames_.resize(300);
        if (!global_cu_ctx) {
            std::lock_guard<decltype(global_cu_ctx_create_mutex)> lock(global_cu_ctx_create_mutex);
            auto cu = global_cu;
            CUdevice display_dev;
            unsigned int device_count;
            logstream << "ObsVideoSink::create before obs_enter_graphics";
            obs_enter_graphics();
            logstream << "ObsVideoSink::create after obs_enter_graphics";
            if (!CHECK_CU(cu->cuGLGetDevices(&device_count, &display_dev, 1, CU_GL_DEVICE_LIST_ALL))) {
                CHECK_CU(cu->cuCtxCreate(&global_cu_ctx, CU_CTX_SCHED_BLOCKING_SYNC, display_dev));
                CUcontext dummy;
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            }
            logstream << "ObsVideoSink::create before obs_leave_graphics";
            obs_leave_graphics();
            logstream << "ObsVideoSink::create after obs_leave_graphics";
        }
    } else {
        logstream << "not having CUDA functions, hwaccel output will not work";
    }
    #endif
    #if HAVE_VAAPI
    EGLImageTargetTexture2DOES = (void (*)(GLenum, GLeglImageOES))eglGetProcAddress("glEGLImageTargetTexture2DOES");
    EGLCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    frames_.resize(60);
    #endif
}

void ObsSinkMediaSpecific<av::VideoFrame>::prepareHwInfo(ObsVideoSink<av::VideoFrame> &vsink, av::VideoFrame &frm)
{
    if (obs_hw_pixel_format_ == AV_PIX_FMT_NONE) return;
    if (frm.raw()->hw_frames_ctx == have_hw_info_for_) return;

    #define CB_COMMON \
        assert(opaque != nullptr); \
        FrameInfo &fi = *reinterpret_cast<FrameInfo*>(opaque); \
        assert(fi.owner != nullptr); \
        ObsVideoSink<av::VideoFrame> &vsink_ref = *fi.owner;
    
    if (obs_hw_pixel_format_ == AV_PIX_FMT_CUDA) {
        #if HAVE_CUDA
        vsink.obs_hw_.borrows_frames = true;
        if (!global_cu) throw Error("CUDA functions not ready");
        vsink.obs_hw_.free_buffer = [](void* opaque, void* buf) {
            CB_COMMON
            if (vsink_ref.debug_timing_) logstream << "free_buffer begin";
            fi.frame = av::VideoFrame::null();
            fi.owner.store(nullptr, std::memory_order_release);
            if (vsink_ref.debug_timing_) logstream << "free_buffer end";
        };
        vsink.obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
            CB_COMMON
            if (vsink_ref.debug_timing_) logstream << "buffer_to_texture begin";
            auto cu = global_cu;
            assert(tex->type==GS_TEXTURE_2D);
            struct gs_texture_2d *tex2d = (struct gs_texture_2d*)tex;
            TextureInfo *ti;
            auto titer = global_textures.find(tex);
            CHECK_CU(cu->cuCtxPushCurrent(global_cu_ctx));
            if (titer == global_textures.end()) {
                logstream << "getting new resource associated with our texture";
                titer = global_textures.emplace_hint(titer, std::pair<gs_texture_t*, TextureInfo>(tex, {}));
                ti = &titer->second;
                CHECK_CU(cu->cuGraphicsGLRegisterImage(&ti->cu_res, tex->texture, tex->gl_target, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
                CHECK_CU(cu->cuGraphicsMapResources(1, &ti->cu_res, 0));
                CHECK_CU(cu->cuGraphicsSubResourceGetMappedArray(&ti->cu_arr, ti->cu_res, 0, 0));
                CHECK_CU(cu->cuGraphicsUnmapResources(1, &ti->cu_res, 0));
                tex->on_destroy_callback = [](gs_texture_t *tex) {
                    auto cu = global_cu;
                    auto titer = global_textures.find(tex);
                    if (titer != global_textures.end()) {
                        logstream << "unregistering resource associated with our texture";
                        CHECK_CU(cu->cuGraphicsUnregisterResource(titer->second.cu_res));
                        global_textures.erase(titer);
                    }
                };
                if (vsink_ref.debug_timing_) logstream << "done getting new resource";
            } else {
                ti = &titer->second;
            }
            CUstream stream;
            CHECK_CU(cu->cuStreamCreate(&stream, 0));
            CUDA_MEMCPY2D cpy = {
                .srcY = 0,
                .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                .srcDevice = (CUdeviceptr)buf,
                .srcPitch = linesize,
                .dstMemoryType = CU_MEMORYTYPE_ARRAY,
                .dstArray = ti->cu_arr,
                .WidthInBytes = tex2d->width * gs_get_format_bpp(tex->format) / 8,
                .Height = tex2d->height
            };
            if (!CHECK_CU(cu->cuMemcpy2DAsync(&cpy, stream))) {
            } else {
                logstream << "buffer_to_texture cuMemcpy failure";
            }
            CHECK_CU(cu->cuStreamSynchronize(stream));
            CHECK_CU(cu->cuStreamDestroy(stream));
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            if (vsink_ref.debug_timing_) logstream << "buffer_to_texture end";
        };
        vsink.obs_hw_.copy_frame_data_plane_from_hw = [](void* opaque, struct obs_source_frame *dst, const struct obs_source_frame *src, uint32_t plane, uint32_t lines) {
            auto cu = global_cu;
            CHECK_CU(cu->cuCtxPushCurrent(global_cu_ctx));
            CUDA_MEMCPY2D cpy = {
                .srcY = 0,
                .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                .srcDevice = (CUdeviceptr)src->data[plane],
                .srcPitch = src->linesize[plane],
                .dstMemoryType = CU_MEMORYTYPE_HOST,
                .dstHost = dst->data[plane],
                .dstPitch = dst->linesize[plane],
                .WidthInBytes = std::min(src->linesize[plane], dst->linesize[plane]),
                .Height = lines
            };
            (void)CHECK_CU(cu->cuMemcpy2D(&cpy));
            CUcontext dummy;
            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        };
        #else
        throw Error("got CUDA frame but compiled without CUDA support");
        #endif
    } else if (obs_hw_pixel_format_ == AV_PIX_FMT_VAAPI) {
        #if HAVE_VAAPI
        vsink.obs_hw_.borrows_frames = true;
        vsink.obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
            CB_COMMON
            assert(tex->type==GS_TEXTURE_2D);
            if (!buf || !g_EGLImageTargetTexture2DOES) return;
            size_t plane = linesize; // abused as plane index
            VADRMPRIMESurfaceDescriptor *prime = reinterpret_cast<VADRMPRIMESurfaceDescriptor*>(buf);
            EGLint img_attr[] = {
                EGL_LINUX_DRM_FOURCC_EXT,      prime->layers[plane].drm_format,
                EGL_WIDTH,                     gs_texture_get_width(tex),
                EGL_HEIGHT,                    gs_texture_get_height(tex),
                EGL_DMA_BUF_PLANE0_FD_EXT,     prime->objects[prime->layers[plane].object_index[0]].fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT, prime->layers[plane].offset[0],
                EGL_DMA_BUF_PLANE0_PITCH_EXT,  prime->layers[plane].pitch[0],
                EGL_NONE
            };
            AVVAAPIDeviceContext* hwctx = ((AVVAAPIDeviceContext*)(((AVHWFramesContext*)(fi.frame.raw()->hw_frames_ctx->data))->device_ctx->hwctx));
            VASurfaceID va_surface = (VASurfaceID)(uintptr_t)fi.frame.raw()->data[3];
            VAStatus va_sync_res = vaSyncSurface(hwctx->display, va_surface);
            if (va_sync_res != VA_STATUS_SUCCESS) {
                logstream << "vaSyncSurface() error: " << vaErrorStr(va_sync_res);
            }
            graphics_t* graphics = gs_get_context();
            EGLImage image = EGLCreateImageKHR(graphics->device->plat->edisplay, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attr);
            const GLuint gltex = *(GLuint *)gs_texture_get_obj(tex);
            gl_bind_texture(GL_TEXTURE_2D, gltex);
            gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            g_EGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);
            if (!gl_success("glEGLImageTargetTexture2DOES")) {
                logstream << "glEGLImageTargetTexture2DOES failed, VAAPI data not copied to texture";
            }
            gl_bind_texture(GL_TEXTURE_2D, 0);
        };
        vsink.obs_hw_.free_buffer = [](void* opaque, void* buf) {
            CB_COMMON
            if (buf) {
                VADRMPRIMESurfaceDescriptor* desc = (VADRMPRIMESurfaceDescriptor*)buf;
                for (int i = 0; i < desc->num_objects; i++)
                    close(desc->objects[i].fd);
                delete desc;
            }
            fi.frame = av::VideoFrame::null();
            fi.owner.store(nullptr, std::memory_order_release);
        };
        vsink.obs_hw_.copy_frame_data_plane_from_hw = nullptr;
        #else
        throw Error("got VAAPI frame but compiled without VAAPI support");
        #endif
    } else {
        throw Error("unsupported hwaccel");
    }
    #undef CB_COMMON
    have_hw_info_for_ = frm.raw()->hw_frames_ctx;
}

void ObsSinkMediaSpecific<av::VideoFrame>::composeObsFrame(ObsVideoSink<av::VideoFrame> &vsink, av::VideoFrame &frm, bool ticks)
{
    AVPixelFormat hw_pixel_format;
    av::PixelFormat real_pixel_format = getHwSwPixelFormat(frm);
    if (real_pixel_format==AV_PIX_FMT_NONE) {
        real_pixel_format = frm.pixelFormat().get();
        hw_pixel_format = AV_PIX_FMT_NONE;
    } else if ((frm.pixelFormat().get()==AV_PIX_FMT_CUDA) || (frm.pixelFormat().get()==AV_PIX_FMT_VAAPI)) {
        hw_pixel_format = frm.pixelFormat().get();
    } else {
        throw Error("got frame with unsupported hwaccel " + std::string(frm.pixelFormat().name()));
    }
    if (real_pixel_format != cur_pix_fmt_ ||
        frm.raw()->colorspace != cur_colorspace_ ||
        frm.raw()->color_range != cur_color_range_) {
        enum video_colorspace new_space = convert_color_space(frm.raw()->colorspace);
        enum video_range_type new_range = convert_color_range(frm.raw()->color_range);
        enum video_format new_format = convert_pixel_format(real_pixel_format);
        vsink.obs_frame_ = {0};
        bool success = (new_format != VIDEO_FORMAT_NONE) &&
                       video_format_get_parameters(new_space, new_range,
                           vsink.obs_frame_.color_matrix,
                           vsink.obs_frame_.color_range_min,
                           vsink.obs_frame_.color_range_max);
        cur_pix_fmt_ = real_pixel_format;
        planes_count_ = success ? real_pixel_format.planesCount() : 0;
        cur_colorspace_ = frm.raw()->colorspace;
        cur_color_range_ = frm.raw()->color_range;
        vsink.obs_frame_.format = new_format;
        vsink.obs_frame_.full_range = new_range == VIDEO_RANGE_FULL;
        if (!success) {
            logstream << "video_format_get_parameters failed, will output empty frames";
        }
    }
    if (planes_count_) {
        obs_hw_pixel_format_ = hw_pixel_format;
        vsink.obs_frame_.hw = hw_pixel_format==AV_PIX_FMT_NONE ? nullptr : &vsink.obs_hw_;
        prepareHwInfo(vsink, frm);
        if (hw_pixel_format == AV_PIX_FMT_CUDA || hw_pixel_format == AV_PIX_FMT_VAAPI) {
            FrameInfo *fi = findFreeFrame();
            if (!fi) {
                logstream << "too many frames buffered, waiting for obs to free some frames";
                if (!ticks) vsink.sleepAndProcessPublic(40);
                return;
            }
            fi->frame = frm;
            fi->owner.store(&vsink, std::memory_order_release);
            vsink.obs_frame_.hw_opaque = fi;
        }
        if (hw_pixel_format==AV_PIX_FMT_NONE || hw_pixel_format==AV_PIX_FMT_CUDA) {
            for (int i=0; i<planes_count_; i++) {
                vsink.obs_frame_.data[i] = frm.raw()->data[i];
                vsink.obs_frame_.linesize[i] = abs(frm.raw()->linesize[i]);
            }
        } else if (hw_pixel_format==AV_PIX_FMT_VAAPI) {
            #if HAVE_VAAPI
            AVVAAPIDeviceContext* hwctx = ((AVVAAPIDeviceContext*)(((AVHWFramesContext*)(frm.raw()->hw_frames_ctx->data))->device_ctx->hwctx));
            VASurfaceID va_surface = (uintptr_t)frm.raw()->data[3];
            VADRMPRIMESurfaceDescriptor* prime = new VADRMPRIMESurfaceDescriptor;
            VAStatus sts_export = vaExportSurfaceHandle(hwctx->display, va_surface,
                VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2, VA_EXPORT_SURFACE_READ_ONLY, prime);
            if (sts_export != VA_STATUS_SUCCESS) {
                logstream << "vaExportSurfaceHandle failed: " << vaErrorStr(sts_export);
                delete prime; prime = nullptr;
            }
            for (int i=0; i<planes_count_; i++) {
                vsink.obs_frame_.data[i] = (uint8_t*)prime;
                vsink.obs_frame_.linesize[i] = i;
            }
            #else
            throw Error("got VAAPI frame but compiled without VAAPI support");
            #endif
        }
        vsink.obs_frame_.width = frm.width();
        vsink.obs_frame_.height = frm.height();
    } else {
        vsink.prepareEmptyFrame();
    }
    vsink.obs_frame_.timestamp = rescaleTS(frm.pts(), av::Rational(1, 1000000000)).timestamp();
    vsink.last_frame_emitted_at_ = wallclock.pts();
}

void ObsSinkMediaSpecific<EglImageFrame>::initOnCreate(ObsVideoSink<EglImageFrame> &vsink)
{
    std::fill(reinterpret_cast<uint8_t*>(&vsink.obs_hw_), reinterpret_cast<uint8_t*>(&vsink.obs_hw_)+sizeof(vsink.obs_hw_), 0);
    vsink.obs_hw_.type = OBS_HW_BUFFER_EGLIMAGE;
    vsink.obs_hw_.borrows_frames = true;
    vsink.obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
        (void)opaque; (void)linesize;
        if (!g_EGLImageTargetTexture2DOES) return;
        EGLImage image = (EGLImage)buf;
        const GLuint gltex = *(GLuint *)gs_texture_get_obj(tex);
        gl_bind_texture(tex->gl_target, gltex);
        // Ensure single LOD to avoid driver sampling stale mip levels
        gl_tex_param_i(tex->gl_target, GL_TEXTURE_BASE_LEVEL, 0);
        gl_tex_param_i(tex->gl_target, GL_TEXTURE_MAX_LEVEL, 0);
        gl_tex_param_i(tex->gl_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl_tex_param_i(tex->gl_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        g_EGLImageTargetTexture2DOES(tex->gl_target, image);
        if (!gl_success("glEGLImageTargetTexture2DOES")) {
            logstream << "EGLImage -> texture import failed; frame may repeat";
        }
        gl_bind_texture(tex->gl_target, 0);
    };
    vsink.obs_hw_.copy_frame_data_plane_from_hw = [](void* opaque, struct obs_source_frame *dst, const struct obs_source_frame *src, uint32_t plane, uint32_t lines) {
        (void)opaque;
        if (!src || !dst)
            return;
        if (plane != 0)
            return;
        if (!src->data[0] || !dst->data[0])
            return;
        if (!g_EGLImageTargetTexture2DOES)
            return;

        const uint32_t width = src->width;
        const uint32_t height = src->height;
        const uint32_t copy_lines = std::min(lines, height);
        if (!width || !height || !copy_lines)
            return;

        // EglImageFrame path advertises RGBA only.
        if (src->format != VIDEO_FORMAT_RGBA || dst->format != VIDEO_FORMAT_RGBA) {
            logstream << "EGLImage hwdownload only supports RGBA; src=" << (int)src->format
                      << " dst=" << (int)dst->format;
            return;
        }

        const size_t packed_row_bytes = (size_t)width * 4;
        const size_t dst_pitch = dst->linesize[0] ? (size_t)dst->linesize[0] : packed_row_bytes;

        GLint prev_fbo = 0;
        GLint prev_tex_2d = 0;
        GLint prev_pack_align = 4;
        GLint prev_pack_row_length = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex_2d);
        glGetIntegerv(GL_PACK_ALIGNMENT, &prev_pack_align);
#ifdef GL_PACK_ROW_LENGTH
        glGetIntegerv(GL_PACK_ROW_LENGTH, &prev_pack_row_length);
#endif

        GLuint tmp_tex = 0;
        GLuint tmp_fbo = 0;
        glGenTextures(1, &tmp_tex);
        glGenFramebuffers(1, &tmp_fbo);
        if (!tmp_tex || !tmp_fbo) {
            if (tmp_fbo)
                glDeleteFramebuffers(1, &tmp_fbo);
            if (tmp_tex)
                glDeleteTextures(1, &tmp_tex);
            return;
        }

        bool ok = true;

        // Import EGLImage into a temporary GL texture.
        ok = ok && gl_bind_texture(GL_TEXTURE_2D, tmp_tex);
        ok = ok && gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        ok = ok && gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        ok = ok && gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        ok = ok && gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        g_EGLImageTargetTexture2DOES(GL_TEXTURE_2D, (void*)src->data[0]);
        ok = ok && gl_success("glEGLImageTargetTexture2DOES(hwdownload)");

        // Attach to FBO and read back.
        glBindFramebuffer(GL_FRAMEBUFFER, tmp_fbo);
        ok = ok && gl_success("glBindFramebuffer(hwdownload)");
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tmp_tex, 0);
        ok = ok && gl_success("glFramebufferTexture2D(hwdownload)");
        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            logstream << "EGLImage hwdownload FBO incomplete: 0x" << std::hex << status;
            ok = false;
        }

        if (ok) {
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            ok = ok && gl_success("glPixelStorei(GL_PACK_ALIGNMENT)");
#ifdef GL_PACK_ROW_LENGTH
            if ((dst_pitch % 4) == 0) {
                glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)(dst_pitch / 4));
                ok = ok && gl_success("glPixelStorei(GL_PACK_ROW_LENGTH)");
                glReadPixels(0, 0, (GLsizei)width, (GLsizei)copy_lines, GL_RGBA, GL_UNSIGNED_BYTE, dst->data[0]);
                ok = ok && gl_success("glReadPixels(EGLImage hwdownload)");
            } else {
                // Fallback: read line-by-line if stride is unusual (should not happen for RGBA)
                for (uint32_t y = 0; y < copy_lines; y++) {
                    glReadPixels(0, (GLint)y, (GLsizei)width, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                                 dst->data[0] + (size_t)y * dst_pitch);
                    if (!gl_success("glReadPixels(EGLImage hwdownload row)")) {
                        ok = false;
                        break;
                    }
                }
            }
#else
            // Very old GL headers: no GL_PACK_ROW_LENGTH. Use a safe row-by-row read.
            for (uint32_t y = 0; y < copy_lines; y++) {
                glReadPixels(0, (GLint)y, (GLsizei)width, 1, GL_RGBA, GL_UNSIGNED_BYTE,
                             dst->data[0] + (size_t)y * dst_pitch);
                if (!gl_success("glReadPixels(EGLImage hwdownload row)")) {
                    ok = false;
                    break;
                }
            }
#endif
        }

        // Restore bindings/state and delete temp objects.
        glPixelStorei(GL_PACK_ALIGNMENT, prev_pack_align);
#ifdef GL_PACK_ROW_LENGTH
        glPixelStorei(GL_PACK_ROW_LENGTH, prev_pack_row_length);
#endif
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
        gl_bind_texture(GL_TEXTURE_2D, (GLuint)prev_tex_2d);
        glDeleteFramebuffers(1, &tmp_fbo);
        glDeleteTextures(1, &tmp_tex);

        if (!ok)
            return;
    };
    vsink.obs_hw_.free_buffer = [](void* opaque, void* buf) {
        (void)buf;
        EglImageOpaque* box = reinterpret_cast<EglImageOpaque*>(opaque);
        if (box) {
            // Ensure release; token's destructor is idempotent as well
            if (box->token) box->token->releaseOnce();
            if (box->release_cb) box->release_cb();
            delete box; // drops holder shared_ptr; token will be deleted if this was the last ref
        }
    };
}

void ObsSinkMediaSpecific<EglImageFrame>::composeObsFrame(ObsVideoSink<EglImageFrame> &vsink, EglImageFrame &frm, bool)
{
    enum video_colorspace new_space = VIDEO_CS_SRGB;
    enum video_range_type new_range = VIDEO_RANGE_FULL;
    enum video_format new_format = VIDEO_FORMAT_RGBA;
    vsink.obs_frame_ = {0};
    bool success = video_format_get_parameters(new_space, new_range,
                       vsink.obs_frame_.color_matrix,
                       vsink.obs_frame_.color_range_min,
                       vsink.obs_frame_.color_range_max);
    if (!success) {
        logstream << "video_format_get_parameters failed, colors may be wrong?!";
    }
    
    vsink.obs_frame_.format = new_format;
    vsink.obs_frame_.full_range = new_range == VIDEO_RANGE_FULL;
    vsink.obs_frame_.width = frm.width();
    vsink.obs_frame_.height = frm.height();
    vsink.obs_frame_.data[0] = (uint8_t*)frm.image();
    vsink.obs_frame_.linesize[0] = 0;
    vsink.obs_frame_.hw = &vsink.obs_hw_;
    // Pass an opaque box holding a strong ref to the token so it survives until free_buffer
    outstanding_frames_.fetch_add(1, std::memory_order_acq_rel);
    EglImageOpaque* box = new EglImageOpaque;
    box->holder = frm.holder();
    box->token = reinterpret_cast<EglImagePoolToken*>(const_cast<void*>(frm.holder().get()));
    box->release_cb = [this]() {
        outstanding_frames_.fetch_sub(1, std::memory_order_acq_rel);
    };
    vsink.obs_frame_.hw_opaque = box;
    vsink.obs_frame_.timestamp = rescaleTS(frm.pts(), av::Rational(1, 1000000000)).timestamp();
    vsink.last_frame_emitted_at_ = wallclock.pts();
}
