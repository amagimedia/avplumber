#include "../node_common.hpp"

#include "../../../../../../libobs-opengl/gl-subsystem.h"

#include <atomic>
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <mutex>
#include <obs-module.h>

#ifndef HAVE_CUDA
#define HAVE_CUDA 0
#endif

#ifndef HAVE_VAAPI
#define HAVE_VAAPI 0
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
#include <GL/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <libavutil/hwcontext_vaapi.h>
#endif // HAVE_VAAPI

// various parts of this code adapted from OBS source code: deps/media-playback/media-playback/media.c
// Copyright (c) 2017 Hugh Bailey <obs.jim@gmail.com>

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

static inline enum EGLint obs_color_format_to_drm(gs_color_format f) {
    switch (f) {
        case GS_R8: return DRM_FORMAT_R8;
        case GS_R8G8: return DRM_FORMAT_RG88;
        default:;
    }
    return DRM_FORMAT_INVALID;
}

class ObsVideoSink: public NodeSingleInput<av::VideoFrame>, public NonBlockingNode<ObsVideoSink>, public IFlushable {
protected:
    InstanceData& app_instance_;
    struct obs_source_frame obs_frame_ = {0};
    enum AVPixelFormat cur_pix_fmt_ = AV_PIX_FMT_NONE;
    enum AVColorSpace cur_colorspace_ = AVCOL_SPC_NB;
    enum AVColorRange cur_color_range_ = AVCOL_RANGE_NB;
    uint_fast8_t planes_count_ = 0;
    AVTS prev_timestamp_ = 0;
    signed int timeout_ms_ = -1;
    AVTS last_frame_emitted_at_ = 0;
    bool unbuffered_ = false;
    bool debug_timing_ = false;
    struct obs_hw_buffer obs_hw_;
    AVPixelFormat obs_hw_pixel_format_ = AV_PIX_FMT_NONE;
    AVBufferRef* have_hw_info_for_ = nullptr;
    struct FrameInfo {
        std::atomic<ObsVideoSink*> owner;
        av::VideoFrame frame;
        operator bool() const {
            return owner;
        }
        FrameInfo(): owner(nullptr), frame(av::VideoFrame::null()) {
        }
        FrameInfo(const FrameInfo &copyfrom): owner(copyfrom.owner.load()), frame(copyfrom.frame) {
        }
    };
    std::vector<FrameInfo> frames_;
    FrameInfo* findFreeFrame() {
        for (FrameInfo &frm: frames_) {
            if (!frm) {
                return &frm;
            }
        }
        return nullptr;
    }
    bool framesEmpty() {
        for (FrameInfo &frm: frames_) {
            if (frm) {
                return false;
            }
        }
        return true;
    }
    size_t occupiedFramesCount() {
        size_t result = 0;
        for (FrameInfo &frm: frames_) {
            if (frm) {
                result++;
            }
        }
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
    #endif // HAVE_CUDA

    void prepareEmptyFrame() {
        logstream << "outputting empty frame";
        obs_frame_.width = 0;
        obs_frame_.height = 0;
        obs_frame_.hw = nullptr;
        obs_frame_.hw_opaque = nullptr;
        obs_frame_.format = VIDEO_FORMAT_NONE;
        obs_frame_.timestamp = prev_timestamp_ + 1;
    }
    av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm) {
        if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
        AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
        if (ctx == nullptr) return AV_PIX_FMT_NONE;
        return ctx->sw_format;
    }

    void prepareHwInfo(av::VideoFrame &frm) {
        if (obs_hw_pixel_format_ == AV_PIX_FMT_NONE) return;
        if (frm.raw()->hw_frames_ctx == have_hw_info_for_) return; // TODO: is this optimization safe?

        #define CB_COMMON \
            assert(opaque != nullptr); \
            FrameInfo &fi = *reinterpret_cast<FrameInfo*>(opaque); \
            assert(fi.owner != nullptr); \
            ObsVideoSink &self = *fi.owner;

        if (obs_hw_pixel_format_ == AV_PIX_FMT_CUDA) {
            #if HAVE_CUDA
            obs_hw_.borrows_frames = true;
            //cuda_dev_ctx_ = (AVCUDADeviceContext*)((AVHWFramesContext*)frm.raw()->hw_frames_ctx->data)->device_ctx->hwctx;
            if (!global_cu) {
                throw Error("CUDA functions not ready");
            }

            obs_hw_.free_buffer = [](void* opaque, void* buf) {
                CB_COMMON
                if (self.debug_timing_) {
                    logstream << "free_buffer begin";
                }
                fi.frame = av::VideoFrame::null();
                fi.owner.store(nullptr, std::memory_order_release);
                if (self.debug_timing_) {
                    logstream << "free_buffer end";
                }
            };
            obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
                CB_COMMON
                if (self.debug_timing_) {
                    logstream << "buffer_to_texture begin";
                }
                auto cu = global_cu;
                assert(tex->type==GS_TEXTURE_2D);
                struct gs_texture_2d *tex2d = (struct gs_texture_2d*)tex;

                TextureInfo *ti;
                auto titer = global_textures.find(tex);
                CHECK_CU(cu->cuCtxPushCurrent(global_cu_ctx));
                if (titer == global_textures.end()) {
                    // texture not yet in our map
                    logstream << "getting new resource associated with our texture";
                    titer = global_textures.emplace_hint(titer, std::pair<gs_texture_t*, TextureInfo>(tex, {}));
                    ti = &titer->second;
                    // from mpv's video/out/hwdec/hwdec_cuda_gl.c
                    CHECK_CU(cu->cuGraphicsGLRegisterImage(&ti->cu_res, tex->texture, tex->gl_target, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
                    CHECK_CU(cu->cuGraphicsMapResources(1, &ti->cu_res, 0));
                    CHECK_CU(cu->cuGraphicsSubResourceGetMappedArray(&ti->cu_arr, ti->cu_res, 0, 0));
                    CHECK_CU(cu->cuGraphicsUnmapResources(1, &ti->cu_res, 0));
                    tex->on_destroy_callback = [](gs_texture_t *tex) {
                        auto cu = global_cu;
                        auto titer = global_textures.find(tex);
                        if (titer != global_textures.end()) {
                            CHECK_CU(cu->cuCtxPushCurrent(global_cu_ctx));
                            logstream << "unregistering resource associated with our texture";
                            CHECK_CU(cu->cuGraphicsUnregisterResource(titer->second.cu_res));
                            CUcontext dummy;
                            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                            global_textures.erase(titer);
                        }
                    };
                    if (self.debug_timing_) {
                        logstream << "done getting new resource";
                    }
                } else {
                    ti = &titer->second;
                }
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
                if (self.debug_timing_) {
                    logstream << "buffer_to_texture before cuMemcpy";
                }
                if (!CHECK_CU(cu->cuMemcpy2DAsync(&cpy, 0))) {
                    //logstream << "buffer_to_texture cuMemcpy success!";
                } else {
                    logstream << "buffer_to_texture cuMemcpy failure";
                }
                if (self.debug_timing_) {
                    logstream << "buffer_to_texture after cuMemcpy";
                }
                CUcontext dummy;
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                if (self.debug_timing_) {
                    logstream << "buffer_to_texture end";
                }
            };
            obs_hw_.copy_frame_data_plane_from_hw = [](void* opaque,
                                                        struct obs_source_frame *dst,
                                                        const struct obs_source_frame *src,
                                                        uint32_t plane, uint32_t lines) {
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
                if (!CHECK_CU(cu->cuMemcpy2D(&cpy))) {
                    //logstream << "copy_frame_data_plane_from_hw cuMemcpy success!";
                } else {
                    logstream << "copy_frame_data_plane_from_hw cuMemcpy failure";
                }
                CUcontext dummy;
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
            };
            #else
            throw Error("got CUDA frame but compiled without CUDA support");
            #endif
        } else if (obs_hw_pixel_format_ == AV_PIX_FMT_VAAPI) {
            #if HAVE_VAAPI
            obs_hw_.borrows_frames = true;
            obs_hw_.buffer_to_texture = [](void* opaque, gs_texture_t* tex, void* buf, size_t linesize) {
                size_t plane = linesize; // not really, abused as plane index
                VADRMPRIMESurfaceDescriptor *prime = reinterpret_cast<VADRMPRIMESurfaceDescriptor*>(buf);
                EGLint img_attr[] = {
                    EGL_LINUX_DRM_FOURCC_EXT,      obs_color_format_to_drm(tex->format),
                    EGL_WIDTH,                     gs_texture_get_width(tex),
                    EGL_HEIGHT,                    gs_texture_get_height(tex),
                    EGL_DMA_BUF_PLANE0_FD_EXT,     prime.objects[prime.layers[0].object_index[plane]].fd,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT, prime.layers[0].offset[plane],
                    EGL_DMA_BUF_PLANE0_PITCH_EXT,  prime.layers[0].pitch[plane],
                    EGL_NONE
                };
                graphics_t* graphics = gs_get_context(void);

                EGLImage image = eglCreateImageKHR(graphics->device->plat->edisplay, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, img_attr);
                const GLuint gltex = *(GLuint *)gs_texture_get_obj(tex);
                gl_bind_texture(GL_TEXTURE_2D, gltex);
                gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                gl_tex_param_i(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, image);
                if (!gl_success("glEGLImageTargetTexture2DOES")) {
                    logstream << "glEGLImageTargetTexture2DOES failed, VAAPI data not copied to texture";
                }
                gl_bind_texture(GL_TEXTURE_2D, 0);
            };
            obs_hw_.free_buffer = [](void* opaque, void* buf) {
                CB_COMMON
                delete (VADRMPRIMESurfaceDescriptor*)buf;
                fi.frame = av::VideoFrame::null();
                fi.owner.store(nullptr, std::memory_order_release);
            };
            obs_hw_.copy_frame_data_plane_from_hw = nullptr;
            #else
            throw Error("got VAAPI frame but compiled without VAAPI support");
            #endif
        } else {
            throw Error("unsupported hwaccel");
        }
        #undef CB_COMMON
        have_hw_info_for_ = frm.raw()->hw_frames_ctx;
    }
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
    using NodeSingleInput::NodeSingleInput;
    virtual void start() {
        app_instance_.doWithObsSource([this](obs_source_t* s) {
            obs_source_set_async_unbuffered(s, unbuffered_);
        });
    }
    virtual void processNonBlocking(EventLoop& evl, bool ticks) {
        av::VideoFrame *pfrm = this->source_->peek(0);
        if (pfrm==nullptr) {
            //logstream << "no frame";
            bool timelimit = planes_count_ && (timeout_ms_>=0);
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
        if (pfrm && *pfrm) {
            //logstream << "have frame";
            av::VideoFrame frm = *pfrm;
            if (ticks) {
                while (this->source_->pop()) {}; // remove outstanding buffered packets
            } else {
                this->source_->pop();
            }
            enum av::PixelFormat hw_pixel_format;
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

                obs_frame_ = {0};

                bool success = (new_format != VIDEO_FORMAT_NONE) &&
                               video_format_get_parameters(new_space, new_range,
                                   obs_frame_.color_matrix,
                                   obs_frame_.color_range_min,
                                   obs_frame_.color_range_max);

                cur_pix_fmt_ = real_pixel_format;
                planes_count_ = success ? real_pixel_format.planesCount() : 0;
                cur_colorspace_ = frm.raw()->colorspace;
                cur_color_range_ = frm.raw()->color_range;

                obs_frame_.format = new_format;
                obs_frame_.full_range = new_range == VIDEO_RANGE_FULL;

                if (!success) {
                    logstream << "video_format_get_parameters failed, will output empty frames";
                }
            }

            if (planes_count_) {
                obs_hw_pixel_format_ = hw_pixel_format;
                obs_frame_.hw = hw_pixel_format==AV_PIX_FMT_NONE ? nullptr : &obs_hw_;
                prepareHwInfo(frm);
                if (hw_pixel_format == AV_PIX_FMT_CUDA || hw_pixel_format == AV_PIX_FMT_VAAPI) {
                    FrameInfo *fi = findFreeFrame();
                    if (!fi) {
                        logstream << "too many frames buffered, waiting for obs to free some frames";
                        if (!ticks) {
                            this->sleepAndProcess(40);
                        }
                        return;
                    }
                    fi->frame = frm;
                    fi->owner.store(this, std::memory_order_release);
                    obs_frame_.hw_opaque = fi;
                }
                if (hw_pixel_format==AV_PIX_FMT_NONE || hw_pixel_format==AV_PIX_FMT_CUDA) {
                    for (int i=0; i<planes_count_; i++) {
                        obs_frame_.data[i] = frm.raw()->data[i];
                        obs_frame_.linesize[i] = abs(frm.raw()->linesize[i]);
                    }
                } else if (hw_pixel_format==AV_PIX_FMT_VAAPI) {
                    AVVAAPIDeviceContext* hwctx = ((AVVAAPIDeviceContext*)(((AVHWFramesContext*)(frm.raw()->hw_frames_ctx->data))->device_ctx->hwctx));
                    VASurfaceID va_surface = (uintptr_t)frm->data[3];
                    VADRMPRIMESurfaceDescriptor* prime = new VADRMPRIMESurfaceDescriptor;
                    if (vaExportSurfaceHandle(hwctx->display, va_surface,
                        VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                        VA_EXPORT_SURFACE_READ_ONLY | VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                        prime) != VA_STATUS_SUCCESS)
                        { logstream << "vaExportSurfaceHandle failed"; }
                    vaSyncSurface(hwctx->display, va_surface);
                    for (int i=0; i<planes_count_; i++) {
                        obs_frame_.data[i] = (uint8_t*)prime;
                        obs_frame_.linesize[i] = i;
                    }
                }
                obs_frame_.width = frm.width();
                obs_frame_.height = frm.height();
            } else {
                prepareEmptyFrame();
            }
            obs_frame_.timestamp = rescaleTS(frm.pts(), av::Rational(1, 1000000000)).timestamp();
            last_frame_emitted_at_ = wallclock.pts();
        } else {
            // timeout or frame empty
            cur_pix_fmt_ = AV_PIX_FMT_NONE;
            planes_count_ = 0;
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
            if (framesEmpty()) {
                break;
            }
            wallclock.sleepms(50);
            if (wallclock.pts() >= warn_at) {
                logstream << "WARNING: still have " << occupiedFramesCount() << " frames in hold buffer, waiting";
                warn_at = wallclock.pts() + 2000;
            }
        }
    }
    ObsVideoSink(std::unique_ptr<SourceType> &&source, InstanceData& app_instance): NodeSingleInput(std::move(source)), app_instance_(app_instance) {
    }
    static std::shared_ptr<ObsVideoSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> edge = edges.find<av::VideoFrame>(params["src"]);
        auto r = std::make_shared<ObsVideoSink>(make_unique<EdgeSource<av::VideoFrame>>(edge), nci.instance);
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
        r->debug_timing_ = debug_timing && debug_timing[0]; // env var is set and non-empty

        std::fill(reinterpret_cast<uint8_t*>(&r->obs_hw_), reinterpret_cast<uint8_t*>(&r->obs_hw_)+sizeof(obs_hw_), 0);
        #if HAVE_CUDA
        if (global_cu) {
            logstream << "have CUDA functions";
            r->frames_.resize(60);

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
        return r;
    }
};

#if HAVE_CUDA
CUcontext ObsVideoSink::global_cu_ctx = nullptr;
std::unordered_map<gs_texture_t*, ObsVideoSink::TextureInfo> ObsVideoSink::global_textures;
std::mutex ObsVideoSink::global_cu_ctx_create_mutex;
#endif

DECLNODE(obs_video_sink, ObsVideoSink);
