#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1

#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../hwaccel/EglImageFrame.hpp"
#include "../../hwaccel/CudaEglImagePool.hpp"
#include "../../hwaccel/EglImagePoolToken.hpp"

#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>

#include <GL/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <dlfcn.h>
#include <mutex>
#include <vector>
#include <optional>
#include <string>

// CUDA driver API (dynlink)
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"
#include <libavutil/hwcontext_cuda.h>

// PTX blob for the conversion kernels (generated at build time)
#include "../../../objs/src/nodes/hwaccel/yuv_to_rgba_709lim_surface.ptx.h"

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

class CudaToEglImage: public NodeSISO<av::VideoFrame, EglImageFrame>, public NonBlockingNode<CudaToEglImage>, public IFlushable {
protected:
	// EGL/GL
	EGLDisplay egl_display_ = EGL_NO_DISPLAY;
	EGLContext egl_context_ = EGL_NO_CONTEXT;
	EGLSurface egl_surface_ = EGL_NO_SURFACE;
	EGLConfig egl_config_ = nullptr;
	PFNGLEGLIMAGETARGETTEXTURE2DOESPROC EGLImageTargetTexture2DOES_ = nullptr;
	PFNEGLCREATEIMAGEKHRPROC EGLCreateImageKHR_ = nullptr;

	// External shared pool
	std::shared_ptr<CudaEglImagePool> pool_;
	std::string pool_id_;
	size_t pool_index_ = 0;
	int pool_size_ = 3; // default; can be overridden by params

	// CUDA
	CUcontext cu_ctx_ = nullptr; // adopted from incoming frame
	CUmodule cu_module_ = nullptr;
	CUfunction cu_kernel_444_ = nullptr;
	CUfunction cu_kernel_420_ = nullptr;
	CUfunction cu_kernel_nv12_ = nullptr;

	// helpers
	av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm)
	{
		if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
		AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
		if (ctx == nullptr) return AV_PIX_FMT_NONE;
		return ctx->sw_format;
	}
	void destroy_cuda_gl_resources()
	{
		// Pool resources are managed externally by CudaEglImagePool
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
	bool ensure_egl_context()
	{
		if (egl_context_ != EGL_NO_CONTEXT) return true;
		egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (egl_display_ == EGL_NO_DISPLAY) {
			logstream << "cuda_to_egl_image: eglGetDisplay failed";
			return false;
		}
		EGLint major, minor;
		if (!eglInitialize(egl_display_, &major, &minor)) {
			logstream << "cuda_to_egl_image: eglInitialize failed";
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		if (!eglBindAPI(EGL_OPENGL_API)) {
			logstream << "cuda_to_egl_image: eglBindAPI(EGL_OPENGL_API) failed";
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
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
			logstream << "cuda_to_egl_image: eglChooseConfig failed";
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLint ctx_attr[] = {
			#ifdef EGL_CONTEXT_MAJOR_VERSION
			EGL_CONTEXT_MAJOR_VERSION, 3,
			EGL_CONTEXT_MINOR_VERSION, 3,
			#endif
			EGL_NONE
		};
		egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctx_attr);
		if (egl_context_ == EGL_NO_CONTEXT) {
			logstream << "cuda_to_egl_image: eglCreateContext failed";
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLint pbuf_attr[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
		egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pbuf_attr);
		if (egl_surface_ == EGL_NO_SURFACE) {
			logstream << "cuda_to_egl_image: eglCreatePbufferSurface failed";
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
			logstream << "cuda_to_egl_image: eglMakeCurrent failed";
			eglDestroySurface(egl_display_, egl_surface_);
			egl_surface_ = EGL_NO_SURFACE;
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		EGLCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
		EGLImageTargetTexture2DOES_ = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
		if (!EGLCreateImageKHR_ || !EGLImageTargetTexture2DOES_) {
			logstream << "cuda_to_egl_image: failed to load EGL image funcs";
			eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
			eglDestroySurface(egl_display_, egl_surface_);
			egl_surface_ = EGL_NO_SURFACE;
			eglDestroyContext(egl_display_, egl_context_);
			egl_context_ = EGL_NO_CONTEXT;
			eglTerminate(egl_display_);
			egl_display_ = EGL_NO_DISPLAY;
			return false;
		}
		return true;
	}
	bool ensure_kernel_loaded()
	{
		if (cu_module_ && cu_kernel_444_ && cu_kernel_420_ && cu_kernel_nv12_) return true;
		if (!cu_ctx_) return false;
		CHECK_CU(cuCtxPushCurrent(cu_ctx_));
		if (!cu_module_) {
			char error_log[8192] = {0};
			char info_log[8192]  = {0};
			CUjit_option opts[4];
			void*        optvals[4];
			opts[0] = CU_JIT_ERROR_LOG_BUFFER;
			optvals[0] = error_log;
			opts[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
			size_t err_size = sizeof(error_log);
			optvals[1] = (void*)err_size;
			opts[2] = CU_JIT_INFO_LOG_BUFFER;
			optvals[2] = info_log;
			opts[3] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
			size_t info_size = sizeof(info_log);
			optvals[3] = (void*)info_size;
			// Ensure PTX is null-terminated for JIT
			const std::string ptx_str(avpl_yuv_rgba709lim_ptx, avpl_yuv_rgba709lim_ptx + avpl_yuv_rgba709lim_ptx_len);
			if (CHECK_CU(cuModuleLoadDataEx(&cu_module_, (const void*)ptx_str.c_str(), 4, opts, optvals))) {
				CUcontext dummy;
				CHECK_CU(cuCtxPopCurrent(&dummy));
				logstream << "cuda_to_egl_image: cuModuleLoadDataEx failed";
				if (error_log[0]) logstream << "CUDA JIT error log: " << error_log;
				if (info_log[0])  logstream << "CUDA JIT info log: "  << info_log;
				return false;
			}
			if (info_log[0]) logstream << "CUDA JIT info log: " << info_log;
		}
		bool ok = true;
		if (!cu_kernel_444_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_444_, cu_module_, "kYUV444p_to_RGBA8_709lim_surface"));
		if (!cu_kernel_420_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_420_, cu_module_, "kYUV420p_to_RGBA8_709lim_surface"));
		if (!cu_kernel_nv12_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_nv12_, cu_module_, "kNV12_to_RGBA8_709lim_surface"));
		CUcontext dummy;
		CHECK_CU(cuCtxPopCurrent(&dummy));
		return ok;
	}
	bool ensure_pool(int W, int H)
	{
		if (!ensure_egl_context()) {
			logstream << "cuda_to_egl_image: ensure_egl_context failed in ensure_pool";
			return false;
		}
		if (!pool_) {
			logstream << "cuda_to_egl_image: pool_ is null";
			return false;
		}
		if (pool_->initializedFor(W, H)) {
			return true;
		}
		eglBindAPI(EGL_OPENGL_API);
		if (eglGetCurrentContext() != egl_context_) {
			if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
				logstream << "cuda_to_egl_image: eglMakeCurrent before pool reinit failed";
				return false;
			}
		}
		if (!pool_->reinit(W, H, std::max(1, pool_size_), egl_display_, egl_context_, EGLCreateImageKHR_, cu_ctx_)) {
			logstream << "cuda_to_egl_image: pool reinit failed";
			return false;
		}
		pool_index_ = 0;
		return true;
	}
	bool run_conversion_to_texture(const av::VideoFrame &frm, AVPixelFormat swfmt, CUgraphicsResource cu_tex_res, int tex_w, int tex_h)
	{
		if (!cu_ctx_ || !cu_tex_res) {
			logstream << "cuda_to_egl_image: run_conversion_to_texture failed - cu_ctx_=" << (void*)cu_ctx_ << " cu_tex_res=" << (void*)cu_tex_res;
			return false;
		}
		CHECK_CU(cuCtxPushCurrent(cu_ctx_));
		if (CHECK_CU(cuGraphicsMapResources(1, &cu_tex_res, 0))) {
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			logstream << "cuda_to_egl_image: cuGraphicsMapResources failed";
			return false;
		}
		CUarray cu_arr = nullptr;
		if (CHECK_CU(cuGraphicsSubResourceGetMappedArray(&cu_arr, cu_tex_res, 0, 0))) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			logstream << "cuda_to_egl_image: cuGraphicsSubResourceGetMappedArray failed";
			return false;
		}
		if (!cuSurfObjectCreate || !cuSurfObjectDestroy) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			logstream << "cuda_to_egl_image: surface object functions not available";
			return false;
		}
		CUDA_RESOURCE_DESC rdesc; memset(&rdesc, 0, sizeof(rdesc));
		rdesc.resType = CU_RESOURCE_TYPE_ARRAY;
		rdesc.res.array.hArray = cu_arr;
		CUsurfObject surf = 0;
		if (CHECK_CU(cuSurfObjectCreate(&surf, &rdesc))) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			logstream << "cuda_to_egl_image: cuSurfObjectCreate failed";
			return false;
		}
		CUdeviceptr dY = (CUdeviceptr)(uintptr_t)frm.raw()->data[0];
		CUdeviceptr dU = (CUdeviceptr)(uintptr_t)frm.raw()->data[1];
		CUdeviceptr dV = (CUdeviceptr)(uintptr_t)frm.raw()->data[2];
		size_t pitchY = (size_t)frm.raw()->linesize[0];
		size_t pitchU = (size_t)frm.raw()->linesize[1];
		size_t pitchV = (size_t)frm.raw()->linesize[2];
		if (!ensure_kernel_loaded()) {
			logstream << "cuda_to_egl_image: ensure_kernel_loaded failed";
			cuSurfObjectDestroy(surf);
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			return false;
		}
		void* args[] = {
			(void*)&dY,
			(void*)&pitchY,
			(void*)&dU,
			(void*)&pitchU,
			(void*)&dV,
			(void*)&pitchV,
			(void*)&surf,
			(void*)&tex_w,
			(void*)&tex_h
		};
		unsigned int blockX = 32, blockY = 8;
		unsigned int gridX = (tex_w + blockX - 1) / blockX;
		unsigned int gridY = (tex_h + blockY - 1) / blockY;
		CUfunction kfun = cu_kernel_444_;
		if (swfmt == AV_PIX_FMT_YUV420P) kfun = cu_kernel_420_;
		else if (swfmt == AV_PIX_FMT_NV12) kfun = cu_kernel_nv12_;
		if (CHECK_CU(cuLaunchKernel(kfun,
		                            gridX, gridY, 1,
		                            blockX, blockY, 1,
		                            0, 0, args, nullptr))) {
			logstream << "cuda_to_egl_image: cuLaunchKernel failed";
			cuSurfObjectDestroy(surf);
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			CUcontext dummy;
			CHECK_CU(cuCtxPopCurrent(&dummy));
			return false;
		}
		//CHECK_CU(cuCtxSynchronize());
		cuSurfObjectDestroy(surf);
		CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
		CUcontext dummy;
		CHECK_CU(cuCtxPopCurrent(&dummy));
		return true;
	}
    std::optional<EglImageFrame> waiting_frame_;
public:
	using NodeSISO<av::VideoFrame, EglImageFrame>::NodeSISO;
	virtual void processNonBlocking(EventLoop& evl, bool ticks)
	{
        if (waiting_frame_) {
            if (!this->sink_->put(*waiting_frame_, true)) {
                //logstream << "cuda_to_egl_image: sink put failed, will retry on sink consumed event";
                if (!ticks) {
                    this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                }
                return;
            } else {
                waiting_frame_ = std::nullopt;
            }
        }
		av::VideoFrame *pfrm = this->source_->peek(0);
		if (pfrm==nullptr) {
			if (!ticks) {
				this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
			}
			return;
		}
		av::VideoFrame frm = *pfrm;
		if (!frm) {
			logstream << "cuda_to_egl_image: frame is invalid/null, discarding";
			this->source_->pop(); // discard
			return;
		}
		// adopt CUDA context from frame (once)
		if (!cu_ctx_) {
			if (frm.raw()->hw_frames_ctx && frm.raw()->hw_frames_ctx->data) {
				AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
				if (fctx && fctx->device_ctx && fctx->device_ctx->hwctx) {
					AVCUDADeviceContext* dc = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
					cu_ctx_ = dc->cuda_ctx;
				} else {
					logstream << "cuda_to_egl_image: frame has hw_frames_ctx but device_ctx/hwctx is null";
				}
			} else {
				logstream << "cuda_to_egl_image: frame has no hw_frames_ctx or data is null";
			}
			if (!cu_ctx_) {
				logstream << "cuda_to_egl_image: cannot adopt CUDA context from frame";
				this->source_->pop();
				return;
			}
		}
		int W = frm.width();
		int H = frm.height();
		AVPixelFormat swfmt = getHwSwPixelFormat(frm);
		if (!(swfmt == AV_PIX_FMT_YUV444P || swfmt == AV_PIX_FMT_YUV420P || swfmt == AV_PIX_FMT_NV12)) {
			logstream << "cuda_to_egl_image: unsupported SW pixel format " << swfmt;
			this->source_->pop();
			return;
		}
		if (!ensure_pool(W, H)) {
			logstream << "cuda_to_egl_image: ensure_pool failed";
			this->source_->pop();
			return;
		}
		auto opt_idx = pool_->acquire();
		if (!opt_idx) {
			logstream << "cuda_to_egl_image: pool acquire failed, no available entry; will retry when resources free";
			// Don't drop the source frame; wait for resources to free up and retry soon
			if (!ticks) {
				// Retry when downstream frees space (likely correlates with pool entry release)
				//this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
				// Also schedule a short delay to poll in case pool release happens outside edge consumption timing
				this->sleepAndProcess(10);
			}
			return;
		}
		const size_t idx = *opt_idx;
		auto &entry = pool_->entry(idx);
		if (!run_conversion_to_texture(frm, swfmt, entry.cu_tex_res, W, H)) {
			logstream << "cuda_to_egl_image: run_conversion_to_texture failed, releasing pool entry";
			pool_->release(idx);
			this->source_->pop();
			return;
		}
		// Create a token (owned by shared_ptr) to release the entry back to pool
		auto token_sp = std::shared_ptr<EglImagePoolToken>(new EglImagePoolToken{
			.release = [p = pool_, index = (int)idx]() {
				p->release(index);
			}
		});
		// Holder owns the token; token captures pool_, so pool stays alive
		std::shared_ptr<void> holder = token_sp;
		EglImageFrame out(entry.egl_image, W, H, frm.pts(), frm.timeBase(), holder);
		if (!this->sink_->put(out, true)) {
            waiting_frame_ = std::make_optional(out);
			//logstream << "cuda_to_egl_image: sink put failed (no space), will retry on sink consumed event";
			// no space, retry on sink consumed
			this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
		} else {
			this->source_->pop();
			if (!ticks) this->yieldAndProcess();
		}
	}
	virtual void flush()
	{
		this->prohibitProcessNonBlocking();
	}
	CudaToEglImage(std::unique_ptr<typename NodeSISO<av::VideoFrame, EglImageFrame>::SourceType> &&source, std::unique_ptr<typename NodeSISO<av::VideoFrame, EglImageFrame>::SinkType> &&sink, std::shared_ptr<CudaEglImagePool> pool, std::string pool_id, int pool_size)
		: NodeSISO<av::VideoFrame, EglImageFrame>(std::move(source), std::move(sink)), pool_(std::move(pool)), pool_id_(std::move(pool_id)), pool_size_(pool_size) {}
	~CudaToEglImage()
	{
		destroy_egl_context();
	}
	static std::shared_ptr<CudaToEglImage> create(NodeCreationInfo &nci)
	{
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
		std::shared_ptr<Edge<EglImageFrame>> dst = edges.find<EglImageFrame>(params["dst"]);
		std::string pool_id = params.count("pool_id") ? (std::string)params["pool_id"] : "default";
		int pool_size = params.count("pool_size") ? (int)params["pool_size"] : 8;
		auto pool = InstanceSharedObjects<CudaEglImagePool>::get(nci.instance, pool_id);
		return std::make_shared<CudaToEglImage>(make_unique<EdgeSource<av::VideoFrame>>(src), make_unique<EdgeSink<EglImageFrame>>(dst), pool, pool_id, pool_size);
	}
};

DECLNODE(cuda_to_egl_image, CudaToEglImage);
