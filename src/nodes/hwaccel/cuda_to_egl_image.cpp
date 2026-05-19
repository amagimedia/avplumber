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

#include <algorithm>
#include <vector>
#include <optional>
#include <string>

// CUDA driver API (dynlink)
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"
#include <libavutil/hwcontext_cuda.h>

// PTX blob for the conversion kernels (generated at build time)
#include "../../../objs/src/nodes/hwaccel/yuv_to_rgba_surface.ptx.h"

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

class CudaToEglImage: public NodeSISO<av::VideoFrame, EglImageFrame> {
protected:
	// External shared pool
	std::shared_ptr<CudaEglImagePool> pool_;
	std::string pool_id_;
	int pool_size_ = 60; // default; can be overridden by params
	int pool_max_size_ = 0; // 0 => disabled; if >0, pool may grow up to this size on demand
	int pool_grow_step_ = 8; // when growing, add this many entries (or less to reach max)
	bool enable_sync_ = false; // whether to call cuCtxSynchronize after kernel launch

	// CUDA
	CUcontext cu_ctx_ = nullptr; // adopted from incoming frame
	CUmodule cu_module_ = nullptr;
	CUfunction cu_kernel_444_ = nullptr;
	CUfunction cu_kernel_420_ = nullptr;
	CUfunction cu_kernel_nv12_ = nullptr;
	CUfunction cu_kernel_422_ = nullptr;
	CUfunction cu_kernel_nv16_ = nullptr;
	CUfunction cu_kernel_rgba_ = nullptr;
	CUfunction cu_kernel_bgra_ = nullptr;
	CUfunction cu_kernel_argb_ = nullptr;
	CUfunction cu_kernel_abgr_ = nullptr;
	CUfunction cu_kernel_rgb0_ = nullptr;
	CUfunction cu_kernel_bgr0_ = nullptr;
	CUfunction cu_kernel_0rgb_ = nullptr;
	CUfunction cu_kernel_0bgr_ = nullptr;
	CUfunction cu_kernel_444_full_ = nullptr;
	CUfunction cu_kernel_420_full_ = nullptr;
	CUfunction cu_kernel_nv12_full_ = nullptr;
	CUfunction cu_kernel_422_full_ = nullptr;
	CUfunction cu_kernel_nv16_full_ = nullptr;
	CUfunction cu_kernel_rgba_full_ = nullptr;
	CUfunction cu_kernel_bgra_full_ = nullptr;
	CUfunction cu_kernel_argb_full_ = nullptr;
	CUfunction cu_kernel_abgr_full_ = nullptr;
	CUfunction cu_kernel_rgb0_full_ = nullptr;
	CUfunction cu_kernel_bgr0_full_ = nullptr;
	CUfunction cu_kernel_0rgb_full_ = nullptr;
	CUfunction cu_kernel_0bgr_full_ = nullptr;

	// helpers
	av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm)
	{
		if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
		AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
		if (ctx == nullptr) return AV_PIX_FMT_NONE;
		return ctx->sw_format;
	}
	bool ensure_pool(int W, int H)
	{
		if (!pool_) {
			logstream << "cuda_to_egl_image: pool_ is null";
			return false;
		}
		if (!pool_->ensureInitialized(W, H, std::max(1, pool_size_), cu_ctx_)) {
			logstream << "cuda_to_egl_image: pool ensureInitialized failed";
			return false;
		}
		return true;
	}
	bool ensure_kernel_loaded()
	{
		if (cu_module_ && cu_kernel_444_ && cu_kernel_420_ && cu_kernel_nv12_) return true;
		if (!cu_ctx_) return false;
		// Rebind context every time: worker thread can change after node restart.
		if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
			logstream << "cuda_to_egl_image: cuCtxSetCurrent failed in ensure_kernel_loaded";
			return false;
		}
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
			const std::string ptx_str(avpl_yuv_rgba_ptx, avpl_yuv_rgba_ptx + avpl_yuv_rgba_ptx_len);
			if (CHECK_CU(cuModuleLoadDataEx(&cu_module_, (const void*)ptx_str.c_str(), 4, opts, optvals))) {
				logstream << "cuda_to_egl_image: cuModuleLoadDataEx failed";
				if (error_log[0]) logstream << "CUDA JIT error log: " << error_log;
				if (info_log[0])  logstream << "CUDA JIT info log: "  << info_log;
				return false;
			}
			if (info_log[0]) logstream << "CUDA JIT info log: " << info_log;
		}
		bool ok = true;
		if (!cu_kernel_444_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_444_, cu_module_, "kYUV444p_709lim_to_RGBA8_surface"));
		if (!cu_kernel_420_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_420_, cu_module_, "kYUV420p_709lim_to_RGBA8_surface"));
		if (!cu_kernel_nv12_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_nv12_, cu_module_, "kNV12_709lim_to_RGBA8_surface"));
		if (!cu_kernel_422_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_422_, cu_module_, "kYUV422p_709lim_to_RGBA8_surface"));
		if (!cu_kernel_nv16_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_nv16_, cu_module_, "kNV16_709lim_to_RGBA8_surface"));
		// Passthrough (use same symbol for both limited/full variants)
		if (!cu_kernel_rgba_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_rgba_, cu_module_, "kRGBA_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_bgra_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_bgra_, cu_module_, "kBGRA_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_argb_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_argb_, cu_module_, "kARGB_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_abgr_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_abgr_, cu_module_, "kABGR_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_rgb0_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_rgb0_, cu_module_, "kRGB0_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_bgr0_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_bgr0_, cu_module_, "kBGR0_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_0rgb_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_0rgb_, cu_module_, "k0RGB_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_0bgr_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_0bgr_, cu_module_, "k0BGR_to_RGBA8_passthrough_surface"));
		if (!cu_kernel_444_full_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_444_full_, cu_module_, "kYUV444p_709full_to_RGBA8_surface"));
		if (!cu_kernel_420_full_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_420_full_, cu_module_, "kYUV420p_709full_to_RGBA8_surface"));
		if (!cu_kernel_nv12_full_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_nv12_full_, cu_module_, "kNV12_709full_to_RGBA8_surface"));
		if (!cu_kernel_422_full_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_422_full_, cu_module_, "kYUV422p_709full_to_RGBA8_surface"));
		if (!cu_kernel_nv16_full_) ok &= !CHECK_CU(cuModuleGetFunction(&cu_kernel_nv16_full_, cu_module_, "kNV16_709full_to_RGBA8_surface"));
		// For passthrough, reuse the same kernel symbol
		if (!cu_kernel_rgba_full_) cu_kernel_rgba_full_ = cu_kernel_rgba_;
		if (!cu_kernel_bgra_full_) cu_kernel_bgra_full_ = cu_kernel_bgra_;
		if (!cu_kernel_argb_full_) cu_kernel_argb_full_ = cu_kernel_argb_;
		if (!cu_kernel_abgr_full_) cu_kernel_abgr_full_ = cu_kernel_abgr_;
		if (!cu_kernel_rgb0_full_) cu_kernel_rgb0_full_ = cu_kernel_rgb0_;
		if (!cu_kernel_bgr0_full_) cu_kernel_bgr0_full_ = cu_kernel_bgr0_;
		if (!cu_kernel_0rgb_full_) cu_kernel_0rgb_full_ = cu_kernel_0rgb_;
		if (!cu_kernel_0bgr_full_) cu_kernel_0bgr_full_ = cu_kernel_0bgr_;
		return ok;
	}
	bool run_conversion_to_texture(const av::VideoFrame &frm, AVPixelFormat swfmt, CUgraphicsResource cu_tex_res, int tex_w, int tex_h)
	{
		if (!cu_ctx_ || !cu_tex_res) {
			logstream << "cuda_to_egl_image: run_conversion_to_texture failed - cu_ctx_=" << (void*)cu_ctx_ << " cu_tex_res=" << (void*)cu_tex_res;
			return false;
		}
		// Rebind context every time: worker thread can change after node restart.
		if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
			logstream << "cuda_to_egl_image: cuCtxSetCurrent failed in run_conversion_to_texture";
			return false;
		}
		if (CHECK_CU(cuGraphicsMapResources(1, &cu_tex_res, 0))) {
			logstream << "cuda_to_egl_image: cuGraphicsMapResources failed";
			return false;
		}
		CUarray cu_arr = nullptr;
		if (CHECK_CU(cuGraphicsSubResourceGetMappedArray(&cu_arr, cu_tex_res, 0, 0))) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			logstream << "cuda_to_egl_image: cuGraphicsSubResourceGetMappedArray failed";
			return false;
		}
		if (!cuSurfObjectCreate || !cuSurfObjectDestroy) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			logstream << "cuda_to_egl_image: surface object functions not available";
			return false;
		}
		CUDA_RESOURCE_DESC rdesc; memset(&rdesc, 0, sizeof(rdesc));
		rdesc.resType = CU_RESOURCE_TYPE_ARRAY;
		rdesc.res.array.hArray = cu_arr;
		CUsurfObject surf = 0;
		if (CHECK_CU(cuSurfObjectCreate(&surf, &rdesc))) {
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
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
		// Choose full vs limited based on frame color range
		const bool is_full_range = (frm.raw()->color_range == AVCOL_RANGE_JPEG);
		CUfunction kfun = is_full_range ? cu_kernel_444_full_ : cu_kernel_444_;
		if (swfmt == AV_PIX_FMT_YUV420P) kfun = is_full_range ? cu_kernel_420_full_ : cu_kernel_420_;
		else if (swfmt == AV_PIX_FMT_NV12) kfun = is_full_range ? cu_kernel_nv12_full_ : cu_kernel_nv12_;
		else if (swfmt == AV_PIX_FMT_YUV422P) kfun = is_full_range ? cu_kernel_422_full_ : cu_kernel_422_;
		else if (swfmt == AV_PIX_FMT_NV16) kfun = is_full_range ? cu_kernel_nv16_full_ : cu_kernel_nv16_;
		else if (swfmt == AV_PIX_FMT_RGBA) kfun = is_full_range ? cu_kernel_rgba_full_ : cu_kernel_rgba_;
		else if (swfmt == AV_PIX_FMT_BGRA) kfun = is_full_range ? cu_kernel_bgra_full_ : cu_kernel_bgra_;
		else if (swfmt == AV_PIX_FMT_ARGB) kfun = is_full_range ? cu_kernel_argb_full_ : cu_kernel_argb_;
		else if (swfmt == AV_PIX_FMT_ABGR) kfun = is_full_range ? cu_kernel_abgr_full_ : cu_kernel_abgr_;
		else if (swfmt == AV_PIX_FMT_RGB0) kfun = is_full_range ? cu_kernel_rgb0_full_ : cu_kernel_rgb0_;
		else if (swfmt == AV_PIX_FMT_BGR0) kfun = is_full_range ? cu_kernel_bgr0_full_ : cu_kernel_bgr0_;
		else if (swfmt == AV_PIX_FMT_0RGB) kfun = is_full_range ? cu_kernel_0rgb_full_ : cu_kernel_0rgb_;
		else if (swfmt == AV_PIX_FMT_0BGR) kfun = is_full_range ? cu_kernel_0bgr_full_ : cu_kernel_0bgr_;
		if (CHECK_CU(cuLaunchKernel(kfun,
		                            gridX, gridY, 1,
		                            blockX, blockY, 1,
		                            0, 0, args, nullptr))) {
			logstream << "cuda_to_egl_image: cuLaunchKernel failed";
			cuSurfObjectDestroy(surf);
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			return false;
		}
		// Ensure all writes to the surface are visible before releasing the resource
		if (enable_sync_ && CHECK_CU(cuCtxSynchronize())) {
			logstream << "cuda_to_egl_image: cuCtxSynchronize failed after cuLaunchKernel";
			cuSurfObjectDestroy(surf);
			CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
			return false;
		}
		cuSurfObjectDestroy(surf);
		CHECK_CU(cuGraphicsUnmapResources(1, &cu_tex_res, 0));
		return true;
	}
public:
	using NodeSISO<av::VideoFrame, EglImageFrame>::NodeSISO;
	virtual void process()
	{
		// Block until an input frame is available, but keep it in the queue
		// until we successfully publish the output (use peek/pop pattern).
		av::VideoFrame *pfrm = this->source_->peek();
		if (pfrm == nullptr) {
			return;
		}
		av::VideoFrame &frm = *pfrm;
		if (!frm) {
			logstream << "cuda_to_egl_image: frame is invalid/null, discarding";
			this->source_->pop(); // discard
			return;
		}
		// Adopt CUDA context from frame (once)
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
			// Make the adopted context current for this node's thread
			if (CHECK_CU(cuCtxSetCurrent(cu_ctx_))) {
				logstream << "cuda_to_egl_image: cuCtxSetCurrent failed during context adoption";
				this->source_->pop();
				return;
			}
		}
		int W = frm.width();
		int H = frm.height();
		AVPixelFormat swfmt = getHwSwPixelFormat(frm);
		if (!(swfmt == AV_PIX_FMT_YUV444P || swfmt == AV_PIX_FMT_YUV420P || swfmt == AV_PIX_FMT_NV12 || swfmt == AV_PIX_FMT_YUV422P || swfmt == AV_PIX_FMT_NV16
		      || swfmt == AV_PIX_FMT_RGBA || swfmt == AV_PIX_FMT_BGRA || swfmt == AV_PIX_FMT_ARGB || swfmt == AV_PIX_FMT_ABGR
		      || swfmt == AV_PIX_FMT_RGB0 || swfmt == AV_PIX_FMT_BGR0 || swfmt == AV_PIX_FMT_0RGB || swfmt == AV_PIX_FMT_0BGR)) {
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
		if (!opt_idx && pool_max_size_ > 0) {
			opt_idx = pool_->acquireOrGrow(W, H, pool_max_size_, pool_grow_step_, cu_ctx_);
		}
		if (!opt_idx) {
			// No available EGL image in the pool yet; keep the input frame and retry shortly
			wallclock.sleepms(1);
			return;
		}
		const size_t idx = *opt_idx;
		auto &entry = pool_->entry(idx);
		// Ensure kernels are ready before conversion
		if (!ensure_kernel_loaded()) {
			logstream << "cuda_to_egl_image: ensure_kernel_loaded failed";
			pool_->release(idx);
			this->source_->pop();
			return;
		}
		if (!run_conversion_to_texture(frm, swfmt, entry.cu_tex_res, W, H)) {
			// One recovery attempt: restart can leave CUDA/GL interop state stale for a frame.
			const int keep_size = std::max((int)pool_->size(), std::max(1, pool_size_));
			if (pool_->ensureInitialized(W, H, keep_size, cu_ctx_) &&
			    run_conversion_to_texture(frm, swfmt, entry.cu_tex_res, W, H)) {
				logstream << "cuda_to_egl_image: run_conversion_to_texture recovered after interop rebind";
			} else {
				logstream << "cuda_to_egl_image: run_conversion_to_texture failed, releasing pool entry";
				pool_->release(idx);
				this->source_->pop();
				return;
			}
		}
		// Create a token (owned by shared_ptr) to release the entry back to pool
		auto token_sp = std::shared_ptr<EglImagePoolToken>(new EglImagePoolToken{
			.release = [p = pool_, index = (int)idx]() {
				p->release(index);
			}
		});
		// Holder owns the token; token captures pool_, so pool stays alive
		std::shared_ptr<void> holder = token_sp;
		EglImageFrame out(entry.egl_image, W, H, frm.pts(), frm.timeBase(), holder, entry.gl_tex_rgba);
		out.copyMetadata(frm);
		// Blocking put: wait for space in sink, then remove input from source
		this->sink_->put(out, false);
		this->source_->pop();
	}
	CudaToEglImage(std::unique_ptr<typename NodeSISO<av::VideoFrame, EglImageFrame>::SourceType> &&source, std::unique_ptr<typename NodeSISO<av::VideoFrame, EglImageFrame>::SinkType> &&sink, std::shared_ptr<CudaEglImagePool> pool, std::string pool_id, int pool_size, bool enable_sync)
		: NodeSISO<av::VideoFrame, EglImageFrame>(std::move(source), std::move(sink)), pool_(std::move(pool)), pool_id_(std::move(pool_id)), pool_size_(pool_size), enable_sync_(enable_sync)
	{
	}
	~CudaToEglImage() override = default;
	static std::shared_ptr<CudaToEglImage> create(NodeCreationInfo &nci)
	{
		if (global_cuda.has_errors) {
			throw Error("cuda_to_egl_image: CUDA not initialized");
		}
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
		std::shared_ptr<Edge<EglImageFrame>> dst = edges.find<EglImageFrame>(params["dst"]);
		std::string pool_id = params.count("pool_id") ? (std::string)params["pool_id"] : "default";
		int pool_size = params.count("pool_size") ? (int)params["pool_size"] : 8;
		int pool_max_size = params.count("pool_max_size") ? (int)params["pool_max_size"] : pool_size;
		int pool_grow_step = params.count("pool_grow_step") ? (int)params["pool_grow_step"] : 8;
		bool enable_sync = params.count("sync") ? (bool)params["sync"] : false;
		auto pool = InstanceSharedObjects<CudaEglImagePool>::get(nci.instance, pool_id);
		auto node = std::make_shared<CudaToEglImage>(make_unique<EdgeSource<av::VideoFrame>>(src), make_unique<EdgeSink<EglImageFrame>>(dst), pool, pool_id, pool_size, enable_sync);
		node->pool_max_size_ = pool_max_size;
		node->pool_grow_step_ = pool_grow_step;
		return node;
	}
};

DECLNODE(cuda_to_egl_image, CudaToEglImage);
