#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../cuda.hpp"
// CUDA driver API (dynlink)
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"

#include <dlfcn.h>
#ifdef __GLIBC__
#include <link.h>   // dlmopen, lmid_t, LM_ID_NEWLM
#endif
#include <string>
#include <vector>
#include <optional>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

// NvOFFRUC interface (from NVIDIA Optical Flow SDK)
#include <NvOFFRUC.h>

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
#define CHECK_CU_FRUC(x) check_cu((x), #x)

class NvOFFruc : public NodeSISO<av::VideoFrame, av::VideoFrame> {
public:
	using NodeSISO::NodeSISO;

private:
	std::shared_ptr<HWAccelDevice> hwaccel_;
	AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

	// Output frame allocator (CUDA frames with NV12 sw_format)
	AVBufferRef* hw_frames_ctx_ = nullptr;
	int width_ = 0;
	int height_ = 0;

	// FRUC dynamic library + function pointers
	void* fruc_lib_ = nullptr;
#ifdef __GLIBC__
	// Load FRUC (and libcuda) into a separate link-map so it won't bind to avplumber's
	// CUDA dynlink shim symbols (e.g. cuCtxGetCurrent as static storage).
	void* fruc_ns_cuda_ = nullptr;
	lmid_t fruc_lmid_ = LM_ID_BASE;
	bool fruc_in_new_namespace_ = false;
#endif
	PtrToFuncNvOFFRUCCreate fn_create_ = nullptr;
	PtrToFuncNvOFFRUCRegisterResource fn_register_ = nullptr;
	PtrToFuncNvOFFRUCUnregisterResource fn_unregister_ = nullptr;
	PtrToFuncNvOFFRUCProcess fn_process_ = nullptr;
	PtrToFuncNvOFFRUCDestroy fn_destroy_ = nullptr;

	NvOFFRUCHandle h_fruc_ = nullptr;
	bool resourcehw_frames_ctx_s_registered_ = false;

	// CUDA buffers shared with FRUC (we copy into these)
	CUdeviceptr render_buf_[2]{0, 0};
	CUdeviceptr interp_buf_{0};
	size_t nv12_size_bytes_ = 0;
	int render_idx_ = 0;

	// State for 2x output scheduling
	bool have_prev_pts_ = false;
	av::Timestamp prev_pts_ = NOTS;

	// Multi-output per input frame: we need a small state machine to handle backpressure.
	enum class Stage { NeedInput, EmitInterp, EmitInput };
	Stage stage_ = Stage::NeedInput;
	av::VideoFrame pending_in_;
	av::VideoFrame pending_interp_;

	std::string fruc_library_path_;
	bool passthrough_on_fail_ = true;

	static av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm)
	{
		if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
		AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
		if (ctx == nullptr) return AV_PIX_FMT_NONE;
		return ctx->sw_format;
	}

	bool load_fruc_library()
	{
		if (fruc_lib_) return true;
		const char* libname = fruc_library_path_.empty() ? "libNvOFFRUC.so" : fruc_library_path_.c_str();
#ifdef __GLIBC__
		// Critical: ensure libNvOFFRUC binds CUDA symbols from libcuda.so.1,
		// not avplumber's CUDA dynlink shim. Use a new link-map namespace.
		fruc_ns_cuda_ = dlmopen(LM_ID_NEWLM, "libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
		if (fruc_ns_cuda_) {
			if (dlinfo(fruc_ns_cuda_, RTLD_DI_LMID, &fruc_lmid_) == 0) {
				fruc_lib_ = dlmopen(fruc_lmid_, libname, RTLD_NOW | RTLD_LOCAL);
				if (fruc_lib_) {
					fruc_in_new_namespace_ = true;
				}
			}
		}
#endif
		if (!fruc_lib_) {
			// Fallback: normal dlopen (may crash if CUDA symbols interpose).
			fruc_lib_ = dlopen(libname, RTLD_LAZY);
		}
		if (!fruc_lib_) {
			logstream << "nvof_fruc: dlopen failed for " << libname << ": " << (dlerror() ? dlerror() : "unknown");
			return false;
		}

		fn_create_ = (PtrToFuncNvOFFRUCCreate)dlsym(fruc_lib_, CreateProcName);
		fn_register_ = (PtrToFuncNvOFFRUCRegisterResource)dlsym(fruc_lib_, RegisterResourceProcName);
		fn_unregister_ = (PtrToFuncNvOFFRUCUnregisterResource)dlsym(fruc_lib_, UnregisterResourceProcName);
		fn_process_ = (PtrToFuncNvOFFRUCProcess)dlsym(fruc_lib_, ProcessProcName);
		fn_destroy_ = (PtrToFuncNvOFFRUCDestroy)dlsym(fruc_lib_, DestroyProcName);

		if (!fn_create_ || !fn_register_ || !fn_unregister_ || !fn_process_ || !fn_destroy_) {
			logstream << "nvof_fruc: dlsym missing required exports";
			return false;
		}
		return true;
	}

	void cleanup_fruc()
	{
		if (resources_registered_ && fn_unregister_ && h_fruc_) {
			NvOFFRUC_UNREGISTER_RESOURCE_PARAM unreg{};
			unreg.uiCount = NvOFFRUC_MIN_RESOURCE;
			unreg.pArrResource[0] = &interp_buf_;
			unreg.pArrResource[1] = &render_buf_[0];
			unreg.pArrResource[2] = &render_buf_[1];
			(void)fn_unregister_(h_fruc_, &unreg);
			resources_registered_ = false;
		}
		if (h_fruc_ && fn_destroy_) {
			(void)fn_destroy_(h_fruc_);
			h_fruc_ = nullptr;
		}
		if (interp_buf_) {
			CHECK_CU_FRUC(cuMemFree(interp_buf_));
			interp_buf_ = 0;
		}
		for (auto &b : render_buf_) {
			if (b) {
				CHECK_CU_FRUC(cuMemFree(b));
				b = 0;
			}
		}
		nv12_size_bytes_ = 0;
	}

	bool ensure_hw_frames_ctx(int w, int h)
	{
		if (!hwaccel_) return false;
		if (w <= 0 || h <= 0) return false;
		if (hw_frames_ctx_ && w == width_ && h == height_) return true;

		if (hw_frames_ctx_) {
			av_buffer_unref(&hw_frames_ctx_);
			hw_frames_ctx_ = nullptr;
		}

		hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
		if (!hw_frames_ctx_) {
			logstream << "nvof_fruc: av_hwframe_ctx_alloc failed";
			return false;
		}
		AVHWFramesContext *frmctx = (AVHWFramesContext *)(hw_frames_ctx_->data);
		frmctx->format = AV_PIX_FMT_CUDA;
		frmctx->sw_format = AV_PIX_FMT_NV12;
		frmctx->width = w;
		frmctx->height = h;
		int r = av_hwframe_ctx_init(hw_frames_ctx_);
		if (r != 0) {
			logstream << "nvof_fruc: av_hwframe_ctx_init failed: " << av::error2string(r);
			av_buffer_unref(&hw_frames_ctx_);
			hw_frames_ctx_ = nullptr;
			return false;
		}
		width_ = w;
		height_ = h;
		return true;
	}

	bool ensure_fruc(int w, int h)
	{
		if (w <= 0 || h <= 0) return false;
		if (!cuda_dev_ctx_) return false;
		if (!load_fruc_library()) return false;

		if (h_fruc_ && w == width_ && h == height_ && nv12_size_bytes_ != 0) {
			return true;
		}

		// Reinitialize on size change
		cleanup_fruc();

		// FRUC CUDA path requires a current CUDA context on this thread.
		if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
			logstream << "nvof_fruc: cuCtxPushCurrent failed (ensure_fruc, pre-create)";
			return false;
		}

		// Allocate our CUDA buffers (contiguous NV12: Y then UV)
		nv12_size_bytes_ = (size_t)w * (size_t)h + ((size_t)w * (size_t)h) / 2;
		int cuerr = 0;
		cuerr |= CHECK_CU_FRUC(cuMemAlloc(&interp_buf_, nv12_size_bytes_));
		cuerr |= CHECK_CU_FRUC(cuMemAlloc(&render_buf_[0], nv12_size_bytes_));
		cuerr |= CHECK_CU_FRUC(cuMemAlloc(&render_buf_[1], nv12_size_bytes_));
		if (cuerr) {
			logstream << "nvof_fruc: cuMemAlloc failed";
			CUcontext dummy;
			CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
			return false;
		}

		// Create FRUC instance
		NvOFFRUC_CREATE_PARAM createParams{};
		createParams.uiWidth = (uint32_t)w;
		createParams.uiHeight = (uint32_t)h;
		createParams.pDevice = nullptr; // for CUDA path, the sample uses internal CUDA ctx; driver-side impl ignores this
		createParams.eResourceType = CudaResource;
		createParams.eSurfaceFormat = NV12Surface;
		createParams.eCUDAResourceType = CudaResourceCuDevicePtr;

		NvOFFRUC_STATUS st = fn_create_(&createParams, &h_fruc_);
		if (st != NvOFFRUC_SUCCESS || !h_fruc_) {
			logstream << "nvof_fruc: NvOFFRUCCreate failed: " << (int)st;
			cleanup_fruc();
			CUcontext dummy;
			CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
			return false;
		}

		// Register resources (1 interpolate + 2 render) like the sample
		NvOFFRUC_REGISTER_RESOURCE_PARAM reg{};
		reg.uiCount = NvOFFRUC_MIN_RESOURCE;
		reg.pArrResource[0] = &interp_buf_;
		reg.pArrResource[1] = &render_buf_[0];
		reg.pArrResource[2] = &render_buf_[1];
		st = fn_register_(h_fruc_, &reg);
		if (st != NvOFFRUC_SUCCESS) {
			logstream << "nvof_fruc: NvOFFRUCRegisterResource failed: " << (int)st;
			cleanup_fruc();
			CUcontext dummy;
			CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
			return false;
		}
		resources_registered_ = true;
		render_idx_ = 0;
		have_prev_pts_ = false;
		prev_pts_ = NOTS;

		CUcontext dummy;
		CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
		return true;
	}

	bool copy_frame_to_nv12_buffer(const av::VideoFrame &in, CUdeviceptr dst_nv12)
	{
		// Copy input CUDA NV12 planes into our contiguous NV12 buffer.
		const int w = in.width();
		const int h = in.height();
		if (w <= 0 || h <= 0) return false;

		CUdeviceptr srcY = (CUdeviceptr)(uintptr_t)in.raw()->data[0];
		CUdeviceptr srcUV = (CUdeviceptr)(uintptr_t)in.raw()->data[1];
		if (!srcY || !srcUV) return false;

		size_t srcPitchY = (size_t)in.raw()->linesize[0];
		size_t srcPitchUV = (size_t)in.raw()->linesize[1];

		CUDA_MEMCPY2D cpyY{};
		cpyY.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyY.srcDevice = srcY;
		cpyY.srcPitch = srcPitchY;
		cpyY.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyY.dstDevice = dst_nv12;
		cpyY.dstPitch = (size_t)w;
		cpyY.WidthInBytes = (size_t)w;
		cpyY.Height = (size_t)h;

		CUDA_MEMCPY2D cpyUV{};
		cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.srcDevice = srcUV;
		cpyUV.srcPitch = srcPitchUV;
		cpyUV.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.dstDevice = dst_nv12 + (size_t)w * (size_t)h;
		cpyUV.dstPitch = (size_t)w;
		cpyUV.WidthInBytes = (size_t)w;
		cpyUV.Height = (size_t)h / 2;

		int cuerr = 0;
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyY, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyUV, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream));
		return cuerr == 0;
	}

	bool copy_nv12_buffer_to_frame(CUdeviceptr src_nv12, av::VideoFrame &out)
	{
		const int w = out.width();
		const int h = out.height();
		if (w <= 0 || h <= 0) return false;

		CUdeviceptr dstY = (CUdeviceptr)(uintptr_t)out.raw()->data[0];
		CUdeviceptr dstUV = (CUdeviceptr)(uintptr_t)out.raw()->data[1];
		if (!dstY || !dstUV) return false;

		size_t dstPitchY = (size_t)out.raw()->linesize[0];
		size_t dstPitchUV = (size_t)out.raw()->linesize[1];

		CUDA_MEMCPY2D cpyY{};
		cpyY.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyY.srcDevice = src_nv12;
		cpyY.srcPitch = (size_t)w;
		cpyY.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyY.dstDevice = dstY;
		cpyY.dstPitch = dstPitchY;
		cpyY.WidthInBytes = (size_t)w;
		cpyY.Height = (size_t)h;

		CUDA_MEMCPY2D cpyUV{};
		cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.srcDevice = src_nv12 + (size_t)w * (size_t)h;
		cpyUV.srcPitch = (size_t)w;
		cpyUV.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.dstDevice = dstUV;
		cpyUV.dstPitch = dstPitchUV;
		cpyUV.WidthInBytes = (size_t)w;
		cpyUV.Height = (size_t)h / 2;

		int cuerr = 0;
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyY, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyUV, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream));
		return cuerr == 0;
	}

	av::Timestamp midpoint_pts(const av::Timestamp &a, const av::Timestamp &b)
	{
		// Return midpoint between a and b in b's timebase
		const av::Rational tb = b.timebase();
		const int64_t ai = a.timestamp(tb);
		const int64_t bi = b.timestamp(tb);
		const int64_t mi = ai + (bi - ai) / 2;
		return av::Timestamp(mi, tb);
	}

	bool run_fruc_for_frame(const av::VideoFrame &in, const av::Timestamp &in_pts, const av::Timestamp &out_pts)
	{
		// Copies input into next render buffer and invokes FRUC to generate into interp_buf_.
		render_idx_ = (render_idx_ + 1) & 1;
		CUdeviceptr cur_render = render_buf_[render_idx_];

		if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
			logstream << "nvof_fruc: cuCtxPushCurrent failed (run_fruc_for_frame)";
			return false;
		}

		bool ok = true;
		if (!copy_frame_to_nv12_buffer(in, cur_render)) {
			logstream << "nvof_fruc: failed to copy input frame to FRUC buffer";
			ok = false;
		}

		if (ok) {
			NvOFFRUC_PROCESS_IN_PARAMS inParams{};
			NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
			bool repeated = false;

			inParams.stFrameDataInput.pFrame = &cur_render; // CUdeviceptr*
			inParams.stFrameDataInput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			inParams.stFrameDataInput.nCuSurfacePitch = 0;
			inParams.bSkipWarp = 0;

			outParams.stFrameDataOutput.pFrame = &interp_buf_; // CUdeviceptr*
			outParams.stFrameDataOutput.nTimeStamp = (double)out_pts.timestamp({1, 1000});
			outParams.stFrameDataOutput.nCuSurfacePitch = 0;
			outParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;

			NvOFFRUC_STATUS st = fn_process_(h_fruc_, &inParams, &outParams);
			if (st != NvOFFRUC_SUCCESS) {
				logstream << "nvof_fruc: NvOFFRUCProcess failed: " << (int)st;
				ok = false;
			}
		}

		CUcontext dummy;
		CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
		return ok;
	}

	bool make_interp_frame_from_buffer(const av::VideoFrame &ref_in, const av::Timestamp &out_pts, av::VideoFrame &out)
	{
		out = av::VideoFrame();
		out.setTimeBase(ref_in.timeBase());
		out.setPts(out_pts);
		out.raw()->color_range = ref_in.raw()->color_range;
		out.raw()->colorspace = ref_in.raw()->colorspace;

		out.raw()->format = AV_PIX_FMT_CUDA;
		out.raw()->width = width_;
		out.raw()->height = height_;

		if (!hw_frames_ctx_) return false;
		if (av_hwframe_get_buffer(hw_frames_ctx_, out.raw(), 0) != 0) {
			logstream << "nvof_fruc: av_hwframe_get_buffer failed";
			return false;
		}
		out.raw()->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);

		if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
			logstream << "nvof_fruc: cuCtxPushCurrent failed (make_interp_frame_from_buffer)";
			return false;
		}
		bool ok = copy_nv12_buffer_to_frame(interp_buf_, out);
		CUcontext dummy;
		CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
		if (!ok) {
			logstream << "nvof_fruc: failed to copy interpolated buffer into output frame";
			return false;
		}
		out.setComplete(true);
		return true;
	}

public:
	void process() override
	{
		// Emit pending frames first (backpressure-safe)
		if (stage_ == Stage::EmitInterp) {
			if (this->sink_->put(pending_interp_, true)) {
				stage_ = Stage::EmitInput;
			} else {
				return;
			}
		}
		if (stage_ == Stage::EmitInput) {
			if (this->sink_->put(pending_in_, true)) {
				// Done with this input
				pending_in_ = av::VideoFrame();
				pending_interp_ = av::VideoFrame();
				stage_ = Stage::NeedInput;
				this->source_->pop();
			} else {
				return;
			}
		}

		av::VideoFrame *pin = this->source_->peek();
		if (!pin) return;
		av::VideoFrame &in = *pin;
		if (!in) {
			this->source_->pop();
			return;
		}

		if (isEofMarker(in)) {
			// Flush: pass EOF through and reset internal state.
			if (!this->sink_->put(in, true)) return;
			this->source_->pop();
			have_prev_pts_ = false;
			prev_pts_ = NOTS;
			return;
		}

		// Only operate on CUDA NV12; otherwise pass through.
		if (in.raw()->format != AV_PIX_FMT_CUDA) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
			return;
		}
		av::PixelFormat swfmt = getHwSwPixelFormat(in);
		if (swfmt != AV_PIX_FMT_NV12) {
			logstream << "nvof_fruc: unsupported CUDA sw_format " << swfmt << " (need NV12), passthrough";
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
			return;
		}

		const int w = in.width();
		const int h = in.height();
		if (!ensure_hw_frames_ctx(w, h)) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
			return;
		}
		if (!ensure_fruc(w, h)) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
			return;
		}

		av::Timestamp in_pts = in.pts();
		if (!in_pts.isValid()) {
			// Without timestamps we can't schedule interpolation; passthrough.
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
			return;
		}

		// First frame: prime the FRUC state and pass through.
		if (!have_prev_pts_) {
			if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
				logstream << "nvof_fruc: cuCtxPushCurrent failed (prime)";
				if (passthrough_on_fail_) {
					if (!this->sink_->put(in, true)) return;
					this->source_->pop();
				}
				return;
			}
			// Copy to current render buffer and call FRUC with skip-warp to update internal state.
			CUdeviceptr cur_render = render_buf_[render_idx_];
			(void)copy_frame_to_nv12_buffer(in, cur_render);
			NvOFFRUC_PROCESS_IN_PARAMS inParams{};
			NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
			inParams.stFrameDataInput.pFrame = &cur_render;
			inParams.stFrameDataInput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			inParams.stFrameDataInput.nCuSurfacePitch = 0;
			inParams.bSkipWarp = 1;
			outParams.stFrameDataOutput.pFrame = &interp_buf_;
			outParams.stFrameDataOutput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			outParams.stFrameDataOutput.nCuSurfacePitch = 0;
			(void)fn_process_(h_fruc_, &inParams, &outParams);
			CUcontext dummy;
			CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));

			// Emit the original frame only.
			if (!this->sink_->put(in, true)) return;
			this->source_->pop();
			have_prev_pts_ = true;
			prev_pts_ = in_pts;
			return;
		}

		// For each subsequent frame: generate one interpolated frame at midpoint(prev, cur),
		// then output interpolated, then output current.
		av::Timestamp out_pts = midpoint_pts(prev_pts_, in_pts);
		if (!run_fruc_for_frame(in, in_pts, out_pts)) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
				prev_pts_ = in_pts;
				return;
			}
			return;
		}

		av::VideoFrame interp;
		if (!make_interp_frame_from_buffer(in, out_pts, interp)) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
				prev_pts_ = in_pts;
				return;
			}
			return;
		}

		// Set up pending output sequence. Don't pop source until both are emitted.
		pending_interp_ = interp;
		pending_in_ = in;
		stage_ = Stage::EmitInterp;

		// Update prev PTS immediately (so if we get re-entered we still use correct state).
		prev_pts_ = in_pts;
	}

	static std::shared_ptr<NvOFFruc> create(NodeCreationInfo &nci)
	{
		if (global_cuda.has_errors) {
			throw Error("nvof_fruc: CUDA not initialized");
		}
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		auto src = edges.find<av::VideoFrame>(params["src"]);
		auto dst = edges.find<av::VideoFrame>(params["dst"]);
		auto r = std::make_shared<NvOFFruc>(make_unique<EdgeSource<av::VideoFrame>>(src), make_unique<EdgeSink<av::VideoFrame>>(dst));
		if (!params.count("hwaccel")) {
			throw Error("nvof_fruc requires hwaccel parameter (CUDA device)");
		}
		r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
		if (!r->hwaccel_) {
			throw Error("nvof_fruc: failed to get hwaccel");
		}
		AVHWDeviceContext* devctx = (AVHWDeviceContext *)(r->hwaccel_->deviceContext()->data);
		r->cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
		if (!r->cuda_dev_ctx_) {
			throw Error("nvof_fruc: CUDA device context missing");
		}
		if (params.count("library")) {
			r->fruc_library_path_ = params["library"].get<std::string>();
		}
		if (params.count("passthrough_on_fail")) {
			r->passthrough_on_fail_ = (bool)params["passthrough_on_fail"];
		}
		return r;
	}

	~NvOFFruc() override
	{
		cleanup_fruc();
		if (hw_frames_ctx_) {
			av_buffer_unref(&hw_frames_ctx_);
			hw_frames_ctx_ = nullptr;
		}
		if (fruc_lib_) {
			dlclose(fruc_lib_);
			fruc_lib_ = nullptr;
		}
#ifdef __GLIBC__
		if (fruc_ns_cuda_) {
			dlclose(fruc_ns_cuda_);
			fruc_ns_cuda_ = nullptr;
		}
#endif
	}
};

DECLNODE(nvof_fruc, NvOFFruc);

