#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include "../../../cuda.hpp"
// CUDA driver API (dynlink)
#include "../../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"

#include <dlfcn.h>
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

class NvOFFruc : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
public:
	using NodeSISO::NodeSISO;
	bool consumeEofIfPresent() override {
		return false;
	}

private:
	std::shared_ptr<HWAccelDevice> hwaccel_;
	AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;

	// Output frame allocator (CUDA frames with NV12 sw_format)
	AVBufferRef* hw_frames_ctx_ = nullptr;
	int width_ = 0;
	int height_ = 0;

	// FRUC dynamic library + function pointers
	void* fruc_lib_ = nullptr;
	PtrToFuncNvOFFRUCCreate fn_create_ = nullptr;
	PtrToFuncNvOFFRUCRegisterResource fn_register_ = nullptr;
	PtrToFuncNvOFFRUCUnregisterResource fn_unregister_ = nullptr;
	PtrToFuncNvOFFRUCProcess fn_process_ = nullptr;
	PtrToFuncNvOFFRUCDestroy fn_destroy_ = nullptr;

	NvOFFRUCHandle h_fruc_ = nullptr;
	bool resources_registered_ = false;

	// CUDA arrays shared with FRUC. Two rolling render buffers hold the pair of
	// input frames FRUC's optical-flow state operates on. One interp buffer per
	// interpolated output slot lets us pipeline: FRUC writes to interp_bufs_[i]
	// while the previous output copy from interp_bufs_[i-1] is still in flight.
	// Sharing a single interp buffer would force a sync between every output
	// copy and the next Process() call.
	CUarray render_buf_[2]{nullptr, nullptr};
	std::vector<CUarray> interp_bufs_;
	int render_idx_ = 0;

	// State for 2x output scheduling
	bool have_prev_pts_ = false;
	av::Timestamp prev_pts_ = NOTS;

	// Multi-output per input frame: we need a small state machine to handle backpressure.
	// pending_out_ holds (factor_-1) interpolated frames followed by the current input frame.
	// emit_idx_ is the next index to emit; source is popped after the last one is put.
	std::vector<av::VideoFrame> pending_out_;
	size_t emit_idx_ = 0;

	std::string fruc_library_path_;
	bool passthrough_on_fail_ = true;
	int factor_ = 2;

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
		// Ensure the real CUDA driver library is loaded, then load FRUC with RTLD_DEEPBIND so its
		// CUDA driver API imports (e.g. cuCtxGetCurrent) resolve to libcuda.so.1, not avplumber's
		// CUDA dynlink shim symbols.
		(void)dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
		fruc_lib_ = dlopen(libname, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
		if (!fruc_lib_) {
			// Last resort fallback
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
			uint32_t idx = 0;
			for (auto &b : interp_bufs_) {
				if (idx < NvOFFRUC_MAX_RESOURCE) unreg.pArrResource[idx++] = b;
			}
			if (idx < NvOFFRUC_MAX_RESOURCE) unreg.pArrResource[idx++] = render_buf_[0];
			if (idx < NvOFFRUC_MAX_RESOURCE) unreg.pArrResource[idx++] = render_buf_[1];
			unreg.uiCount = idx;
			(void)fn_unregister_(h_fruc_, &unreg);
			resources_registered_ = false;
		}
		if (h_fruc_ && fn_destroy_) {
			(void)fn_destroy_(h_fruc_);
			h_fruc_ = nullptr;
		}
		for (auto &b : interp_bufs_) {
			if (b) {
				CHECK_CU_FRUC(cuArrayDestroy(b));
				b = nullptr;
			}
		}
		interp_bufs_.clear();
		for (auto &b : render_buf_) {
			if (b) {
				CHECK_CU_FRUC(cuArrayDestroy(b));
				b = nullptr;
			}
		}
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

		const size_t needed_interp = (size_t)(factor_ - 1);
		if (h_fruc_ && w == width_ && h == height_ && interp_bufs_.size() == needed_interp
			&& render_buf_[0] && render_buf_[1]) {
			return true;
		}

		// Reinitialize on size change
		cleanup_fruc();

		// FRUC CUDA path requires a current CUDA context on this thread.
		if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
			logstream << "nvof_fruc: cuCtxPushCurrent failed (ensure_fruc, pre-create)";
			return false;
		}

		// Allocate CUDA arrays for NV12 (single channel array with height * 3/2)
		CUDA_ARRAY_DESCRIPTOR desc{};
		desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
		desc.Width = (size_t)w;
		desc.Height = (size_t)h + (size_t)h / 2;
		desc.NumChannels = 1;
		int cuerr = 0;
		interp_bufs_.assign(needed_interp, nullptr);
		for (auto &b : interp_bufs_) {
			cuerr |= CHECK_CU_FRUC(cuArrayCreate(&b, &desc));
		}
		cuerr |= CHECK_CU_FRUC(cuArrayCreate(&render_buf_[0], &desc));
		cuerr |= CHECK_CU_FRUC(cuArrayCreate(&render_buf_[1], &desc));
		if (cuerr) {
			logstream << "nvof_fruc: cuArrayCreate failed";
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
		createParams.eCUDAResourceType = CudaResourceCuArray;

		NvOFFRUC_STATUS st = fn_create_(&createParams, &h_fruc_);
		if (st != NvOFFRUC_SUCCESS || !h_fruc_) {
			logstream << "nvof_fruc: NvOFFRUCCreate failed: " << (int)st;
			cleanup_fruc();
			CUcontext dummy;
			CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));
			return false;
		}

		// Register resources: (factor_-1) interp buffers + 2 render buffers.
		NvOFFRUC_REGISTER_RESOURCE_PARAM reg{};
		uint32_t idx = 0;
		for (auto &b : interp_bufs_) {
			if (idx >= NvOFFRUC_MAX_RESOURCE) break;
			reg.pArrResource[idx++] = b;
		}
		if (idx < NvOFFRUC_MAX_RESOURCE) reg.pArrResource[idx++] = render_buf_[0];
		if (idx < NvOFFRUC_MAX_RESOURCE) reg.pArrResource[idx++] = render_buf_[1];
		reg.uiCount = idx;
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

	// Issue an async device->array copy of an NV12 CUDA frame. Callers must
	// synchronize the stream before FRUC reads from dst_nv12.
	bool copy_frame_to_nv12_buffer_async(const av::VideoFrame &in, CUarray dst_nv12)
	{
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
		cpyY.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		cpyY.dstArray = dst_nv12;
		cpyY.dstY = 0;
		cpyY.WidthInBytes = (size_t)w;
		cpyY.Height = (size_t)h;

		CUDA_MEMCPY2D cpyUV{};
		cpyUV.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.srcDevice = srcUV;
		cpyUV.srcPitch = srcPitchUV;
		cpyUV.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		cpyUV.dstArray = dst_nv12;
		cpyUV.dstY = (size_t)h;
		cpyUV.WidthInBytes = (size_t)w;
		cpyUV.Height = (size_t)h / 2;

		int cuerr = 0;
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyY, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyUV, cuda_dev_ctx_->stream));
		return cuerr == 0;
	}

	bool copy_frame_to_nv12_buffer(const av::VideoFrame &in, CUarray dst_nv12)
	{
		if (!copy_frame_to_nv12_buffer_async(in, dst_nv12)) return false;
		return CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
	}

	// Issue an async array->device copy of an NV12 CUDA frame. Callers must
	// synchronize the stream before downstream reads out.
	bool copy_nv12_buffer_to_frame_async(CUarray src_nv12, av::VideoFrame &out)
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
		cpyY.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		cpyY.srcArray = src_nv12;
		cpyY.srcY = 0;
		cpyY.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyY.dstDevice = dstY;
		cpyY.dstPitch = dstPitchY;
		cpyY.WidthInBytes = (size_t)w;
		cpyY.Height = (size_t)h;

		CUDA_MEMCPY2D cpyUV{};
		cpyUV.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		cpyUV.srcArray = src_nv12;
		cpyUV.srcY = (size_t)h;
		cpyUV.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		cpyUV.dstDevice = dstUV;
		cpyUV.dstPitch = dstPitchUV;
		cpyUV.WidthInBytes = (size_t)w;
		cpyUV.Height = (size_t)h / 2;

		int cuerr = 0;
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyY, cuda_dev_ctx_->stream));
		cuerr |= CHECK_CU_FRUC(cuMemcpy2DAsync(&cpyUV, cuda_dev_ctx_->stream));
		return cuerr == 0;
	}

	bool copy_nv12_buffer_to_frame(CUarray src_nv12, av::VideoFrame &out)
	{
		if (!copy_nv12_buffer_to_frame_async(src_nv12, out)) return false;
		return CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream)) == 0;
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

	// Linear interpolation of PTS at fraction f in [0,1] between a and b (in b's timebase).
	av::Timestamp lerp_pts(const av::Timestamp &a, const av::Timestamp &b, double f)
	{
		const av::Rational tb = b.timebase();
		const int64_t ai = a.timestamp(tb);
		const int64_t bi = b.timestamp(tb);
		const int64_t mi = ai + (int64_t)((double)(bi - ai) * f);
		return av::Timestamp(mi, tb);
	}

	// Allocate an output CUDA VideoFrame and issue an async copy from src_nv12.
	// The caller is responsible for a single cuStreamSynchronize at the end of
	// the burst before publishing the frame downstream.
	bool make_interp_frame_from_buffer_async(const av::VideoFrame &ref_in, const av::Timestamp &out_pts,
		CUarray src_nv12, av::VideoFrame &out)
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

		bool ok = copy_nv12_buffer_to_frame_async(src_nv12, out);
		if (!ok) {
			logstream << "nvof_fruc: failed to schedule copy of interpolated buffer into output frame";
			return false;
		}
		out.setComplete(true);
		return true;
	}

public:
	void process() override
	{
		// Emit any pending frames first (backpressure-safe).
		while (emit_idx_ < pending_out_.size()) {
			if (!this->sink_->put(pending_out_[emit_idx_], true)) {
				return; // downstream full; try again later
			}
			emit_idx_++;
		}
		if (!pending_out_.empty()) {
			// All buffered outputs delivered — commit by popping the source input we produced them from.
			pending_out_.clear();
			emit_idx_ = 0;
			this->source_->pop();
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
			this->finished_ = true;
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
			CUarray cur_render = render_buf_[render_idx_];
			(void)copy_frame_to_nv12_buffer(in, cur_render);
			NvOFFRUC_PROCESS_IN_PARAMS inParams{};
			NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
			bool repeated = false;
			inParams.stFrameDataInput.pFrame = cur_render;
			inParams.stFrameDataInput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			inParams.stFrameDataInput.nCuSurfacePitch = 0;
			inParams.bSkipWarp = 1;
			outParams.stFrameDataOutput.pFrame = interp_bufs_[0];
			outParams.stFrameDataOutput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			outParams.stFrameDataOutput.nCuSurfacePitch = 0;
			outParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
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

		// For each subsequent input: generate (factor_ - 1) interpolated frames spaced evenly
		// between prev_pts_ and in_pts, then emit them followed by the current input.
		//
		// NvOFFRUC's Process() consumes the incoming frame to update its internal 2-frame
		// window each call. To sample multiple intermediate times we must copy the input
		// into a render buffer *once* and then call Process() multiple times with different
		// output timestamps but the same input render buffer. render_idx_ is advanced only
		// once per input frame.
		const int nsteps = factor_ - 1;
		pending_out_.clear();
		pending_out_.reserve((size_t)nsteps + 1);

		// Pipelining: one CUDA context push covers the whole burst. Input copy
		// is issued async but MUST be synced before the first Process() call
		// because FRUC uses its own internal stream. Each Process() writes to a
		// distinct interp buffer, so subsequent output copies can be issued
		// back-to-back without draining the stream. A single sync at the end
		// covers all output copies before we publish the frames.
		if (CHECK_CU_FRUC(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
				prev_pts_ = in_pts;
			}
			return;
		}

		render_idx_ = (render_idx_ + 1) & 1;
		CUarray cur_render = render_buf_[render_idx_];
		bool step_failed = false;

		if (!copy_frame_to_nv12_buffer_async(in, cur_render)
			|| CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
			step_failed = true;
		}

		for (int i = 1; !step_failed && i <= nsteps; ++i) {
			double f = (double)i / (double)factor_;
			av::Timestamp out_pts = lerp_pts(prev_pts_, in_pts, f);

			NvOFFRUC_PROCESS_IN_PARAMS inParams{};
			NvOFFRUC_PROCESS_OUT_PARAMS outParams{};
			bool repeated = false;
			inParams.stFrameDataInput.pFrame = cur_render;
			inParams.stFrameDataInput.nTimeStamp = (double)in_pts.timestamp({1, 1000});
			inParams.stFrameDataInput.nCuSurfacePitch = 0;
			inParams.bSkipWarp = 0;
			CUarray out_buf = interp_bufs_[(size_t)(i - 1)];
			outParams.stFrameDataOutput.pFrame = out_buf;
			outParams.stFrameDataOutput.nTimeStamp = (double)out_pts.timestamp({1, 1000});
			outParams.stFrameDataOutput.nCuSurfacePitch = 0;
			outParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &repeated;
			NvOFFRUC_STATUS st = fn_process_(h_fruc_, &inParams, &outParams);
			if (st != NvOFFRUC_SUCCESS) {
				logstream << "nvof_fruc: NvOFFRUCProcess failed at step " << i << "/" << nsteps << ": " << (int)st;
				step_failed = true;
				break;
			}

			av::VideoFrame interp;
			if (!make_interp_frame_from_buffer_async(in, out_pts, out_buf, interp)) {
				step_failed = true;
				break;
			}
			pending_out_.push_back(std::move(interp));
		}

		// One sync covers every output copy issued in the burst.
		if (!step_failed && CHECK_CU_FRUC(cuStreamSynchronize(cuda_dev_ctx_->stream))) {
			step_failed = true;
		}

		CUcontext dummy;
		CHECK_CU_FRUC(cuCtxPopCurrent(&dummy));

		if (step_failed) {
			pending_out_.clear();
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
				prev_pts_ = in_pts;
			}
			return;
		}

		// Append current input as the last frame in the burst.
		pending_out_.push_back(in);
		emit_idx_ = 0;

		// Update prev PTS immediately so state stays coherent across re-entries.
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
		r->auto_eof_ = false;
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
		if (params.count("factor")) {
			int f = (int)params["factor"];
			// FRUC registers (factor-1) interp buffers + 2 render buffers and
			// caps total registered resources at NvOFFRUC_MAX_RESOURCE.
			const int max_factor = (int)NvOFFRUC_MAX_RESOURCE - 2 + 1;
			if (f < 2 || f > max_factor) {
				throw Error("nvof_fruc: factor must be in [2, " + std::to_string(max_factor) + "]");
			}
			r->factor_ = f;
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
	}
};

DECLNODE(nvof_fruc, NvOFFruc);
