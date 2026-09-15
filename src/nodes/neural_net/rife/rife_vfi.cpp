// rife_vfi -- learned video frame interpolation via RIFE 4.26 (TensorRT).
//
// Drop-in quality upgrade over nvof_fruc. Same graph API: rolling 2-input
// window (prev, cur), for each new cur emits (factor-1) interpolated frames
// followed by cur. Unlike NvOFFRUC's hardware optical flow, RIFE handles
// disocclusion (regions revealed by moving objects) and produces sharp edges
// through its learned refinement network -- see the "torn hand" artifact
// class that nvof_fruc could not fix.
//
// Inputs are CUDA/NV12 hardware frames. The node keeps two fp16 RGB tensors
// (planar, padded to a multiple of 32) as its rolling 2-frame window, runs
// the RIFE engine to produce the interpolated fp16 RGB tensor, then converts
// back to CUDA/NV12 into a fresh AVFrame buffer. The output pool matches the
// input frame ctx (BT.709 limited-range NV12), matching what NVENC expects.
//
// Engine: pre-built with build_trt.py, fixed at H=1088, W=1920. Non-1080p
// sources aren't handled yet (would need dynamic shape profiles + variable
// padding). Colorspace assumed BT.709 limited range for now.

#include "../../node_common.hpp"
#include "../../../hwaccel.hpp"
#include "../../../cuda.hpp"
#include "../../../../deps/cuda_loader/cuda_drvapi_dynlink_cuda.h"

#include <string>
#include <vector>
#include <memory>
#include <fstream>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <NvInfer.h>

// PTX byte array for our color-conversion kernels (generated via ptx_kernel
// rule in the Makefile).
#include "../../../../objs/src/nodes/neural_net/rife/rife_kernels.ptx.h"

static int rife_check_cu(CUresult err, const char *func)
{
	if (err == CUDA_SUCCESS) return 0;
	const char *err_name = nullptr, *err_string = nullptr;
	if (cuGetErrorName && cuGetErrorString) {
		cuGetErrorName(err, &err_name);
		cuGetErrorString(err, &err_string);
	}
	logstream << "cuda function: " << func << " failed: "
	          << (err_name ? err_name : "?") << ": " << (err_string ? err_string : "?");
	return -1;
}
#define CHECK_CU_RIFE(x) rife_check_cu((x), #x)

class RifeTRTLogger : public nvinfer1::ILogger {
public:
	void log(Severity sev, const char* msg) noexcept override {
		if (sev == Severity::kERROR || sev == Severity::kINTERNAL_ERROR || sev == Severity::kWARNING) {
			logstream << "rife_vfi tensorrt: " << (msg ? msg : "");
		}
	}
};

class RifeVfi : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
public:
	using NodeSISO::NodeSISO;
	bool consumeEofIfPresent() override { return false; }

private:
	// hwaccel
	std::shared_ptr<HWAccelDevice> hwaccel_;
	AVCUDADeviceContext* cuda_dev_ctx_ = nullptr;
	AVBufferRef* hw_frames_ctx_ = nullptr;
	int width_ = 0, height_ = 0;    // input resolution
	int padded_w_ = 0, padded_h_ = 0;  // engine input resolution (multiple of 32)

	// TRT
	RifeTRTLogger logger_;
	nvinfer1::IRuntime* runtime_ = nullptr;
	nvinfer1::ICudaEngine* engine_ = nullptr;
	nvinfer1::IExecutionContext* trt_ctx_ = nullptr;

	// CUDA module + kernel handles
	CUmodule cu_module_ = nullptr;
	CUfunction k_nv12_to_rgb_ = nullptr;
	CUfunction k_rgb_to_nv12_ = nullptr;
	CUfunction k_fill_ts_ = nullptr;

	// Rolling 2-frame window of fp16 RGB tensors (device buffers).
	// input_bufs_[input_idx_] holds the most recent input.
	CUdeviceptr input_bufs_[2] = {0, 0};
	int input_idx_ = 0;
	CUdeviceptr dbuf_mid_ = 0;        // engine output tensor
	CUdeviceptr dbuf_timestep_ = 0;   // 1 fp16 scalar
	size_t rgb_tensor_bytes_ = 0;     // 3 * padded_h_ * padded_w_ * sizeof(fp16)

	// Config
	std::string engine_path_;
	int factor_ = 2;
	bool passthrough_on_fail_ = false;

	// Pipeline state
	bool have_prev_pts_ = false;
	av::Timestamp prev_pts_ = NOTS;
	std::vector<av::VideoFrame> pending_out_;
	size_t emit_idx_ = 0;

	static av::PixelFormat getHwSwPixelFormat(av::VideoFrame &frm)
	{
		if (frm.raw()->hw_frames_ctx == nullptr) return AV_PIX_FMT_NONE;
		AVHWFramesContext* ctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
		if (ctx == nullptr) return AV_PIX_FMT_NONE;
		return ctx->sw_format;
	}

	static int round_up_32(int x) { return ((x + 31) / 32) * 32; }

	// ---------------------------------------------------------------
	// Init / teardown
	// ---------------------------------------------------------------

	bool load_kernels()
	{
		if (cu_module_) return true;
		// PTX images are embedded via xxd; ensure null-terminated for JIT.
		const std::string ptx_str(avpl_rife_ptx, avpl_rife_ptx + avpl_rife_ptx_len);
		if (CHECK_CU_RIFE(cuModuleLoadDataEx(&cu_module_, (const void*)ptx_str.c_str(), 0, nullptr, nullptr))) {
			logstream << "rife_vfi: cuModuleLoadDataEx failed";
			return false;
		}
		bool ok = true;
		ok &= !CHECK_CU_RIFE(cuModuleGetFunction(&k_nv12_to_rgb_, cu_module_, "nv12_to_rgb_fp16"));
		ok &= !CHECK_CU_RIFE(cuModuleGetFunction(&k_rgb_to_nv12_, cu_module_, "rgb_fp16_to_nv12"));
		ok &= !CHECK_CU_RIFE(cuModuleGetFunction(&k_fill_ts_,    cu_module_, "fill_fp16_scalar"));
		if (!ok) logstream << "rife_vfi: cuModuleGetFunction failed for one of the kernels";
		return ok;
	}

	bool load_engine()
	{
		if (engine_) return true;
		std::ifstream f(engine_path_, std::ios::binary | std::ios::ate);
		if (!f) {
			logstream << "rife_vfi: cannot open engine " << engine_path_;
			return false;
		}
		std::streamsize sz = f.tellg();
		std::vector<char> blob((size_t)sz);
		f.seekg(0, std::ios::beg);
		f.read(blob.data(), sz);

		runtime_ = nvinfer1::createInferRuntime(logger_);
		if (!runtime_) { logstream << "rife_vfi: createInferRuntime failed"; return false; }
		engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
		if (!engine_) { logstream << "rife_vfi: deserializeCudaEngine failed"; return false; }
		trt_ctx_ = engine_->createExecutionContext();
		if (!trt_ctx_) { logstream << "rife_vfi: createExecutionContext failed"; return false; }
		return true;
	}

	bool ensure_hw_frames_ctx(int w, int h)
	{
		if (!hwaccel_) return false;
		if (w <= 0 || h <= 0) return false;
		if (hw_frames_ctx_ && w == width_ && h == height_) return true;

		if (hw_frames_ctx_) { av_buffer_unref(&hw_frames_ctx_); hw_frames_ctx_ = nullptr; }
		hw_frames_ctx_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
		if (!hw_frames_ctx_) { logstream << "rife_vfi: av_hwframe_ctx_alloc failed"; return false; }
		AVHWFramesContext *fc = (AVHWFramesContext *)(hw_frames_ctx_->data);
		fc->format = AV_PIX_FMT_CUDA;
		fc->sw_format = AV_PIX_FMT_NV12;
		fc->width = w;
		fc->height = h;
		int r = av_hwframe_ctx_init(hw_frames_ctx_);
		if (r != 0) {
			logstream << "rife_vfi: av_hwframe_ctx_init failed: " << av::error2string(r);
			av_buffer_unref(&hw_frames_ctx_);
			hw_frames_ctx_ = nullptr;
			return false;
		}
		width_ = w;
		height_ = h;
		return true;
	}

	bool ensure_device_buffers()
	{
		const size_t bytes = (size_t)3 * (size_t)padded_h_ * (size_t)padded_w_ * sizeof(uint16_t);
		if (bytes == rgb_tensor_bytes_ && input_bufs_[0] && input_bufs_[1] && dbuf_mid_ && dbuf_timestep_)
			return true;

		free_device_buffers();
		rgb_tensor_bytes_ = bytes;
		if (CHECK_CU_RIFE(cuMemAlloc(&input_bufs_[0], bytes))) return false;
		if (CHECK_CU_RIFE(cuMemAlloc(&input_bufs_[1], bytes))) return false;
		if (CHECK_CU_RIFE(cuMemAlloc(&dbuf_mid_, bytes))) return false;
		if (CHECK_CU_RIFE(cuMemAlloc(&dbuf_timestep_, sizeof(uint16_t)))) return false;
		return true;
	}

	void free_device_buffers()
	{
		for (auto& p : input_bufs_) { if (p) { CHECK_CU_RIFE(cuMemFree(p)); p = 0; } }
		if (dbuf_mid_) { CHECK_CU_RIFE(cuMemFree(dbuf_mid_)); dbuf_mid_ = 0; }
		if (dbuf_timestep_) { CHECK_CU_RIFE(cuMemFree(dbuf_timestep_)); dbuf_timestep_ = 0; }
		rgb_tensor_bytes_ = 0;
	}

	bool ensure_trt_shapes()
	{
		nvinfer1::Dims4 img_shape(1, 3, padded_h_, padded_w_);
		nvinfer1::Dims4 ts_shape(1, 1, 1, 1);
		if (!trt_ctx_->setInputShape("img0", img_shape)) return false;
		if (!trt_ctx_->setInputShape("img1", img_shape)) return false;
		if (!trt_ctx_->setInputShape("timestep", ts_shape)) return false;
		if (!trt_ctx_->setTensorAddress("timestep", (void*)dbuf_timestep_)) return false;
		if (!trt_ctx_->setTensorAddress("mid", (void*)dbuf_mid_)) return false;
		return true;
	}

	// ---------------------------------------------------------------
	// Kernel launches
	// ---------------------------------------------------------------

	bool convert_nv12_to_rgb(const av::VideoFrame &in, CUdeviceptr dst_rgb)
	{
		CUdeviceptr srcY  = (CUdeviceptr)(uintptr_t)in.raw()->data[0];
		CUdeviceptr srcUV = (CUdeviceptr)(uintptr_t)in.raw()->data[1];
		if (!srcY || !srcUV) return false;
		int y_pitch  = in.raw()->linesize[0];
		int uv_pitch = in.raw()->linesize[1];
		int H = height_, W = width_, Hp = padded_h_, Wp = padded_w_;

		void* args[] = { &srcY, &y_pitch, &srcUV, &uv_pitch,
		                 &H, &W, &dst_rgb, &Hp, &Wp };
		dim3 block(16, 16, 1);
		dim3 grid((Wp + block.x - 1) / block.x, (Hp + block.y - 1) / block.y, 1);
		if (CHECK_CU_RIFE(cuLaunchKernel(k_nv12_to_rgb_,
			grid.x, grid.y, grid.z, block.x, block.y, block.z,
			0, cuda_dev_ctx_->stream, args, nullptr))) return false;
		return true;
	}

	bool convert_rgb_to_nv12(CUdeviceptr src_rgb, av::VideoFrame &out)
	{
		CUdeviceptr dstY  = (CUdeviceptr)(uintptr_t)out.raw()->data[0];
		CUdeviceptr dstUV = (CUdeviceptr)(uintptr_t)out.raw()->data[1];
		if (!dstY || !dstUV) return false;
		int y_pitch  = out.raw()->linesize[0];
		int uv_pitch = out.raw()->linesize[1];
		int H = height_, W = width_, Hp = padded_h_, Wp = padded_w_;

		void* args[] = { &src_rgb, &Hp, &Wp,
		                 &dstY, &y_pitch, &dstUV, &uv_pitch,
		                 &H, &W };
		// One thread per 2x2 output block.
		int bw = (W + 1) / 2, bh = (H + 1) / 2;
		dim3 block(16, 16, 1);
		dim3 grid((bw + block.x - 1) / block.x, (bh + block.y - 1) / block.y, 1);
		if (CHECK_CU_RIFE(cuLaunchKernel(k_rgb_to_nv12_,
			grid.x, grid.y, grid.z, block.x, block.y, block.z,
			0, cuda_dev_ctx_->stream, args, nullptr))) return false;
		return true;
	}

	bool set_timestep(float t)
	{
		void* args[] = { &dbuf_timestep_, &t };
		if (CHECK_CU_RIFE(cuLaunchKernel(k_fill_ts_,
			1, 1, 1, 1, 1, 1,
			0, cuda_dev_ctx_->stream, args, nullptr))) return false;
		return true;
	}

	bool run_engine(CUdeviceptr img0, CUdeviceptr img1)
	{
		if (!trt_ctx_->setTensorAddress("img0", (void*)img0)) return false;
		if (!trt_ctx_->setTensorAddress("img1", (void*)img1)) return false;
		if (!trt_ctx_->enqueueV3(cuda_dev_ctx_->stream)) {
			logstream << "rife_vfi: enqueueV3 failed";
			return false;
		}
		return true;
	}

	bool make_output_frame(const av::VideoFrame &ref_in, const av::Timestamp &out_pts, av::VideoFrame &out)
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
			logstream << "rife_vfi: av_hwframe_get_buffer failed";
			return false;
		}
		out.raw()->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
		out.setComplete(true);
		return true;
	}

public:
	void process() override
	{
		// Drain any pending burst outputs first.
		while (emit_idx_ < pending_out_.size()) {
			if (!this->sink_->put(pending_out_[emit_idx_], true)) return;
			emit_idx_++;
		}
		if (!pending_out_.empty()) {
			pending_out_.clear();
			emit_idx_ = 0;
			this->source_->pop();
		}

		av::VideoFrame *pin = this->source_->peek();
		if (!pin) return;
		av::VideoFrame &in = *pin;
		if (!in) { this->source_->pop(); return; }

		if (isEofMarker(in)) {
			if (!this->sink_->put(in, true)) return;
			this->source_->pop();
			have_prev_pts_ = false;
			prev_pts_ = NOTS;
			this->finished_ = true;
			return;
		}

		auto passthrough = [&] {
			if (passthrough_on_fail_) {
				if (!this->sink_->put(in, true)) return;
				this->source_->pop();
			}
		};

		if (in.raw()->format != AV_PIX_FMT_CUDA) { passthrough(); return; }
		if (getHwSwPixelFormat(in) != AV_PIX_FMT_NV12) {
			logstream << "rife_vfi: unsupported CUDA sw_format (need NV12), passthrough";
			passthrough();
			return;
		}

		const int w = in.width();
		const int h = in.height();
		const int Hp = round_up_32(h);
		const int Wp = round_up_32(w);
		if (Hp != 1088 || Wp != 1920) {
			logstream << "rife_vfi: engine is fixed at 1088x1920 (input " << w << "x" << h
			          << " padded to " << Wp << "x" << Hp << "); passthrough";
			passthrough();
			return;
		}

		// Push CUDA context before ANY module/engine/memory calls so all
		// resources live in the same context we later launch kernels in.
		// Without this cuLaunchKernel fails with INVALID_HANDLE because the
		// function belongs to a different context than the current one.
		if (CHECK_CU_RIFE(cuCtxPushCurrent(cuda_dev_ctx_->cuda_ctx))) { passthrough(); return; }
		bool init_ok = load_engine() && load_kernels() && ensure_hw_frames_ctx(w, h);
		padded_h_ = Hp;
		padded_w_ = Wp;
		init_ok = init_ok && ensure_device_buffers() && ensure_trt_shapes();
		if (!init_ok) {
			CUcontext dummy; CHECK_CU_RIFE(cuCtxPopCurrent(&dummy));
			passthrough();
			return;
		}

		av::Timestamp in_pts = in.pts();
		if (!in_pts.isValid()) {
			CUcontext dummy; CHECK_CU_RIFE(cuCtxPopCurrent(&dummy));
			passthrough();
			return;
		}

		// Convert current input to fp16 RGB into the "cur" slot.
		input_idx_ = (input_idx_ + 1) & 1;
		CUdeviceptr cur_rgb = input_bufs_[input_idx_];
		CUdeviceptr prev_rgb = input_bufs_[input_idx_ ^ 1];
		if (!convert_nv12_to_rgb(in, cur_rgb)) {
			CUcontext dummy; CHECK_CU_RIFE(cuCtxPopCurrent(&dummy));
			passthrough();
			return;
		}

		// First frame primes the window: emit input, no interp yet.
		if (!have_prev_pts_) {
			CHECK_CU_RIFE(cuStreamSynchronize(cuda_dev_ctx_->stream));
			CUcontext dummy; CHECK_CU_RIFE(cuCtxPopCurrent(&dummy));
			if (!this->sink_->put(in, true)) return;
			this->source_->pop();
			have_prev_pts_ = true;
			prev_pts_ = in_pts;
			return;
		}

		// For each intermediate timestep, run the engine and produce an output frame.
		const int nsteps = factor_ - 1;
		pending_out_.clear();
		pending_out_.reserve((size_t)nsteps + 1);
		bool step_failed = false;
		for (int i = 1; !step_failed && i <= nsteps; ++i) {
			float t = (float)i / (float)factor_;
			// Output pts = lerp(prev_pts, in_pts, t) in in_pts timebase.
			av::Rational tb = in_pts.timebase();
			int64_t a = prev_pts_.timestamp(tb);
			int64_t b = in_pts.timestamp(tb);
			int64_t m = a + (int64_t)((double)(b - a) * (double)t);
			av::Timestamp out_pts(m, tb);

			if (!set_timestep(t)) { step_failed = true; break; }
			if (!run_engine(prev_rgb, cur_rgb)) { step_failed = true; break; }

			av::VideoFrame out;
			if (!make_output_frame(in, out_pts, out)) { step_failed = true; break; }
			if (!convert_rgb_to_nv12(dbuf_mid_, out)) { step_failed = true; break; }
			pending_out_.push_back(std::move(out));
		}

		if (!step_failed) {
			if (CHECK_CU_RIFE(cuStreamSynchronize(cuda_dev_ctx_->stream))) step_failed = true;
		}

		CUcontext dummy; CHECK_CU_RIFE(cuCtxPopCurrent(&dummy));

		if (step_failed) {
			pending_out_.clear();
			passthrough();
			prev_pts_ = in_pts;
			return;
		}

		// Append current input as the last frame in the burst.
		pending_out_.push_back(in);
		emit_idx_ = 0;
		prev_pts_ = in_pts;
	}

	static std::shared_ptr<RifeVfi> create(NodeCreationInfo &nci)
	{
		if (global_cuda.has_errors) throw Error("rife_vfi: CUDA not initialized");
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		auto src = edges.find<av::VideoFrame>(params["src"]);
		auto dst = edges.find<av::VideoFrame>(params["dst"]);
		auto r = std::make_shared<RifeVfi>(make_unique<EdgeSource<av::VideoFrame>>(src),
		                                    make_unique<EdgeSink<av::VideoFrame>>(dst));
		r->auto_eof_ = false;
		if (!params.count("hwaccel")) throw Error("rife_vfi requires hwaccel parameter");
		r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
		if (!r->hwaccel_) throw Error("rife_vfi: failed to get hwaccel");
		AVHWDeviceContext* devctx = (AVHWDeviceContext *)(r->hwaccel_->deviceContext()->data);
		r->cuda_dev_ctx_ = (AVCUDADeviceContext*)(devctx->hwctx);
		if (!r->cuda_dev_ctx_) throw Error("rife_vfi: CUDA device context missing");

		if (!params.count("engine")) throw Error("rife_vfi: 'engine' parameter (path to .engine) is required");
		r->engine_path_ = params["engine"].get<std::string>();
		if (params.count("factor")) {
			int f = (int)params["factor"];
			if (f < 2 || f > 16) throw Error("rife_vfi: factor must be in [2, 16]");
			r->factor_ = f;
		}
		if (params.count("passthrough_on_fail"))
			r->passthrough_on_fail_ = (bool)params["passthrough_on_fail"];
		return r;
	}

	~RifeVfi() override
	{
		if (trt_ctx_) { delete trt_ctx_; trt_ctx_ = nullptr; }
		if (engine_)  { delete engine_;  engine_  = nullptr; }
		if (runtime_) { delete runtime_; runtime_ = nullptr; }
		free_device_buffers();
		if (cu_module_) { CHECK_CU_RIFE(cuModuleUnload(cu_module_)); cu_module_ = nullptr; }
		if (hw_frames_ctx_) { av_buffer_unref(&hw_frames_ctx_); hw_frames_ctx_ = nullptr; }
	}
};

DECLNODE(rife_vfi, RifeVfi);
