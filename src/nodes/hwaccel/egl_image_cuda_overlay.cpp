#include "../node_common.hpp"
#include "../../cuda.hpp"
#include "../../hwaccel.hpp"
#include "../../hwaccel/EglImageFrame.hpp"
#include "../../../deps/cuda_loader/cuda_drvapi_dynlink_gl.h"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../../objs/src/nodes/hwaccel/egl_image_cuda_overlay.ptx.h"

namespace {

static int checkCu(CUresult error, const char *function) {
	if (error == CUDA_SUCCESS)
		return 0;
	const char *name = nullptr;
	const char *message = nullptr;
	if (cuGetErrorName && cuGetErrorString) {
		cuGetErrorName(error, &name);
		cuGetErrorString(error, &message);
	}
	logstream << "egl_image_cuda_overlay: " << function << " failed: "
	          << (name ? name : "?") << ": " << (message ? message : "?");
	return -1;
}

#define CHECK_CU(call) checkCu((call), #call)

struct LayerSpec {
	int dst_x = 0;
	int dst_y = 0;
	int dst_w = 0;
	int dst_h = 0;
};

static std::vector<LayerSpec> parseLayers(const Parameters &params) {
	if (!params.contains("layers") || !params["layers"].is_array())
		throw Error("egl_image_cuda_overlay: layers array required");
	std::vector<LayerSpec> layers;
	for (const auto &item : params["layers"]) {
		if (!item.is_object())
			throw Error("egl_image_cuda_overlay: every layer must be an object");
		LayerSpec layer;
		layer.dst_x = item.value("dst_x", 0);
		layer.dst_y = item.value("dst_y", 0);
		layer.dst_w = item.value("dst_w", 0);
		layer.dst_h = item.value("dst_h", 0);
		if (layer.dst_w <= 0 || layer.dst_h <= 0)
			throw Error("egl_image_cuda_overlay: layer dst_w and dst_h must be positive");
		layers.push_back(layer);
	}
	return layers;
}

static bool frameUsable(const EglImageFrame &frame) {
	return frame.isComplete() && frame.pts().isValid();
}

static const char *frameTypeName(CUeglFrameType type) {
	switch (type) {
	case CU_EGL_FRAME_TYPE_ARRAY:
		return "array";
	case CU_EGL_FRAME_TYPE_PITCH:
		return "pitch";
	default:
		return "unknown";
	}
}

static const char *colorFormatName(CUeglColorFormat format) {
	switch (format) {
	case CU_EGL_COLOR_FORMAT_RGB:
		return "rgb";
	case CU_EGL_COLOR_FORMAT_BGR:
		return "bgr";
	case CU_EGL_COLOR_FORMAT_ARGB:
		return "argb";
	case CU_EGL_COLOR_FORMAT_RGBA:
		return "rgba";
	case CU_EGL_COLOR_FORMAT_ABGR:
		return "abgr";
	case CU_EGL_COLOR_FORMAT_BGRA:
		return "bgra";
	default:
		return "unsupported";
	}
}

static bool colorLanes(CUeglColorFormat format, int &red, int &green, int &blue) {
	// cudaEGL.h documents the byte ordering returned for each named format.
	switch (format) {
	case CU_EGL_COLOR_FORMAT_RGB:
	case CU_EGL_COLOR_FORMAT_ARGB:
		red = 2;
		green = 1;
		blue = 0;
		return true;
	case CU_EGL_COLOR_FORMAT_BGR:
	case CU_EGL_COLOR_FORMAT_ABGR:
		red = 0;
		green = 1;
		blue = 2;
		return true;
	case CU_EGL_COLOR_FORMAT_RGBA:
		red = 3;
		green = 2;
		blue = 1;
		return true;
	case CU_EGL_COLOR_FORMAT_BGRA:
		red = 1;
		green = 2;
		blue = 3;
		return true;
	default:
		return false;
	}
}

} // namespace

class EglImageCudaOverlay : public NodeMultiInput<EglImageFrame>,
	                           public NodeSingleOutput<av::VideoFrame>,
	                           public IVideoFormatSource,
	                           public IFrameRateSource,
	                           public ITimeBaseSource {
	using Clock = std::chrono::steady_clock;

	struct InteropEntry {
		EGLImageKHR source_image = EGL_NO_IMAGE_KHR;
		std::shared_ptr<void> allocation_holder;
		CUcontext cuda_context = nullptr;
		CUgraphicsResource resource = nullptr;
		CUeglFrame egl_frame{};
		CUtexObject texture = 0;
		int source_width = 0;
		int source_height = 0;
		int red_lane = 0;
		int green_lane = 1;
		int blue_lane = 2;
		int64_t last_used_ms = 0;

		~InteropEntry() {
			if ((!texture && !resource) || !cuda_context)
				return;
			CUcontext previous = nullptr;
			if (cuCtxPushCurrent(cuda_context) != CUDA_SUCCESS)
				return;
			if (texture)
				CHECK_CU(cuTexObjectDestroy(texture));
			if (resource)
				CHECK_CU(cuGraphicsUnregisterResource(resource));
			CHECK_CU(cuCtxPopCurrent(&previous));
		}
	};

	struct InFlightRead {
		CUevent event = nullptr;
		std::vector<std::shared_ptr<void>> frame_holders;
		std::vector<std::shared_ptr<InteropEntry>> entries;
	};

	std::shared_ptr<HWAccelDevice> hwaccel_;
	AVBufferRef *out_frames_ref_ = nullptr;
	AVCUDADeviceContext *cuda_device_ = nullptr;
	CUmodule cuda_module_ = nullptr;
	CUfunction clear_kernel_ = nullptr;
	CUfunction composite_kernel_ = nullptr;

	int canvas_width_ = 0;
	int canvas_height_ = 0;
	std::vector<LayerSpec> layers_;
	int64_t cache_ttl_ms_ = 3000;
	size_t max_cache_entries_ = 440;
	int debug_log_every_n_ = 0;

	av::Rational frame_rate_{60, 1};
	av::Rational output_time_base_{1, 60};
	Clock::duration frame_period_ = std::chrono::duration_cast<Clock::duration>(
		std::chrono::duration<double>(1.0 / 60.0));
	Clock::time_point next_tick_{};
	bool clock_started_ = false;
	int64_t next_output_pts_ = 0;

	uint64_t rendered_ticks_ = 0;
	uint64_t missed_deadlines_ = 0;
	std::vector<uint64_t> reuse_counts_;
	std::vector<bool> updated_since_tick_;
	std::vector<Clock::time_point> last_update_;

	std::vector<EglImageFrame> held_;
	std::vector<bool> held_valid_;
	std::vector<bool> input_eof_;
	std::vector<std::shared_ptr<InteropEntry>> interop_cache_;
	std::vector<InFlightRead> in_flight_reads_;
	std::vector<CUevent> free_events_;

	bool pushCuda() {
		return cuda_device_ && !CHECK_CU(cuCtxPushCurrent(cuda_device_->cuda_ctx));
	}

	void popCuda() {
		CUcontext previous = nullptr;
		CHECK_CU(cuCtxPopCurrent(&previous));
	}

	void setFrameRate(const av::Rational &frame_rate) {
		const double fps = frame_rate.getDouble();
		if (!(fps > 0.0))
			throw Error("egl_image_cuda_overlay: fps must be positive");
		frame_rate_ = frame_rate;
		output_time_base_ = av::Rational(frame_rate.getDenominator(), frame_rate.getNumerator());
		frame_period_ = std::chrono::duration_cast<Clock::duration>(
			std::chrono::duration<double>(1.0 / fps));
		if (frame_period_ <= Clock::duration::zero())
			throw Error("egl_image_cuda_overlay: fps is too high");
	}

	bool ensureCudaModule() {
		if (cuda_module_ && clear_kernel_ && composite_kernel_)
			return true;
		char error_log[8192] = {};
		char info_log[8192] = {};
		CUjit_option options[] = {
			CU_JIT_ERROR_LOG_BUFFER,
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_INFO_LOG_BUFFER,
			CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		};
		size_t error_size = sizeof(error_log);
		size_t info_size = sizeof(info_log);
		void *values[] = {error_log, reinterpret_cast<void *>(error_size), info_log,
		                  reinterpret_cast<void *>(info_size)};
		const std::string module_image(avpl_egl_cuda_overlay_ptx,
		                               avpl_egl_cuda_overlay_ptx + avpl_egl_cuda_overlay_ptx_len);
		if (CHECK_CU(cuModuleLoadDataEx(&cuda_module_, module_image.c_str(), 4, options, values))) {
			if (error_log[0])
				logstream << "egl_image_cuda_overlay: CUDA JIT: " << error_log;
			return false;
		}
		if (CHECK_CU(cuModuleGetFunction(&clear_kernel_, cuda_module_, "clear_rgb0")) ||
		    CHECK_CU(cuModuleGetFunction(&composite_kernel_, cuda_module_, "composite_rgba_texture")))
			return false;
		return true;
	}

	void pruneInteropCache(int64_t now_ms) {
		if (cache_ttl_ms_ <= 0)
			return;
		interop_cache_.erase(
			std::remove_if(interop_cache_.begin(), interop_cache_.end(), [=](const auto &entry) {
				return !entry || now_ms - entry->last_used_ms >= cache_ttl_ms_;
			}),
			interop_cache_.end());
	}

	std::shared_ptr<InteropEntry> createInteropEntry(const EglImageFrame &frame,
	                                                 int64_t now_ms) {
		auto entry = std::make_shared<InteropEntry>();
		entry->source_image = reinterpret_cast<EGLImageKHR>(frame.image());
		entry->allocation_holder = frame.holder();
		entry->cuda_context = cuda_device_->cuda_ctx;
		entry->source_width = frame.width();
		entry->source_height = frame.height();
		entry->last_used_ms = now_ms;

		if (CHECK_CU(cuGraphicsEGLRegisterImage(
			&entry->resource, entry->source_image, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
			return nullptr;
		if (CHECK_CU(cuGraphicsResourceGetMappedEglFrame(
			&entry->egl_frame, entry->resource, 0, 0)))
			return nullptr;

		const CUeglFrame &egl_frame = entry->egl_frame;
		if (egl_frame.planeCount != 1 || egl_frame.width != static_cast<unsigned int>(frame.width()) ||
		    egl_frame.height != static_cast<unsigned int>(frame.height()) ||
		    egl_frame.numChannels < 3 || egl_frame.numChannels > 4 ||
		    egl_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8) {
			logstream << "egl_image_cuda_overlay: unsupported EGL frame planes=" << egl_frame.planeCount
			          << " size=" << egl_frame.width << "x" << egl_frame.height
			          << " channels=" << egl_frame.numChannels
			          << " cu_format=" << static_cast<int>(egl_frame.cuFormat);
			return nullptr;
		}
		if (!colorLanes(egl_frame.eglColorFormat, entry->red_lane, entry->green_lane,
		                entry->blue_lane)) {
			logstream << "egl_image_cuda_overlay: unsupported EGL color format "
			          << static_cast<int>(egl_frame.eglColorFormat);
			return nullptr;
		}

		CUDA_RESOURCE_DESC resource_desc{};
		if (egl_frame.frameType == CU_EGL_FRAME_TYPE_ARRAY && egl_frame.frame.pArray[0]) {
			resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
			resource_desc.res.array.hArray = egl_frame.frame.pArray[0];
		} else if (egl_frame.frameType == CU_EGL_FRAME_TYPE_PITCH && egl_frame.frame.pPitch[0] &&
		           egl_frame.pitch > 0) {
			resource_desc.resType = CU_RESOURCE_TYPE_PITCH2D;
			resource_desc.res.pitch2D.devPtr = static_cast<CUdeviceptr>(
				reinterpret_cast<uintptr_t>(egl_frame.frame.pPitch[0]));
			resource_desc.res.pitch2D.format = egl_frame.cuFormat;
			resource_desc.res.pitch2D.numChannels = egl_frame.numChannels;
			resource_desc.res.pitch2D.width = egl_frame.width;
			resource_desc.res.pitch2D.height = egl_frame.height;
			resource_desc.res.pitch2D.pitchInBytes = egl_frame.pitch;
		} else {
			logstream << "egl_image_cuda_overlay: unsupported EGL frame type "
			          << static_cast<int>(egl_frame.frameType);
			return nullptr;
		}

		CUDA_TEXTURE_DESC texture_desc{};
		texture_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
		texture_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		texture_desc.filterMode = CU_TR_FILTER_MODE_LINEAR;
		texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
		if (CHECK_CU(cuTexObjectCreate(&entry->texture, &resource_desc, &texture_desc, nullptr)))
			return nullptr;

		if (interop_cache_.size() >= max_cache_entries_) {
			auto oldest = std::min_element(
				interop_cache_.begin(), interop_cache_.end(),
				[](const auto &left, const auto &right) {
					return left->last_used_ms < right->last_used_ms;
				});
			interop_cache_.erase(oldest);
		}
		interop_cache_.push_back(entry);
		logstream << "egl_image_cuda_overlay: registered EGLImage size="
		          << entry->source_width << "x" << entry->source_height
		          << " type=" << frameTypeName(egl_frame.frameType)
		          << " color=" << colorFormatName(egl_frame.eglColorFormat)
		          << " channels=" << egl_frame.numChannels
		          << " lanes=" << entry->red_lane << "," << entry->green_lane
		          << "," << entry->blue_lane;
		return entry;
	}

	std::shared_ptr<InteropEntry> interopFor(const EglImageFrame &frame, int64_t now_ms) {
		const EGLImageKHR source_image = reinterpret_cast<EGLImageKHR>(frame.image());
		for (const auto &entry : interop_cache_) {
			if (entry->source_image == source_image && entry->source_width == frame.width() &&
			    entry->source_height == frame.height()) {
				entry->last_used_ms = now_ms;
				return entry;
			}
		}
		return createInteropEntry(frame, now_ms);
	}

	bool pollInFlightReads() {
		for (auto it = in_flight_reads_.begin(); it != in_flight_reads_.end();) {
			const CUresult result = cuEventQuery(it->event);
			if (result == CUDA_ERROR_NOT_READY) {
				++it;
				continue;
			}
			if (result != CUDA_SUCCESS) {
				CHECK_CU(result);
				return false;
			}
			it->frame_holders.clear();
			it->entries.clear();
			free_events_.push_back(it->event);
			it = in_flight_reads_.erase(it);
		}
		return true;
	}

	bool recordInFlightRead(const std::vector<std::shared_ptr<InteropEntry>> &sources,
	                        CUstream stream) {
		InFlightRead read;
		for (size_t index = 0; index < sources.size(); ++index) {
			if (!sources[index])
				continue;
			if (std::find(read.entries.begin(), read.entries.end(), sources[index]) == read.entries.end())
				read.entries.push_back(sources[index]);
			const auto &holder = held_[index].frameHolder();
			if (holder && std::find(read.frame_holders.begin(), read.frame_holders.end(), holder) ==
			                  read.frame_holders.end())
				read.frame_holders.push_back(holder);
		}
		if (read.entries.empty())
			return true;

		if (!free_events_.empty()) {
			read.event = free_events_.back();
			free_events_.pop_back();
		} else if (CHECK_CU(cuEventCreate(&read.event, CU_EVENT_DISABLE_TIMING))) {
			return false;
		}
		if (CHECK_CU(cuEventRecord(read.event, stream))) {
			CHECK_CU(cuEventDestroy(read.event));
			return false;
		}
		in_flight_reads_.push_back(std::move(read));
		return true;
	}

	bool render(int64_t output_pts) {
		const int64_t now_ms = wallclock.pts();
		pruneInteropCache(now_ms);

		av::VideoFrame output;
		const int alloc_result = av_hwframe_get_buffer(out_frames_ref_, output.raw(), 0);
		if (alloc_result < 0) {
			logstream << "egl_image_cuda_overlay: av_hwframe_get_buffer failed: "
			          << av::error2string(alloc_result);
			return false;
		}
		output.raw()->format = AV_PIX_FMT_CUDA;
		output.raw()->width = canvas_width_;
		output.raw()->height = canvas_height_;
		output.raw()->color_range = AVCOL_RANGE_JPEG;
		output.raw()->colorspace = AVCOL_SPC_RGB;
		output.setTimeBase(output_time_base_);
		output.setPts(av::Timestamp(output_pts, output_time_base_));
		output.setComplete(true);

		if (!pushCuda())
			return false;
		bool ok = pollInFlightReads() && ensureCudaModule();
		std::vector<std::shared_ptr<InteropEntry>> sources(held_.size());
		for (size_t index = 0; ok && index < held_.size(); ++index) {
			if (held_valid_[index]) {
				sources[index] = interopFor(held_[index], now_ms);
				ok = static_cast<bool>(sources[index]);
			}
		}

		CUstream stream = reinterpret_cast<CUstream>(cuda_device_->stream);
		CUdeviceptr destination = reinterpret_cast<CUdeviceptr>(output.raw()->data[0]);
		size_t destination_pitch = static_cast<size_t>(output.raw()->linesize[0]);
		void *clear_args[] = {
			&destination, &destination_pitch, &canvas_width_, &canvas_height_,
		};
		const unsigned int block_x = 32;
		const unsigned int block_y = 8;
		if (ok)
			ok = !CHECK_CU(cuLaunchKernel(
				clear_kernel_,
				(canvas_width_ + block_x - 1) / block_x,
				(canvas_height_ + block_y - 1) / block_y,
				1, block_x, block_y, 1, 0, stream, clear_args, nullptr));

		for (size_t index = 0; ok && index < sources.size(); ++index) {
			const auto &source = sources[index];
			if (!source)
				continue;
			const LayerSpec &layer = layers_[index];
			void *args[] = {
				&source->texture,
				&destination,
				&destination_pitch,
				const_cast<int *>(&layer.dst_x),
				const_cast<int *>(&layer.dst_y),
				const_cast<int *>(&layer.dst_w),
				const_cast<int *>(&layer.dst_h),
				&source->red_lane,
				&source->green_lane,
				&source->blue_lane,
			};
			ok = !CHECK_CU(cuLaunchKernel(
				composite_kernel_,
				(layer.dst_w + block_x - 1) / block_x,
				(layer.dst_h + block_y - 1) / block_y,
				1, block_x, block_y, 1, 0, stream, args, nullptr));
		}
		if (ok)
			ok = recordInFlightRead(sources, stream);
		popCuda();
		if (!ok)
			return false;

		this->sink_->put(std::move(output));
		++rendered_ticks_;
		if (debug_log_every_n_ > 0 &&
		    rendered_ticks_ % static_cast<uint64_t>(debug_log_every_n_) == 0) {
			std::ostringstream stats;
			stats << "egl_image_cuda_overlay: frames=" << rendered_ticks_
			      << " missed_deadlines=" << missed_deadlines_
			      << " cache=" << interop_cache_.size()
			      << " in_flight=" << in_flight_reads_.size();
			const auto now = Clock::now();
			for (size_t index = 0; index < reuse_counts_.size(); ++index) {
				stats << " input" << index << "_reuse=" << reuse_counts_[index];
				if (held_valid_[index]) {
					const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
						now - last_update_[index]);
					stats << " input" << index << "_age_ms=" << age.count();
				}
			}
			logstream << stats.str();
		}
		return true;
	}

	void drainLatest(size_t input) {
		while (true) {
			EglImageFrame *frame = this->source_edges_[input]->peek();
			if (!frame)
				return;
			if (isEofMarker(*frame)) {
				this->source_edges_[input]->pop();
				input_eof_[input] = true;
				return;
			}
			if (frameUsable(*frame)) {
				held_[input] = *frame;
				held_valid_[input] = true;
				updated_since_tick_[input] = true;
				last_update_[input] = Clock::now();
			}
			this->source_edges_[input]->pop();
		}
	}

	void drainAllInputs() {
		for (size_t input = 0; input < this->source_edges_.size(); ++input)
			drainLatest(input);
	}

	int millisecondsUntil(Clock::time_point deadline) const {
		const auto remaining = deadline - Clock::now();
		if (remaining <= Clock::duration::zero())
			return 0;
		const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
		const int64_t millis = (nanos + 999999) / 1000000;
		return static_cast<int>(std::min<int64_t>(millis, std::numeric_limits<int>::max()));
	}

	void advanceClock() {
		next_tick_ += frame_period_;
		++next_output_pts_;
		const auto now = Clock::now();
		while (next_tick_ <= now) {
			next_tick_ += frame_period_;
			++next_output_pts_;
			++missed_deadlines_;
		}
	}

	void destroyResources() {
		if (pushCuda()) {
			for (auto &read : in_flight_reads_) {
				CHECK_CU(cuEventSynchronize(read.event));
				CHECK_CU(cuEventDestroy(read.event));
			}
			in_flight_reads_.clear();
			for (CUevent event : free_events_)
				CHECK_CU(cuEventDestroy(event));
			free_events_.clear();
			interop_cache_.clear();
			if (cuda_module_)
				CHECK_CU(cuModuleUnload(cuda_module_));
			popCuda();
		}
		cuda_module_ = nullptr;
		clear_kernel_ = nullptr;
		composite_kernel_ = nullptr;
		av_buffer_unref(&out_frames_ref_);
	}

public:
	EglImageCudaOverlay(std::unique_ptr<Sink<av::VideoFrame>> &&sink,
	                    std::shared_ptr<HWAccelDevice> hwaccel,
	                    int width,
	                    int height,
	                    std::vector<LayerSpec> layers)
		: NodeSingleOutput<av::VideoFrame>(std::move(sink)),
		  hwaccel_(std::move(hwaccel)),
		  canvas_width_(width),
		  canvas_height_(height),
		  layers_(std::move(layers)) {
		AVHWDeviceContext *device_context = reinterpret_cast<AVHWDeviceContext *>(
			hwaccel_->deviceContext()->data);
		cuda_device_ = reinterpret_cast<AVCUDADeviceContext *>(device_context->hwctx);
		if (!cuda_device_ || !cuda_device_->cuda_ctx)
			throw Error("egl_image_cuda_overlay: CUDA device context missing");

		out_frames_ref_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
		if (!out_frames_ref_)
			throw Error("egl_image_cuda_overlay: av_hwframe_ctx_alloc failed");
		auto *frames = reinterpret_cast<AVHWFramesContext *>(out_frames_ref_->data);
		frames->format = AV_PIX_FMT_CUDA;
		frames->sw_format = AV_PIX_FMT_RGB0;
		frames->width = canvas_width_;
		frames->height = canvas_height_;
		const int result = av_hwframe_ctx_init(out_frames_ref_);
		if (result < 0)
			throw Error(std::string("egl_image_cuda_overlay: av_hwframe_ctx_init failed: ") +
			            av::error2string(result));

		held_.resize(layers_.size());
		held_valid_.resize(layers_.size());
		input_eof_.resize(layers_.size());
		reuse_counts_.resize(layers_.size());
		updated_since_tick_.resize(layers_.size());
		last_update_.resize(layers_.size());
		this->auto_eof_ = false;
	}

	~EglImageCudaOverlay() override { destroyResources(); }

	void init(EdgeManager &edges, const Parameters &params) override {
		NodeSingleOutput<av::VideoFrame>::init(edges, params);
	}

	std::weak_ptr<Node> sourceNode() override {
		return source_edges_.empty() ? std::weak_ptr<Node>() : source_edges_.front()->producer();
	}

	std::shared_ptr<EdgeBase> sourceEdge() override {
		return source_edges_.empty() ? std::shared_ptr<EdgeBase>() : source_edges_.front();
	}

	void process() override {
		if (source_edges_.empty())
			return;
		drainAllInputs();

		if (!clock_started_) {
			next_tick_ = Clock::now();
			clock_started_ = true;
		}
		if (Clock::now() < next_tick_) {
			const int ready = this->findSourceWithData(millisecondsUntil(next_tick_));
			if (ready >= 0)
				drainAllInputs();
		}
		if (Clock::now() < next_tick_)
			return;

		for (size_t input = 0; input < held_.size(); ++input) {
			if (held_valid_[input] && !updated_since_tick_[input])
				++reuse_counts_[input];
		}
		if (!render(next_output_pts_))
			throw Error("egl_image_cuda_overlay: render failed");
		std::fill(updated_since_tick_.begin(), updated_since_tick_.end(), false);
		advanceClock();
	}

	int width() override { return canvas_width_; }
	int height() override { return canvas_height_; }
	av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
	av::PixelFormat realPixelFormat() override { return av::PixelFormat(AV_PIX_FMT_RGB0); }
	av::Rational frameRate() override { return frame_rate_; }
	av::Rational timeBase() override { return output_time_base_; }

	static std::shared_ptr<EglImageCudaOverlay> create(NodeCreationInfo &nci) {
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		const auto source_names = jsonToStringList(params["src"]);
		if (source_names.empty())
			throw Error("egl_image_cuda_overlay: at least one input required");
		auto layers = parseLayers(params);
		if (layers.size() != source_names.size())
			throw Error("egl_image_cuda_overlay: layers length must match src length");

		const int width = params.at("width").get<int>();
		const int height = params.at("height").get<int>();
		if (width <= 0 || height <= 0)
			throw Error("egl_image_cuda_overlay: width and height must be positive");
		for (const LayerSpec &layer : layers) {
			if (layer.dst_x < 0 || layer.dst_y < 0 ||
			    layer.dst_x + layer.dst_w > width || layer.dst_y + layer.dst_h > height)
				throw Error("egl_image_cuda_overlay: layer rectangle exceeds canvas");
		}

		if (!params.contains("hwaccel"))
			throw Error("egl_image_cuda_overlay: hwaccel parameter required");
		auto hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
		if (!hwaccel)
			throw Error("egl_image_cuda_overlay: failed to resolve hwaccel");
		auto output = edges.find<av::VideoFrame>(params["dst"]);
		auto node = std::make_shared<EglImageCudaOverlay>(
			make_unique<EdgeSink<av::VideoFrame>>(output), std::move(hwaccel), width, height,
			std::move(layers));
		node->createSourcesFromParameters(edges, params);
		output->setProducer(node);

		node->setFrameRate(parseRatio(params.value("fps", std::string("60/1"))));
		const double ttl_seconds = params.value("cache_ttl", 3.0);
		if (ttl_seconds < 0)
			throw Error("egl_image_cuda_overlay: cache_ttl must be >= 0");
		node->cache_ttl_ms_ = static_cast<int64_t>(ttl_seconds * 1000.0 + 0.5);
		const int max_entries = params.value("max_cache_entries", 440);
		if (max_entries < 1)
			throw Error("egl_image_cuda_overlay: max_cache_entries must be >= 1");
		node->max_cache_entries_ = static_cast<size_t>(max_entries);
		node->debug_log_every_n_ = params.value("debug_log_every_n", 0);
		if (node->debug_log_every_n_ < 0)
			throw Error("egl_image_cuda_overlay: debug_log_every_n must be >= 0");
		return node;
	}
};

DECLNODE(egl_image_cuda_overlay, EglImageCudaOverlay);
