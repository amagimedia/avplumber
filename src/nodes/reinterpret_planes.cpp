#include "node_common.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
}

namespace {

static std::map<int,int> parsePlaneMapObject(const Parameters &params, const char *key) {
	std::map<int,int> result;
	if (params.count(key)) {
		const Parameters &pm = params[key];
		if (!pm.is_object()) {
			throw Error(std::string(key) + " must be an object of {dst_plane: src_plane}");
		}
		for (auto it = pm.begin(); it != pm.end(); ++it) {
			int d = std::stoi(it.key());
			int s = it.value().get<int>();
			result[d] = s;
		}
	}
	return result;
}

static void copyAllBuffersFromSrc(const AVFrame *src, AVFrame *dst) {
	for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
		if (src->buf[i]) {
			dst->buf[i] = av_buffer_ref(src->buf[i]);
			if (!dst->buf[i]) {
				throw Error("av_buffer_ref failed while duplicating source buffers");
			}
		}
	}
	// Note: extended_buf copying is intentionally omitted; typical frames don't use it for planes.
}

static bool isChromaPlane(const AVPixFmtDescriptor *desc, int plane_index) {
	if (!desc) return false;
	if (desc->flags & AV_PIX_FMT_FLAG_RGB) return false;
	if (desc->nb_components < 3) return false;
	int u_plane = desc->comp[1].plane;
	int v_plane = desc->comp[2].plane;
	return plane_index == u_plane || plane_index == v_plane;
}

template<typename T> struct MediaSpecific;

// ---------------- Video ----------------
template<> struct MediaSpecific<av::VideoFrame> {
	struct Params {
		AVPixelFormat dst_pix_fmt = AV_PIX_FMT_NONE;
		std::map<int,int> plane_map; // dst_plane -> src_plane
		bool hw_frames = false;
	};

	static Params parseParams(const Parameters &params) {
		Params p;
		if (params.count("dst_pixel_format") != 1) {
			throw Error("dst_pixel_format is required");
		}
		std::string fmt = params["dst_pixel_format"].get<std::string>();
		p.dst_pix_fmt = av_get_pix_fmt(fmt.c_str());
		if (p.dst_pix_fmt == AV_PIX_FMT_NONE) {
			throw Error("Unknown dst_pixel_format: " + fmt);
		}
		p.plane_map = parsePlaneMapObject(params, "plane_map");
		if (params.count("hw_frames")) {
			p.hw_frames = params["hw_frames"].get<bool>();
		}
		return p;
	}

	static int planeCount(enum AVPixelFormat fmt) {
		int pc = av_pix_fmt_count_planes(fmt);
		if (pc < 0) return 0;
		return pc;
	}

    static void validateAndComputeWH(const AVFrame *src, enum AVPixelFormat src_fmt,
					      const Params &p, int &out_w, int &out_h) {
		// Default mapping: identity for first min(dst_planes, src_planes)
		if (p.plane_map.empty()) {
			// Example: yuva420p -> yuv420p (drop alpha)
			return; // keep src width/height; mapping handled later
		}

		const AVPixFmtDescriptor *src_desc = av_pix_fmt_desc_get(src_fmt);
		const AVPixFmtDescriptor *dst_desc = av_pix_fmt_desc_get(p.dst_pix_fmt);
		if (!src_desc || !dst_desc) {
			throw Error("Failed to get pixel format descriptors");
		}
		int src_hshift = 0, src_vshift = 0;
		av_pix_fmt_get_chroma_sub_sample(src_fmt, &src_hshift, &src_vshift);
		int dst_hshift = 0, dst_vshift = 0;
		av_pix_fmt_get_chroma_sub_sample(p.dst_pix_fmt, &dst_hshift, &dst_vshift);

		int max_w = src->width;
		int max_h = src->height;

		for (const auto &kv : p.plane_map) {
			int dst_plane = kv.first;
			int src_plane = kv.second;
			if (dst_plane < 0 || src_plane < 0) {
				throw Error("plane_map contains negative indices");
			}
			// Linesize constraint
			int unit_bytes = av_image_get_linesize(p.dst_pix_fmt, 1, dst_plane);
			if (unit_bytes <= 0) {
				throw Error("Unsupported dst plane or unit size <= 0");
			}
            int src_line = std::abs(src->linesize[src_plane]);
            if (src_line > 0) {
                int plane_max_w = src_line / unit_bytes;
                if (plane_max_w < max_w) max_w = plane_max_w;
            }

			// Height constraint: rows available in source plane must be >= required rows in dst plane
			int src_plane_h = isChromaPlane(src_desc, src_plane) ? AV_CEIL_RSHIFT(src->height, src_vshift) : src->height;
			int required_dst_rows_factor = isChromaPlane(dst_desc, dst_plane) ? dst_vshift : 0;
			int plane_max_h = src_plane_h << required_dst_rows_factor;
			if (plane_max_h < max_h) max_h = plane_max_h;
		}

		if (max_w <= 0 || max_h <= 0) {
			throw Error("Computed non-positive dimensions for destination frame");
		}
		out_w = std::min(out_w, max_w);
		out_h = std::min(out_h, max_h);

		// Validate final size for dst format
		int ret = av_image_check_size2(out_w, out_h, 0, p.dst_pix_fmt, 0, nullptr);
		if (ret < 0) {
			throw Error("Illegal destination size for pixel format: " + av::error2string(ret));
		}
	}

	static void setOrCloneHWFramesCtx(const AVFrame *src, AVFrame *dst, enum AVPixelFormat desired_sw_format) {
		if (!src->hw_frames_ctx) {
			throw Error("Expected hw_frames_ctx on hardware frame");
		}
		AVHWFramesContext *src_fctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
		if (!src_fctx) {
			throw Error("Invalid src hw_frames_ctx");
		}
		if (src_fctx->sw_format == desired_sw_format) {
			dst->hw_frames_ctx = av_buffer_ref(src->hw_frames_ctx);
			if (!dst->hw_frames_ctx) {
				throw Error("av_buffer_ref failed for hw_frames_ctx");
			}
			return;
		}
		AVBufferRef *device_ref = src_fctx->device_ref;
		if (!device_ref && src_fctx->device_ctx) {
			// Older APIs may not expose device_ref, but we must have a device AVBufferRef
			throw Error("Cannot access device_ref from hw_frames_ctx");
		}
		AVBufferRef *new_frames = av_hwframe_ctx_alloc(device_ref);
		if (!new_frames) {
			throw Error("av_hwframe_ctx_alloc failed");
		}
		AVHWFramesContext *nf = (AVHWFramesContext*)new_frames->data;
		nf->format = src_fctx->format; // keep hardware pixel format unchanged
		nf->sw_format = desired_sw_format; // change software format only
		nf->width = src_fctx->width ? src_fctx->width : src->width;
		nf->height = src_fctx->height ? src_fctx->height : src->height;
		nf->initial_pool_size = src_fctx->initial_pool_size;
		int ir = av_hwframe_ctx_init(new_frames);
		if (ir < 0) {
			av_buffer_unref(&new_frames);
			throw Error("av_hwframe_ctx_init failed: " + av::error2string(ir));
		}
		dst->hw_frames_ctx = new_frames;
	}

	static av::VideoFrame build(const av::VideoFrame &in_frame, const Params &p) {
        const AVFrame *src = in_frame.raw();
        enum AVPixelFormat src_fmt = (enum AVPixelFormat)src->format;
        // Determine logical source format for plane math (use sw_format for HW frames)
        enum AVPixelFormat real_src_fmt = src_fmt;
        if (src->hw_frames_ctx && p.hw_frames) {
            AVHWFramesContext *fctx = (AVHWFramesContext*)src->hw_frames_ctx->data;
            if (!fctx) {
                throw Error("Invalid hw_frames_ctx on source frame");
            }
            real_src_fmt = (enum AVPixelFormat)fctx->sw_format;
        }
        int dst_planes = planeCount(p.dst_pix_fmt);
        int src_planes = planeCount(real_src_fmt);
        if (dst_planes <= 0 || src_planes <= 0) {
            throw Error("Unsupported pixel formats for reinterpretation: " + std::string(av::PixelFormat(real_src_fmt).name()) + " -> " + std::string(av::PixelFormat(p.dst_pix_fmt).name()));
        }

        const AVPixFmtDescriptor *src_desc_hwflag = av_pix_fmt_desc_get(src_fmt);
        bool src_is_hw = (src_desc_hwflag && (src_desc_hwflag->flags & AV_PIX_FMT_FLAG_HWACCEL));
		if (p.hw_frames && !src_is_hw) {
			throw Error("hw_frames=true requires hardware source frame");
		}
		if (!p.hw_frames && src_is_hw) {
			throw Error("Cannot reinterpret hardware frame without hw_frames=true");
		}

		// Compute output dimensions
		int out_w = src->width;
		int out_h = src->height;
        validateAndComputeWH(src, real_src_fmt, p, out_w, out_h);

		av::VideoFrame out;
		AVFrame *dst = out.raw();
		if (p.hw_frames) {
			// Keep hardware format; only change sw_format in frames ctx later
			dst->format = src_fmt;
		} else {
			dst->format = p.dst_pix_fmt;
		}
		dst->width = out_w;
		dst->height = out_h;

		// Build mapping
		std::map<int,int> mapping = p.plane_map;
		if (mapping.empty()) {
			int use_planes = std::min(dst_planes, src_planes);
			for (int i=0; i<use_planes; i++) mapping[i] = i;
		}

		for (auto &kv : mapping) {
			int dp = kv.first;
			int sp = kv.second;
			if (dp < 0 || dp >= dst_planes) {
				throw Error("Invalid dst plane index: " + std::to_string(dp));
			}
			if (sp < 0 || sp >= src_planes) {
				throw Error("Invalid src plane index: " + std::to_string(sp));
			}
			if (!src->data[sp]) {
				throw Error("Null source plane pointer: " + std::to_string(sp));
			}
			// Validate linesize capacity for selected width
			int need_line = av_image_get_linesize(p.dst_pix_fmt, out_w, dp);
			if (need_line < 0) {
				throw Error("av_image_get_linesize failed");
			}
			if (std::abs(src->linesize[sp]) < need_line) {
				throw Error("Source linesize too small for mapped plane");
			}
			dst->data[dp] = src->data[sp];
			dst->linesize[dp] = src->linesize[sp];
		}
		// Duplicate buffer ownership once, after data pointers are set
		copyAllBuffersFromSrc(src, dst);
		// Ensure extended_data points to data
		dst->extended_data = dst->data;

		if (p.hw_frames) {
			setOrCloneHWFramesCtx(src, dst, p.dst_pix_fmt);
		}
		return out;
	}
};

// ---------------- Audio ----------------
template<> struct MediaSpecific<av::AudioSamples> {
	struct Params {
		AVSampleFormat dst_sample_fmt = AV_SAMPLE_FMT_NONE;
		std::map<int,int> plane_map; // dst_plane/channel -> src_plane/channel
	};

	static Params parseParams(const Parameters &params) {
		Params p;
		if (params.count("dst_sample_format") != 1) {
			throw Error("dst_sample_format is required for audio");
		}
		std::string fmt = params["dst_sample_format"].get<std::string>();
		p.dst_sample_fmt = av_get_sample_fmt(fmt.c_str());
		if (p.dst_sample_fmt == AV_SAMPLE_FMT_NONE) {
			throw Error("Unknown dst_sample_format: " + fmt);
		}
		p.plane_map = parsePlaneMapObject(params, "plane_map");
		return p;
	}

	static av::AudioSamples build(const av::AudioSamples &in, const Params &p) {
		const AVFrame *src = in.raw();
		AVSampleFormat src_fmt = (AVSampleFormat)src->format;
		bool src_planar = av_sample_fmt_is_planar(src_fmt) != 0;
		bool dst_planar = av_sample_fmt_is_planar(p.dst_sample_fmt) != 0;
		if (src_planar != dst_planar) {
			throw Error("Planar/interleaved mismatch without copying is not supported");
		}
		if (!dst_planar) {
			// Interleaved: only safe if formats exactly match
			if (p.dst_sample_fmt != src_fmt) {
				throw Error("Interleaved audio reinterpret requires identical sample format");
			}
			// No plane mapping possible; just pass-through
			return in;
		}

		// Planar mapping
		int dst_planes = p.plane_map.empty() ? in.channelsCount() : (int)p.plane_map.size();
		int bytes_per_sample = av_get_bytes_per_sample(p.dst_sample_fmt);
		if (bytes_per_sample <= 0) {
			throw Error("Invalid bytes per sample for destination format");
		}

		av::AudioSamples out;
		AVFrame *dst = out.raw();
		dst->format = p.dst_sample_fmt;
		dst->nb_samples = src->nb_samples;
		dst->sample_rate = src->sample_rate;
		// Set channel layout to unspecified with correct channel count
		#if API_NEW_CHANNEL_LAYOUT
		av_channel_layout_uninit(&dst->ch_layout);
		dst->ch_layout.order = AV_CHANNEL_ORDER_UNSPEC;
		dst->ch_layout.nb_channels = dst_planes;
		dst->ch_layout.u.mask = 0;
		#else
		dst->channels = dst_planes;
		dst->channel_layout = 0;
		#endif

		// Default mapping: identity for first planes
		std::map<int,int> mapping = p.plane_map;
		if (mapping.empty()) {
			int max_planes = std::min(dst_planes, in.channelsCount());
			for (int i=0; i<max_planes; i++) mapping[i] = i;
		}

		for (auto &kv : mapping) {
			int dp = kv.first;
			int sp = kv.second;
			if (dp < 0 || sp < 0) {
				throw Error("plane_map contains negative indices");
			}
			if (sp >= in.channelsCount()) {
				throw Error("Invalid src plane/channel index");
			}
			if (!src->data[sp]) {
				throw Error("Null source audio plane pointer");
			}
			int need_bytes = bytes_per_sample * src->nb_samples;
			int avail = src->linesize[sp] > 0 ? src->linesize[sp] : need_bytes; // best effort
			if (avail < need_bytes) {
				throw Error("Source audio plane too small for requested format");
			}
			dst->data[dp] = src->data[sp];
			dst->linesize[dp] = src->linesize[sp];
		}
		copyAllBuffersFromSrc(src, dst);
		dst->extended_data = dst->data;
		return out;
	}
};

template<typename T> class ReinterpretPlanes: public NodeSISO<T, T>, public NodeDoesNotBuffer {
protected:
	using MS = MediaSpecific<T>;
	typename MS::Params params_;
public:
	using NodeSISO<T, T>::NodeSISO;
	virtual void process() {
		T in = this->source_->get();
		if (isEofMarker(in)) { this->sink_->put(in); return; }
		T out = MS::build(in, params_);
        out.setTimeBase(in.timeBase());
		out.setPts(in.pts());
        out.setComplete(in.isComplete());
        out.setStreamIndex(in.streamIndex());
		this->sink_->put(out);
	}
	virtual void init(EdgeManager &edges, const Parameters &params) {
		NodeSISO<T, T>::init(edges, params);
		params_ = MS::parseParams(params);
	}
	static std::shared_ptr<ReinterpretPlanes> create(NodeCreationInfo &nci) {
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		std::shared_ptr<ReinterpretPlanes> r = NodeSISO<T, T>::template createCommon<ReinterpretPlanes>(edges, params);
		r->params_ = MS::parseParams(params);
		return r;
	}
};

} // namespace

class VideoReinterpretPlanes: public ReinterpretPlanes<av::VideoFrame> {};
class AudioReinterpretPlanes: public ReinterpretPlanes<av::AudioSamples> {};

DECLNODE(reinterpret_planes_video, VideoReinterpretPlanes);
DECLNODE(reinterpret_planes_audio, AudioReinterpretPlanes);
