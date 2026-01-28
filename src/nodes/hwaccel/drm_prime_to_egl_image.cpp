#define EGL_EGLEXT_PROTOTYPES 1
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "../node_common.hpp"
#include "../../hwaccel/EglImageFrame.hpp"
#include "../../hwaccel/EglImagePoolToken.hpp"

extern "C" {
#include <libavutil/hwcontext_drm.h>
#include <libdrm/drm_fourcc.h>
}

#include <unistd.h>
#include <sys/stat.h>

#include <cstring>
#include <unordered_map>
#include <memory>
#include <string>

class DRMPrimeToEglImage : public NodeSISO<av::VideoFrame, EglImageFrame> {
protected:
	struct EglState {
		EGLDisplay dpy = EGL_NO_DISPLAY;
		PFNEGLDESTROYIMAGEKHRPROC destroy_fn = nullptr;
		~EglState() {
			// Must outlive any EGLImage created from it.
			if (dpy != EGL_NO_DISPLAY) {
				//eglTerminate(dpy);
				dpy = EGL_NO_DISPLAY;
			}
		}
	};

	struct Entry {
		std::shared_ptr<EglState> egl;

		EGLImageKHR image = EGL_NO_IMAGE_KHR;
		int dup_fd = -1;
		dev_t st_dev{};
		ino_t st_ino{};

		// Attributes used for validation / cache correctness
		uint32_t fourcc = 0;
		int width = 0;
		int height = 0;
		uint32_t pitch = 0;
		uint32_t offset = 0;
		uint64_t modifier = 0;

		int64_t last_seen_ms = 0;

		~Entry() {
			if (image != EGL_NO_IMAGE_KHR && egl && egl->dpy != EGL_NO_DISPLAY && egl->destroy_fn) {
				egl->destroy_fn(egl->dpy, image);
				image = EGL_NO_IMAGE_KHR;
			}
			if (dup_fd >= 0) {
				close(dup_fd);
				dup_fd = -1;
			}
		}
	};

	// Cache keyed by *incoming* FD number (per requirement); entries hold a dup() of FD.
	std::unordered_map<int, std::shared_ptr<Entry>> cache_;

	// Eviction + format tracking
	int64_t ttl_ms_ = 5000;
	int last_w_ = 0;
	int last_h_ = 0;
	int64_t last_purge_scan_ms_ = 0;
	int64_t purge_scan_interval_ms_ = 250;

	// EGL state
	std::shared_ptr<EglState> egl_;
	bool have_dma_buf_import_ = false;
	bool have_mods_ = false;
	PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR_ = nullptr;
	PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR_ = nullptr;

	static inline const char* safe_str(const char* s) { return s ? s : ""; }

	bool ensureEGL() {
		if (egl_ && egl_->dpy != EGL_NO_DISPLAY) return true;
		auto st = std::make_shared<EglState>();
		st->dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (st->dpy == EGL_NO_DISPLAY) {
			logstream << "drm2egl: eglGetDisplay failed";
			return false;
		}
		EGLint major = 0, minor = 0;
		if (!eglInitialize(st->dpy, &major, &minor)) {
			logstream << "drm2egl: eglInitialize failed";
			return false;
		}
		const char* exts = eglQueryString(st->dpy, EGL_EXTENSIONS);
		have_dma_buf_import_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import");
		have_mods_ = exts && strstr(exts, "EGL_EXT_image_dma_buf_import_modifiers");
		if (!have_dma_buf_import_) {
			logstream << "drm2egl: EGL_EXT_image_dma_buf_import missing, exts=" << safe_str(exts);
			return false;
		}
		if (!p_eglCreateImageKHR_) {
			p_eglCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
			if (!p_eglCreateImageKHR_) {
				// Some stacks expose eglCreateImage (EGL 1.5) but we still prefer KHR; keep null if not present.
				p_eglCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImage");
			}
		}
		if (!p_eglDestroyImageKHR_) {
			p_eglDestroyImageKHR_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
			if (!p_eglDestroyImageKHR_) {
				p_eglDestroyImageKHR_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImage");
			}
		}
		if (!p_eglCreateImageKHR_ || !p_eglDestroyImageKHR_) {
			logstream << "drm2egl: failed to load eglCreateImageKHR/eglDestroyImageKHR";
			return false;
		}
		st->destroy_fn = p_eglDestroyImageKHR_;
		egl_ = std::move(st);
		return true;
	}

	void maybeScanAndEvict(int64_t now_ms) {
		if (last_purge_scan_ms_ != 0 && now_ms - last_purge_scan_ms_ < purge_scan_interval_ms_) return;
		last_purge_scan_ms_ = now_ms;
		if (ttl_ms_ <= 0) return;

		for (auto it = cache_.begin(); it != cache_.end();) {
			const auto &sp = it->second;
			if (!sp) {
				it = cache_.erase(it);
				continue;
			}
			if (now_ms - sp->last_seen_ms > ttl_ms_) {
				it = cache_.erase(it);
				continue;
			}
			++it;
		}
	}

	static bool fstat_identity(int fd, dev_t &out_dev, ino_t &out_ino) {
		struct stat st;
		if (fstat(fd, &st) != 0) return false;
		out_dev = st.st_dev;
		out_ino = st.st_ino;
		return true;
	}

	std::shared_ptr<Entry> getOrCreateEntry(const AVDRMFrameDescriptor *desc, int in_fd_key, int width, int height, int64_t now_ms) {
		if (!ensureEGL()) return nullptr;
		if (!egl_) return nullptr;
		if (!desc || desc->nb_layers < 1 || desc->layers[0].nb_planes < 1) return nullptr;

		const AVDRMLayerDescriptor &layer = desc->layers[0];
		const AVDRMPlaneDescriptor &pl = layer.planes[0];
		const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];

		// Support only single-plane ABGR/ARGB
		if (!(layer.format == DRM_FORMAT_ABGR8888 || layer.format == DRM_FORMAT_ARGB8888)) {
			logstream << "drm2egl: unsupported DRM fourcc=" << layer.format << " (need ABGR8888/ARGB8888)";
			return nullptr;
		}

		// Resolution change purges the cache (per requirement)
		if (last_w_ > 0 && last_h_ > 0 && (width != last_w_ || height != last_h_)) {
			cache_.clear();
		}
		last_w_ = width;
		last_h_ = height;

		// Periodic eviction
		maybeScanAndEvict(now_ms);

		dev_t cur_dev{};
		ino_t cur_ino{};
		bool have_id = fstat_identity(obj.fd, cur_dev, cur_ino);

		auto it = cache_.find(in_fd_key);
		if (it != cache_.end() && it->second) {
			auto &e = it->second;
			bool ok = true;
			if (have_id) ok &= (e->st_dev == cur_dev && e->st_ino == cur_ino);
			ok &= (e->fourcc == layer.format);
			ok &= (e->width == width && e->height == height);
			ok &= (e->pitch == pl.pitch && e->offset == pl.offset);
			const uint64_t mod = obj.format_modifier;
			ok &= (e->modifier == mod);
			if (ok && e->image != EGL_NO_IMAGE_KHR) {
				e->last_seen_ms = now_ms;
				return e;
			}
			// Stale / FD number reused / attributes changed
			cache_.erase(it);
		}

		int dup_fd = dup(obj.fd);
		if (dup_fd < 0) {
			logstream << "drm2egl: dup(fd) failed";
			return nullptr;
		}

		// Create EGLImage from DMA-BUF
		EGLint attrs[64];
		int a = 0;
		attrs[a++] = EGL_WIDTH;  attrs[a++] = (EGLint)width;
		attrs[a++] = EGL_HEIGHT; attrs[a++] = (EGLint)height;
		attrs[a++] = EGL_LINUX_DRM_FOURCC_EXT; attrs[a++] = (EGLint)layer.format;
		attrs[a++] = EGL_DMA_BUF_PLANE0_FD_EXT; attrs[a++] = (EGLint)dup_fd;
		attrs[a++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; attrs[a++] = (EGLint)pl.pitch;
		attrs[a++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; attrs[a++] = (EGLint)pl.offset;
		if (have_mods_ && obj.format_modifier) {
			EGLint mod_lo = (EGLint)(obj.format_modifier & 0xFFFFFFFFu);
			EGLint mod_hi = (EGLint)(obj.format_modifier >> 32);
			attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; attrs[a++] = mod_lo;
			attrs[a++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; attrs[a++] = mod_hi;
		}
		attrs[a++] = EGL_NONE;

		EGLImageKHR img = p_eglCreateImageKHR_(egl_->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer)nullptr, attrs);
		if (img == EGL_NO_IMAGE_KHR) {
			logstream << "drm2egl: eglCreateImageKHR failed for w=" << width << " h=" << height;
			close(dup_fd);
			return nullptr;
		}

		auto e = std::make_shared<Entry>();
		e->egl = egl_;
		e->image = img;
		e->dup_fd = dup_fd;
		e->fourcc = layer.format;
		e->width = width;
		e->height = height;
		e->pitch = pl.pitch;
		e->offset = pl.offset;
		e->modifier = obj.format_modifier;
		if (have_id) {
			e->st_dev = cur_dev;
			e->st_ino = cur_ino;
		}
		e->last_seen_ms = now_ms;
		cache_[in_fd_key] = e;
		return e;
	}

public:
	using NodeSISO::NodeSISO;

	void process() override {
		av::VideoFrame in = this->source_->get();
		if (!in) return;

		if (in.raw()->format != AV_PIX_FMT_DRM_PRIME) {
			// Can't pass through (output type differs); just drop.
			return;
		}

		const AVDRMFrameDescriptor *desc = (const AVDRMFrameDescriptor*)in.raw()->data[0];
		if (!desc) {
			logstream << "drm2egl: missing DRM descriptor";
			return;
		}

		// Keyed by the dma-buf object fd (per requirement).
		if (desc->nb_layers < 1 || desc->layers[0].nb_planes < 1) {
			logstream << "drm2egl: unsupported layer/plane count";
			return;
		}
		const AVDRMPlaneDescriptor &pl = desc->layers[0].planes[0];
		const AVDRMObjectDescriptor &obj = desc->objects[pl.object_index];
		const int in_fd_key = obj.fd;

		const int W = in.width();
		const int H = in.height();
		const int64_t now_ms = wallclock.pts();

		auto entry = getOrCreateEntry(desc, in_fd_key, W, H, now_ms);
		if (!entry) return;

		// Create holder token that keeps Entry alive until consumer releases.
		auto token_sp = std::shared_ptr<EglImagePoolToken>(new EglImagePoolToken{
			.release = [keep = entry]() mutable {
				keep.reset();
			}
		});
		std::shared_ptr<void> holder = token_sp;
		EglImageFrame out(entry->image, W, H, in.pts(), in.timeBase(), holder);
		this->sink_->put(out);
	}

	~DRMPrimeToEglImage() override {
		cache_.clear();
		egl_.reset();
	}

	static std::shared_ptr<DRMPrimeToEglImage> create(NodeCreationInfo &nci) {
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		std::shared_ptr<Edge<av::VideoFrame>> src = edges.find<av::VideoFrame>(params["src"]);
		std::shared_ptr<Edge<EglImageFrame>> dst = edges.find<EglImageFrame>(params["dst"]);
		auto r = std::make_shared<DRMPrimeToEglImage>(
			make_unique<EdgeSource<av::VideoFrame>>(src),
			make_unique<EdgeSink<EglImageFrame>>(dst)
		);
		if (params.count("ttl")) {
			const float ttl_s = params["ttl"].get<float>();
			if (ttl_s < 0) throw Error("drm_prime_to_egl_image: ttl must be >= 0");
			r->ttl_ms_ = (int64_t)(ttl_s * 1000.0f + 0.5f);
		}
		return r;
	}
};

DECLNODE(drm_prime_to_egl_image, DRMPrimeToEglImage);


