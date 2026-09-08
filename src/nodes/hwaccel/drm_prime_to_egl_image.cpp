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

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

enum class CacheMode {
	Reuse,
	Off,
};

struct CacheKey {
	dev_t st_dev{};
	ino_t st_ino{};
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t stride = 0;
	uint32_t fourcc = 0;
	uint64_t modifier = 0;
	uint64_t offset = 0;

	bool operator==(const CacheKey &other) const {
		return st_dev == other.st_dev && st_ino == other.st_ino &&
		       width == other.width && height == other.height &&
		       stride == other.stride && fourcc == other.fourcc &&
		       modifier == other.modifier && offset == other.offset;
	}
};

} // namespace

class DRMPrimeToEglImage : public NodeSISO<av::VideoFrame, EglImageFrame> {
protected:
	struct EglState {
		EGLDisplay dpy = EGL_NO_DISPLAY;
		PFNEGLDESTROYIMAGEKHRPROC destroy_image = nullptr;

		~EglState() {
			// EGLDisplay is process-global on the supported drivers. Other nodes may
			// still own images or contexts for it, so do not call eglTerminate here.
			dpy = EGL_NO_DISPLAY;
		}
	};

	struct Entry {
		std::shared_ptr<EglState> egl;
		EGLImageKHR image = EGL_NO_IMAGE_KHR;
		int dup_fd = -1;
		CacheKey key;
		int64_t last_seen_ms = 0;

		~Entry() {
			if (image != EGL_NO_IMAGE_KHR && egl && egl->dpy != EGL_NO_DISPLAY && egl->destroy_image)
			egl->destroy_image(egl->dpy, image);
			if (dup_fd >= 0)
				close(dup_fd);
		}
	};

	std::vector<std::shared_ptr<Entry>> cache_;
	CacheMode cache_mode_ = CacheMode::Reuse;
	int64_t ttl_ms_ = 3000;
	size_t max_cache_entries_ = 64;
	int64_t last_purge_scan_ms_ = 0;
	int64_t purge_scan_interval_ms_ = 1000;
	int last_w_ = 0;
	int last_h_ = 0;

	std::shared_ptr<EglState> egl_;
	bool have_modifiers_ = false;
	PFNEGLCREATEIMAGEKHRPROC create_image_ = nullptr;
	PFNEGLDESTROYIMAGEKHRPROC destroy_image_ = nullptr;

	uint64_t frames_ = 0;
	uint64_t cache_hits_ = 0;
	uint64_t fresh_imports_ = 0;
	uint64_t evictions_ = 0;
	int debug_log_every_n_ = 0;

	static const char *safeString(const char *value) { return value ? value : ""; }

	static bool extensionSupported(EGLDisplay display, const char *extension) {
		const char *extensions = eglQueryString(display, EGL_EXTENSIONS);
		if (!extensions || !extension || !extension[0])
			return false;
		const size_t len = std::strlen(extension);
		const char *current = extensions;
		while ((current = std::strstr(current, extension)) != nullptr) {
			const bool starts = current == extensions || current[-1] == ' ';
			const bool ends = current[len] == '\0' || current[len] == ' ';
			if (starts && ends)
				return true;
			current += len;
		}
		return false;
	}

	bool ensureEGL() {
		if (egl_ && egl_->dpy != EGL_NO_DISPLAY)
			return true;

		auto state = std::make_shared<EglState>();
		state->dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
		if (state->dpy == EGL_NO_DISPLAY) {
			logstream << "drm2egl: eglGetDisplay failed";
			return false;
		}

		EGLint major = 0;
		EGLint minor = 0;
		if (!eglInitialize(state->dpy, &major, &minor)) {
			logstream << "drm2egl: eglInitialize failed";
			return false;
		}

		if (!extensionSupported(state->dpy, "EGL_EXT_image_dma_buf_import")) {
			logstream << "drm2egl: EGL_EXT_image_dma_buf_import missing, exts="
			          << safeString(eglQueryString(state->dpy, EGL_EXTENSIONS));
			return false;
		}
		have_modifiers_ = extensionSupported(state->dpy, "EGL_EXT_image_dma_buf_import_modifiers");

		create_image_ = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
		if (!create_image_)
			create_image_ = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImage"));
		destroy_image_ = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
		if (!destroy_image_)
			destroy_image_ = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImage"));
		if (!create_image_ || !destroy_image_) {
			logstream << "drm2egl: failed to load eglCreateImage/eglDestroyImage";
			return false;
		}

		state->destroy_image = destroy_image_;
		egl_ = std::move(state);
		return true;
	}

	void maybeEvict(int64_t now_ms) {
		if (last_purge_scan_ms_ && now_ms - last_purge_scan_ms_ < purge_scan_interval_ms_)
			return;
		last_purge_scan_ms_ = now_ms;
		if (ttl_ms_ <= 0)
			return;
		const auto before = cache_.size();
		cache_.erase(std::remove_if(cache_.begin(), cache_.end(), [=](const auto &entry) {
			return !entry || now_ms - entry->last_seen_ms >= ttl_ms_;
		}), cache_.end());
		evictions_ += before - cache_.size();
	}

	std::shared_ptr<Entry> createAndInsert(const CacheKey &key, int fd, int64_t now_ms) {
		const int dup_fd = dup(fd);
		if (dup_fd < 0) {
			logstream << "drm2egl: dup(fd) failed: " << std::strerror(errno);
			return nullptr;
		}

		EGLint attrs[32];
		int index = 0;
		attrs[index++] = EGL_WIDTH;
		attrs[index++] = static_cast<EGLint>(key.width);
		attrs[index++] = EGL_HEIGHT;
		attrs[index++] = static_cast<EGLint>(key.height);
		attrs[index++] = EGL_LINUX_DRM_FOURCC_EXT;
		attrs[index++] = static_cast<EGLint>(key.fourcc);
		attrs[index++] = EGL_DMA_BUF_PLANE0_FD_EXT;
		attrs[index++] = dup_fd;
		attrs[index++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT;
		attrs[index++] = static_cast<EGLint>(key.offset);
		attrs[index++] = EGL_DMA_BUF_PLANE0_PITCH_EXT;
		attrs[index++] = static_cast<EGLint>(key.stride);
		if (have_modifiers_ && key.modifier) {
			attrs[index++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
			attrs[index++] = static_cast<EGLint>(key.modifier & 0xffffffffu);
			attrs[index++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
			attrs[index++] = static_cast<EGLint>(key.modifier >> 32);
		}
		attrs[index++] = EGL_NONE;

		EGLImageKHR image = create_image_(egl_->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
		                                  static_cast<EGLClientBuffer>(nullptr), attrs);
		if (image == EGL_NO_IMAGE_KHR) {
			logstream << "drm2egl: eglCreateImage failed for " << key.width << "x" << key.height
			          << ", EGL error=" << eglGetError();
			close(dup_fd);
			return nullptr;
		}

		auto entry = std::make_shared<Entry>();
		entry->egl = egl_;
		entry->image = image;
		entry->dup_fd = dup_fd;
		entry->key = key;
		entry->last_seen_ms = now_ms;

		if (max_cache_entries_ > 0 && cache_.size() >= max_cache_entries_) {
			auto oldest = std::min_element(cache_.begin(), cache_.end(), [](const auto &a, const auto &b) {
				return a->last_seen_ms < b->last_seen_ms;
			});
			cache_.erase(oldest);
			++evictions_;
		}
		cache_.push_back(entry);
		++fresh_imports_;
		return entry;
	}

	std::shared_ptr<Entry> getOrCreateEntry(const AVDRMFrameDescriptor *desc, int width, int height,
	                                        int64_t now_ms) {
		if (!ensureEGL() || !desc || desc->nb_layers != 1 || desc->layers[0].nb_planes != 1)
			return nullptr;

		const AVDRMLayerDescriptor &layer = desc->layers[0];
		const AVDRMPlaneDescriptor &plane = layer.planes[0];
		if (plane.object_index < 0 || plane.object_index >= desc->nb_objects)
			return nullptr;
		const AVDRMObjectDescriptor &object = desc->objects[plane.object_index];
		if (layer.format != DRM_FORMAT_ABGR8888 && layer.format != DRM_FORMAT_ARGB8888) {
			logstream << "drm2egl: unsupported DRM fourcc=" << layer.format
			          << " (need ABGR8888/ARGB8888)";
			return nullptr;
		}

		struct stat statbuf{};
		if (fstat(object.fd, &statbuf) != 0) {
			logstream << "drm2egl: fstat(fd) failed: " << std::strerror(errno);
			return nullptr;
		}
		CacheKey key{
			statbuf.st_dev,
			statbuf.st_ino,
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height),
			static_cast<uint32_t>(plane.pitch),
			layer.format,
			object.format_modifier,
			static_cast<uint64_t>(plane.offset),
		};

		if (last_w_ && (last_w_ != width || last_h_ != height)) {
			evictions_ += cache_.size();
			cache_.clear();
		}
		last_w_ = width;
		last_h_ = height;
		maybeEvict(now_ms);

		if (cache_mode_ != CacheMode::Off) {
			for (const auto &entry : cache_) {
				if (!entry || !(entry->key == key))
					continue;
				entry->last_seen_ms = now_ms;
				++cache_hits_;
				return entry;
			}
		}

		return createAndInsert(key, object.fd, now_ms);
	}

public:
	using NodeSISO<av::VideoFrame, EglImageFrame>::NodeSISO;

	void process() override {
		av::VideoFrame input = this->source_->get();
		if (!input)
			return;
		if (input.raw()->format != AV_PIX_FMT_DRM_PRIME)
			return;

		const auto *desc = reinterpret_cast<const AVDRMFrameDescriptor *>(input.raw()->data[0]);
		if (!desc) {
			logstream << "drm2egl: missing DRM descriptor";
			return;
		}

		auto entry = getOrCreateEntry(desc, input.width(), input.height(), wallclock.pts());
		if (!entry)
			return;

		auto token = std::shared_ptr<EglImagePoolToken>(new EglImagePoolToken{
			.release = [keep = entry]() mutable { keep.reset(); },
		});
		auto frame_lifetime = std::make_shared<av::VideoFrame>(input);
		EglImageFrame output(entry->image, input.width(), input.height(), input.pts(),
		                     input.timeBase(), token, frame_lifetime);
		output.copyMetadata(input);
		this->sink_->put(output);

		++frames_;
		if (debug_log_every_n_ > 0 && frames_ % static_cast<uint64_t>(debug_log_every_n_) == 0) {
			logstream << "drm2egl: frames=" << frames_ << " hits=" << cache_hits_
			          << " imports=" << fresh_imports_ << " evictions=" << evictions_
			          << " cache=" << cache_.size();
		}
	}

	~DRMPrimeToEglImage() override {
		cache_.clear();
		egl_.reset();
	}

	static std::shared_ptr<DRMPrimeToEglImage> create(NodeCreationInfo &nci) {
		EdgeManager &edges = nci.edges;
		const Parameters &params = nci.params;
		auto src = edges.find<av::VideoFrame>(params["src"]);
		auto dst = edges.find<EglImageFrame>(params["dst"]);
		auto node = std::make_shared<DRMPrimeToEglImage>(
			make_unique<EdgeSource<av::VideoFrame>>(src),
			make_unique<EdgeSink<EglImageFrame>>(dst));

		const std::string mode = params.value("cache_mode", std::string("reuse"));
		if (mode == "reuse")
			node->cache_mode_ = CacheMode::Reuse;
		else if (mode == "off")
			node->cache_mode_ = CacheMode::Off;
		else
			throw Error("drm_prime_to_egl_image: cache_mode must be reuse or off");

		const double ttl_seconds = params.value("ttl", 3.0);
		if (ttl_seconds < 0)
			throw Error("drm_prime_to_egl_image: ttl must be >= 0");
		node->ttl_ms_ = static_cast<int64_t>(ttl_seconds * 1000.0 + 0.5);
		const int max_entries = params.value("max_cache_entries", 64);
		if (max_entries < 1)
			throw Error("drm_prime_to_egl_image: max_cache_entries must be >= 1");
		node->max_cache_entries_ = static_cast<size_t>(max_entries);
		node->debug_log_every_n_ = params.value("debug_log_every_n", 0);
		if (node->debug_log_every_n_ < 0)
			throw Error("drm_prime_to_egl_image: debug_log_every_n must be >= 0");
		return node;
	}
};

DECLNODE(drm_prime_to_egl_image, DRMPrimeToEglImage);
