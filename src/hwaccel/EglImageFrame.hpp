#pragma once

#include <memory>
#include <avcpp/timestamp.h>
#include <avcpp/rational.h>

// Try to include real EGL types if available; otherwise, fall back to opaque typedef
//#if __has_include(<EGL/egl.h>)
#if 0
#include <EGL/egl.h>
#include <EGL/eglext.h>
#else
typedef void* EGLImageKHR;
#endif

// Minimal GPU image carrier for passing EGLImage through queues with proper timing info.
// Copy is shallow; lifecycle is controlled by a holder with custom deleter.
class EglImageFrame {
private:
	EGLImageKHR image_ = nullptr;
	int width_ = 0;
	int height_ = 0;
	av::Timestamp pts_;
	av::Rational tb_;
	std::shared_ptr<void> holder_; // keeps producer-owned resources alive until consumer releases
	std::map<std::string, std::string> metadata_;
public:
	EglImageFrame() = default;
	EglImageFrame(EGLImageKHR image,
	              int width,
	              int height,
	              av::Timestamp pts = {},
	              av::Rational tb = {},
	              std::shared_ptr<void> holder = nullptr)
	    : image_(image), width_(width), height_(height), pts_(pts), tb_(tb), holder_(std::move(holder)) {}

	bool isComplete() const {
		return image_ != nullptr && width_ > 0 && height_ > 0;
	}

	// PTS/timebase API (compatible with av::Frame-like API)
	av::Timestamp pts() const { return pts_; }
	void setPts(av::Timestamp ts) { pts_ = ts; }
	av::Rational timeBase() const { return tb_; }
	void setTimeBase(av::Rational tb) { tb_ = tb; }

	// Geometry
	int width() const { return width_; }
	int height() const { return height_; }

	// Underlying EGL image handle
	EGLImageKHR image() const { return image_; }

	// Holder accessor (for advanced lifetime coordination if needed)
	const std::shared_ptr<void>& holder() const { return holder_; }
	void setHolder(std::shared_ptr<void> h) { holder_ = std::move(h); }

	// Metadata accessor
	void setMetadata(const std::string& key, const std::string& value) { metadata_[key] = value; }
	std::string getMetadata(const std::string& key) const { return metadata_.at(key); }
	bool hasMetadata(const std::string& key) const { return metadata_.find(key) != metadata_.end(); }
	void clearMetadata() { metadata_.clear(); }
	template<typename T>
	void copyMetadata(const T& frm) {
		av::Dictionary dict = av::Dictionary(frm.raw()->metadata, false);
		for (auto it = dict.begin(); it != dict.end(); ++it) {
			metadata_[it->key()] = it->value();
		}
	}
};


