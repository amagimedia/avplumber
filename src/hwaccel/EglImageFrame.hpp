#pragma once

#include <map>
#include <memory>
#include <string>
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
	std::shared_ptr<void> allocation_holder_;
	std::shared_ptr<void> frame_holder_;
	std::map<std::string, std::string> metadata_;
public:
	EglImageFrame() = default;
	EglImageFrame(EGLImageKHR image,
	              int width,
	              int height,
	              av::Timestamp pts = {},
	              av::Rational tb = {},
	              std::shared_ptr<void> allocation_holder = nullptr,
	              std::shared_ptr<void> frame_holder = nullptr)
	    : image_(image), width_(width), height_(height), pts_(pts), tb_(tb),
	      allocation_holder_(std::move(allocation_holder)), frame_holder_(std::move(frame_holder)) {}

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

	// Allocation lifetime may be cached across frames. Per-frame lifetime must
	// end after the last consumer read so a producer can safely reuse its buffer.
	const std::shared_ptr<void>& holder() const { return allocation_holder_; }
	void setHolder(std::shared_ptr<void> h) { allocation_holder_ = std::move(h); }
	const std::shared_ptr<void>& frameHolder() const { return frame_holder_; }
	void setFrameHolder(std::shared_ptr<void> h) { frame_holder_ = std::move(h); }

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
