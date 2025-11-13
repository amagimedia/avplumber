#pragma once
#include <functional>
#include <memory>
#include <atomic>
#
// Lightweight token passed along with EglImageFrame so that consumers
// can return the underlying image back to its pool by calling release().
struct EglImagePoolToken {
	std::function<void()> release;
	std::atomic<bool> released{false};
	// Ensure the release callback is invoked at most once
	inline void releaseOnce() {
		bool expected = false;
		if (released.compare_exchange_strong(expected, true)) {
			if (release) release();
		}
	}
	~EglImagePoolToken() {
		// Auto-release if not already released
		releaseOnce();
	}
};

// Wrapper used when passing to OBS as hw_opaque to hold ownership
// of the token via shared_ptr until OBS calls free_buffer.
struct EglImageOpaque {
	std::shared_ptr<void> holder;           // shared_ptr owning EglImagePoolToken
	EglImagePoolToken* token = nullptr;     // raw view for convenience/assertion
};


