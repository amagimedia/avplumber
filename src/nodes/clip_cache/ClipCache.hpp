#pragma once

#include "../../instance_shared.hpp"
#include "../../avutils.hpp"

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace avp::clipcache {

/// Decoded clips held in GPU memory, keyed by the path they were loaded from.
///
/// A media wipe is a short clip replayed on demand. Decoding it again for every
/// take costs a file open, a decoder and a set of node threads at the exact
/// moment the picture has to stay smooth. This store keeps the frames from the
/// first pass so later takes are pure playback.
///
/// Frames are reference-counted AVFrames, so a cached clip costs its decoded
/// size once no matter how many nodes replay it.
struct Clip {
    std::vector<av::VideoFrame> frames;
    size_t bytes = 0;
    bool complete = false;
    int64_t last_used_ms = 0;
};

class ClipCache : public InstanceShared<ClipCache> {
    std::mutex mutex_;
    std::map<std::string, Clip> clips_;
    size_t budget_bytes_ = 1024u * 1024u * 1024u;   // a wipe library, not a video server
    size_t bytes_ = 0;

    static int64_t nowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /// Caller holds the lock. Drops whole clips, least recently used first,
    /// never the one being written.
    void evictFor(size_t incoming, const std::string& keep) {
        while (bytes_ + incoming > budget_bytes_) {
            auto oldest = clips_.end();
            for (auto it = clips_.begin(); it != clips_.end(); ++it) {
                if (it->first == keep || !it->second.complete) continue;
                if (oldest == clips_.end() || it->second.last_used_ms < oldest->second.last_used_ms)
                    oldest = it;
            }
            if (oldest == clips_.end()) return;   // nothing evictable; the caller decides
            bytes_ -= oldest->second.bytes;
            clips_.erase(oldest);
        }
    }

public:
    void setBudget(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        budget_bytes_ = bytes;
    }

    /// The cached frames for *key*, or an empty vector when it is not cached yet.
    std::vector<av::VideoFrame> take(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = clips_.find(key);
        if (it == clips_.end() || !it->second.complete) return {};
        it->second.last_used_ms = nowMs();
        return it->second.frames;
    }

    void begin(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& clip = clips_[key];
        bytes_ -= clip.bytes;
        clip = Clip{};
        clip.last_used_ms = nowMs();
    }

    /// Returns false when the clip would not fit even after eviction; the
    /// caller then keeps passing frames through without caching them.
    bool append(const std::string& key, const av::VideoFrame& frame, size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = clips_.find(key);
        if (it == clips_.end()) return false;
        evictFor(bytes, key);
        if (bytes_ + bytes > budget_bytes_) {
            bytes_ -= it->second.bytes;
            clips_.erase(it);
            return false;
        }
        it->second.frames.push_back(frame);
        it->second.bytes += bytes;
        bytes_ += bytes;
        return true;
    }

    void finish(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = clips_.find(key);
        if (it != clips_.end()) it->second.complete = !it->second.frames.empty();
    }

    Parameters status() {
        std::lock_guard<std::mutex> lock(mutex_);
        Parameters clips = Parameters::array();
        for (const auto& [key, clip] : clips_)
            clips.push_back(Parameters({{"path", key}, {"frames", clip.frames.size()},
                                        {"bytes", clip.bytes}, {"complete", clip.complete}}));
        return Parameters({{"clips", clips}, {"bytes", bytes_}, {"budget_bytes", budget_bytes_}});
    }
};

}  // namespace avp::clipcache
