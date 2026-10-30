#pragma once
// Decode surfaces of a fixed-size pool: NVDEC CUarray output with unsafe_output, where every
// decoded frame is the decoder's own surface for as long as any consumer holds it. A decode that
// finds no free surface fails mid-picture ("No decoder surfaces left"), and the frames referencing
// that picture are corrupt until the next keyframe.
#include "instance_shared.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

namespace avp {

/// Counts the decoded frames consumers still hold. Holding the next packet back while
/// limit() frames are out keeps a free surface for every decode, whatever the codec's reference
/// count: the pool is FFmpeg's references and reorder delay plus two plus extra_hw_frames, and
/// limit() is extra_hw_frames plus one.
class DecoderSurfaceGate {
    struct State {
        std::mutex mutex;
        std::condition_variable released;
        int held = 0;
    };
    // Shared with every tracked frame, which may outlive the decoder.
    std::shared_ptr<State> state_ = std::make_shared<State>();
    const int limit_;

public:
    explicit DecoderSurfaceGate(int limit) : limit_(limit) {}
    int limit() const { return limit_; }
    int held() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->held;
    }

    /// Counts `frame` until its last reference is released: a buffer added to a free buf[] slot
    /// travels with the frame's other buffers through av_frame_ref() and av_frame_unref().
    /// False, counting nothing, when the frame has no free slot.
    bool track(AVFrame *frame) {
        int slot = 0;
        while (slot < AV_NUM_DATA_POINTERS && frame->buf[slot]) ++slot;
        if (slot == AV_NUM_DATA_POINTERS) return false;
        auto *kept = new std::shared_ptr<State>(state_);
        AVBufferRef *ref = av_buffer_create(reinterpret_cast<uint8_t *>(kept), sizeof(*kept),
                                            &DecoderSurfaceGate::released, nullptr, 0);
        if (!ref) {
            delete kept;
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->held;
        }
        frame->buf[slot] = ref;
        return true;
    }

    /// Waits up to `timeout` until fewer than limit() frames are held: true when the decoder
    /// may take its next packet.
    bool wait(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(state_->mutex);
        return state_->released.wait_for(lock, timeout, [&] { return state_->held < limit_; });
    }

private:
    static void released(void *, uint8_t *data) {
        auto *kept = reinterpret_cast<std::shared_ptr<State> *>(data);
        {
            std::lock_guard<std::mutex> lock((*kept)->mutex);
            --(*kept)->held;
        }
        (*kept)->released.notify_all();
        delete kept;
    }
};

/// The device memory of an instance's fixed-pool decoders against one budget, so a show that
/// needs more fails at decoder open, by name, instead of running out of device memory later.
class DecoderPoolLedger : public InstanceShared<DecoderPoolLedger> {
    std::mutex mutex_;
    std::unordered_map<const void *, int64_t> pools_;

public:
    /// Records `owner`'s pool of `bytes`, replacing its previous one, unless all pools together
    /// would exceed `budget` bytes (0: no budget). Returns the total, including this pool either way.
    int64_t reserve(const void *owner, int64_t bytes, int64_t budget, bool &accepted) {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t total = bytes;
        for (const auto &[other, size] : pools_)
            if (other != owner) total += size;
        accepted = !budget || total <= budget;
        if (accepted) pools_[owner] = bytes;
        return total;
    }
    void release(const void *owner) {
        std::lock_guard<std::mutex> lock(mutex_);
        pools_.erase(owner);
    }
};

}
