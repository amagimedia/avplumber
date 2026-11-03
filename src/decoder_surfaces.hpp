#pragma once
// Decode surfaces of a fixed-size pool: NVDEC CUarray output with unsafe_output, where every
// decoded frame is the decoder's own surface for as long as any consumer holds it. A decode that
// finds no free surface fails mid-picture ("No decoder surfaces left"), and the frames referencing
// that picture are corrupt until the next keyframe.
#include "instance_shared.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#if defined(HAVE_CUDA) && HAVE_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
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
    int limit_;

public:
    explicit DecoderSurfaceGate(int limit) : limit_(limit) {}
    int limit() const { return limit_; }
    /// A pool smaller than asked for (see reservePool in decoders.cpp) leaves fewer frames to lend.
    /// Call from the thread that decodes.
    void setLimit(int limit) { limit_ = limit; }
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

/// What a decoder does while its pool has no surface for a picture. An episode begins with the
/// first failure and ends when a picture is shown again (shown()); meanwhile each packet waits
/// for the consumers to hold fewer than the gate's limit (or `retry_after`, in case the
/// shortage is the decoder's own references) before it is sent, so the decoder does not run
/// through its stream unpaced while it cannot decode. A decode that reports the shortage
/// (failed(): HEVC returns the hwaccel's ENOMEM) keeps its packet and sends it again, so the
/// picture is not lost; after `give_up` of that the packet is dropped, since a stream whose
/// references alone exceed the pool must keep moving. A decoder that went on without the
/// surface and failed later (lost(): H.264 swallows the ENOMEM and fails on the reference it
/// lost) has lost the picture already. Once a picture is lost, nothing is shown until the next
/// keyframe (showPicture()): what the decoder predicts from a reference it never had is not a
/// picture.
class SurfaceExhaustion {
public:
    using Clock = std::chrono::steady_clock;
    enum class Next { Retry, Drop };
    struct Counters {
        uint64_t exhaustions = 0;       // episodes
        uint64_t retries = 0;           // packets sent while out of surfaces
        uint64_t packets_dropped = 0;   // packets given up on, their picture lost
        uint64_t pictures_lost = 0;     // pictures the decoder lost on its own
        uint64_t pictures_hidden = 0;   // decoded pictures not shown, waiting for a keyframe
        uint64_t recoveries = 0;        // episodes ended by a picture shown
        uint64_t repeats = 0;           // picture periods of ended episodes: about the pictures repeated
    };

private:
    const std::chrono::milliseconds retry_after_, give_up_;
    Counters counters_;
    bool stuck_ = false;
    bool hide_ = false;
    Clock::time_point since_{};   // when the episode, or the packet's wait, began
    Clock::time_point episode_since_{};

    void begin(Clock::time_point now) {
        if (stuck_) return;
        stuck_ = true;
        since_ = episode_since_ = now;
        ++counters_.exhaustions;
    }

public:
    explicit SurfaceExhaustion(std::chrono::milliseconds retry_after = std::chrono::milliseconds(100),
                               std::chrono::milliseconds give_up = std::chrono::milliseconds(1000))
        : retry_after_(retry_after), give_up_(give_up) {}

    /// The packet's decode reported no surface at `now`. Retry: send it again. Drop: it has
    /// waited `give_up`; the next packet gets a wait of its own.
    Next failed(Clock::time_point now) {
        begin(now);
        if (now - since_ < give_up_) return Next::Retry;
        ++counters_.packets_dropped;
        hide_ = true;   // the picture is lost: what follows predicts from it
        since_ = now;
        return Next::Drop;
    }
    /// The decoder lost a picture to the shortage at `now`, the packet used up.
    void lost(Clock::time_point now) {
        begin(now);
        ++counters_.pictures_lost;
        hide_ = true;
    }
    /// Before a packet is sent: how long at most to wait for room first.
    std::chrono::milliseconds retryAfter() const { return retry_after_; }
    void retrying() { ++counters_.retries; }
    /// Picture periods since the episode began, 0 outside one: about how many times the
    /// consumers repeated the source's last picture so far, whether the pictures meanwhile were
    /// late (retried), lost or hidden.
    uint64_t episodePictures(Clock::time_point now, std::chrono::milliseconds period) const {
        if (!stuck_ || period.count() <= 0 || now < episode_since_) return 0;
        return uint64_t((now - episode_since_) / period);
    }
    /// A picture is on air again at `now`: the episode is over. Returns its episodePictures(),
    /// which the counters add to `repeats`.
    uint64_t shown(Clock::time_point now, std::chrono::milliseconds period) {
        if (!stuck_) return 0;
        const uint64_t pictures = episodePictures(now, period);
        stuck_ = false;
        ++counters_.recoveries;
        counters_.repeats += pictures;
        return pictures;
    }
    /// Whether a decoded picture is shown: false until a keyframe follows a lost picture.
    bool showPicture(bool keyframe) {
        if (!hide_) return true;
        if (keyframe) {
            hide_ = false;
            return true;
        }
        ++counters_.pictures_hidden;
        return false;
    }
    bool stuck() const { return stuck_; }
    bool hiding() const { return hide_; }
    Clock::time_point since() const { return since_; }
    Clock::time_point episodeSince() const { return episode_since_; }
    const Counters &counters() const { return counters_; }
};

/// The device memory of an instance's fixed-pool decoders against one budget, so a show that
/// needs more fails at decoder open, by name, instead of running out of device memory later.
/// Also the show's surface-exhaustion totals, which every decoder reports here and any node can
/// publish.
class DecoderPoolLedger : public InstanceShared<DecoderPoolLedger> {
    std::mutex mutex_;
    std::unordered_map<const void *, int64_t> pools_;
    uint64_t exhaustions_ = 0;
    uint64_t repeats_ = 0;
    int exhausted_now_ = 0;
    int degraded_now_ = 0;

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
    int64_t total() {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t total = 0;
        for (const auto &[owner, size] : pools_) total += size;
        return total;
    }

    /// A decoder ran out of surfaces, or has them again.
    void exhausted(bool now) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (now) {
            ++exhaustions_;
            ++exhausted_now_;
        } else if (exhausted_now_ > 0) {
            --exhausted_now_;
        }
    }
    /// An episode ended, or its decoder closed during it: about `pictures` were repeated.
    void repeated(uint64_t pictures) {
        std::lock_guard<std::mutex> lock(mutex_);
        repeats_ += pictures;
    }
    /// Pictures the show's sources repeated, about, while their decoders had no surface.
    uint64_t repeats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return repeats_;
    }
    /// Episodes so far, and the decoders without surfaces right now.
    std::pair<uint64_t, int> exhaustion() {
        std::lock_guard<std::mutex> lock(mutex_);
        return {exhaustions_, exhausted_now_};
    }
    /// A decoder stopped giving pictures (packets failing in a row, no pool within the budget),
    /// or gives them again.
    void degraded(bool now) {
        std::lock_guard<std::mutex> lock(mutex_);
        degraded_now_ += now ? 1 : (degraded_now_ > 0 ? -1 : 0);
    }
    int degradedNow() {
        std::lock_guard<std::mutex> lock(mutex_);
        return degraded_now_;
    }
};

/// Decoders that left the show, kept alive instead of destroyed: destroying one unloads its
/// module and destroys every surface of its pool under the driver lock, which stalls every
/// thread using the device for tens of milliseconds, in the middle of a show. One decoded frame
/// of the dead decoder holds its NVDEC session and its pool (surface index, decoder reference
/// and frames context travel with the frame); nothing else of it is held, so every other
/// surface of that pool is free, and a frame of a running decoder is never parked: only a
/// decoder's own close parks, never a consumer letting go of a frame.
///
/// A new source that fits a parked decoder takes it over (take(); hwaccel_context in nvdec.c),
/// so changing sources of one format creates and destroys nothing, and a show that shrinks
/// parks no more decoders than it once ran: the device held them all then. Only decoders beyond
/// that, left by sources a decoder of another format replaced, cost memory the show never used,
/// and `limit` bounds them (a parked decoder holds more device memory than its pool: its NVDEC
/// session and textures). Past it the oldest is freed: a free still stalls every thread for the
/// session's module unload, which is why a shrinking show frees nothing. The parked pools stay in
/// the ledger, under their entry, until they are taken or freed: a live decoder whose pool does
/// not fit the budget frees parked ones first, oldest first. Decoders are always freed one at a
/// time with a rest between them, and between the surfaces of a pool, so that no change of sources
/// holds the driver lock end to end. freeOne() does this when the operator chooses.
class DecoderParking : public InstanceShared<DecoderParking> {
public:
    struct Geometry {
        int codec = 0;   // AVCodecID
        int width = 0, height = 0;
        int sw_format = -1;   // AVPixelFormat
        int surfaces = 0;
        /// Whether a decoder of this shape can decode a stream needing `need`.
        bool fits(const Geometry &need) const {
            return codec == need.codec && width == need.width && height == need.height &&
                   sw_format == need.sw_format && surfaces >= need.surfaces;
        }
    };

private:
    struct Entry {
        AVFrame *frame = nullptr;
        int64_t bytes = 0;
        Geometry geometry;
        std::string name;
        std::shared_ptr<DecoderPoolLedger> ledger;   // the entry's pool is reserved under &*this
    };
    std::mutex mutex_;
    std::mutex freeing_;   // held for each free and the rest after it
    static constexpr auto kRestAfterFree = std::chrono::milliseconds(250);
    static constexpr int kRestAfterSurfaceUs = 20000;   // most of a 25 to 60 fps tick stays free
    std::deque<std::unique_ptr<Entry>> parked_;   // oldest first
    uint64_t parked_total_ = 0, freed_ = 0, reused_ = 0;
    int live_ = 0, peak_live_ = 0;   // decoders with a session now, and the most there were at once

    // Destroys the decoder the entry held; called without the lock, as the destruction is slow.
    static void destroy(std::unique_ptr<Entry> entry) {
#ifdef AV_CUDA_FRAMES_HAS_FREE_PACE   // the FFmpeg patch series
        if (entry->frame && entry->frame->format == AV_PIX_FMT_CUARRAY && entry->frame->hw_frames_ctx) {
            auto *frames = reinterpret_cast<AVHWFramesContext *>(entry->frame->hw_frames_ctx->data);
            static_cast<AVCUDAFramesContext *>(frames->hwctx)->cuarray_free_pace_us = kRestAfterSurfaceUs;
        }
#endif
        av_frame_free(&entry->frame);
        if (entry->ledger) entry->ledger->release(entry.get());
    }
    void freeOutOfTurn(std::unique_ptr<Entry> entry) {
        std::lock_guard<std::mutex> lock(freeing_);
        destroy(std::move(entry));
        std::this_thread::sleep_for(kRestAfterFree);
    }
    std::unique_ptr<Entry> popOldest() {
        if (parked_.empty()) return nullptr;
        auto entry = std::move(parked_.front());
        parked_.pop_front();
        ++freed_;
        return entry;
    }

public:
    ~DecoderParking() {
        for (auto &entry : parked_) destroy(std::move(entry));
    }

    /// Parks the decoder `frame` belongs to (a reference is taken; `frame` stays the caller's),
    /// accounting `bytes` for its pool in `ledger` until it is taken or freed, and frees the
    /// oldest parked decoders beyond `limit`. Returns how many were freed for that.
    int park(const AVFrame *frame, const Geometry &geometry, int64_t bytes,
             std::shared_ptr<DecoderPoolLedger> ledger, std::string name, int limit) {
        if (!frame || limit <= 0) return 0;
        auto entry = std::make_unique<Entry>();
        entry->frame = av_frame_clone(frame);
        if (!entry->frame) return 0;
        entry->bytes = bytes;
        entry->geometry = geometry;
        entry->name = std::move(name);
        if (ledger && bytes > 0) {
            bool accepted = false;
            ledger->reserve(entry.get(), bytes, 0, accepted);
            entry->ledger = std::move(ledger);
        }
        std::deque<std::unique_ptr<Entry>> evicted;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            parked_.push_back(std::move(entry));
            ++parked_total_;
            while (int(parked_.size()) > limit && live_ + int(parked_.size()) > peak_live_ + limit)
                evicted.push_back(popOldest());
        }
        for (auto &old : evicted) freeOutOfTurn(std::move(old));
        return int(evicted.size());
    }

    /// A decoder of the instance opened its session, or is closing it (before it parks).
    void opened() {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_live_ = std::max(peak_live_, ++live_);
    }
    void closed() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (live_ > 0) --live_;
    }

    /// Frees the oldest parked decoder: true when there was one.
    bool freeOne() {
        std::unique_ptr<Entry> entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entry = popOldest();
        }
        if (!entry) return false;
        freeOutOfTurn(std::move(entry));
        return true;
    }

    /// Frees parked decoders, oldest first, until `bytes` more fit `budget` in `ledger` or none
    /// is left. Returns how many were freed.
    int makeRoom(DecoderPoolLedger &ledger, int64_t bytes, int64_t budget) {
        int freed = 0;
        while (budget && ledger.total() + bytes > budget && freeOne()) ++freed;
        return freed;
    }

    /// The smallest parked decoder that can decode a stream needing `need`, if any: the frame
    /// that holds it, for hwaccel_context. Its pool leaves the ledger; `bytes` is what it holds.
    AVFrame *take(const Geometry &need, int64_t &bytes) {
        std::unique_ptr<Entry> entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto best = parked_.end();
            for (auto it = parked_.begin(); it != parked_.end(); ++it)
                if ((*it)->geometry.fits(need) &&
                    (best == parked_.end() || (*it)->geometry.surfaces < (*best)->geometry.surfaces))
                    best = it;
            if (best == parked_.end()) return nullptr;
            entry = std::move(*best);
            parked_.erase(best);
            ++reused_;
        }
        AVFrame *frame = entry->frame;
        entry->frame = nullptr;
        bytes = entry->bytes;
        if (entry->ledger) entry->ledger->release(entry.get());
        return frame;
    }

    struct Status {
        int parked = 0;
        int64_t bytes = 0;
        uint64_t parked_total = 0, freed = 0, reused = 0;
    };
    Status status() {
        std::lock_guard<std::mutex> lock(mutex_);
        Status s;
        s.parked = int(parked_.size());
        for (const auto &entry : parked_) s.bytes += entry->bytes;
        s.parked_total = parked_total_;
        s.freed = freed_;
        s.reused = reused_;
        return s;
    }
};

}
