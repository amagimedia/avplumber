#include "../node_common.hpp"
#include "ClipCache.hpp"
#include "../../MultiEventWait.hpp"
#include "../../mixer/primitives/MonotonicClock.hpp"
#include "../../mixer/primitives/TickGrid.hpp"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
}

#include <chrono>
#include <mutex>
#include <thread>

/// Replay a short clip from GPU memory, filling the cache on a loading pass.
///
/// The node runs for the life of the graph; nothing is created, started or
/// stopped for a take. "load" <path> reads that clip from the decode chain
/// upstream into the cache; "play" <path> replays a cached clip from its first
/// frame; "stop" ends a replay; "forget" <path> drops a clip, cached or loading.
/// Between clips the thread waits on its events.
///
/// A load ends when the decode chain's end-of-stream marker arrives (realtime
/// passes it on with forward_eof), and on nothing else: a producer stalled for
/// seconds looks exactly like the end of a clip to anyone timing the silence.
/// The marker says the chain ended, not that every frame got here, so whoever
/// asked for the load compares the frame count with the decoder's.
///
/// A replay is stamped on the output tick grid from the tick after it was
/// armed and each frame is pushed when its tick begins, so a clocked compositor
/// downstream matches frame k to tick T0+k with its whole deadline to draw it,
/// and rejects anything stamped before the arm with one reset.
static constexpr int IDLE_WAKE_MS = 200;

class ClipCacheNode : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                      public IInputsObjects, public IReturnsObjects,
                      public IFrameRateSource, public ITimeBaseSource {
protected:
    std::shared_ptr<avp::clipcache::ClipCache> cache_;
    av::Rational fps_{60, 1};
    avp::mixer::TickGrid grid_{av::Rational(60, 1)};
    Event wake_;                               // a play or load request
    std::unique_ptr<MultiEventWait> events_;   // a frame from upstream or wake_

    // Both the node thread and setObject() touch the state below. Holding the
    // mutex across a frame push makes a stop and a push exclusive: no frame is
    // pushed after stop returns, so the downstream reset that follows a stop
    // sees only frames stamped before it.
    std::mutex mutex_;
    std::string loading_key_;                  // clip being read from upstream, empty when none
    size_t loaded_frames_ = 0;
    int logged_first_ = 0;
    std::string key_;                          // clip being replayed
    std::vector<av::VideoFrame> playback_;
    size_t index_ = 0;
    int64_t start_tick_ = 0;
    bool playing_ = false;

    static size_t frameBytes(const av::VideoFrame& frame) {
        const AVFrame* raw = frame.raw();
        if (!raw) return 0;
        if (raw->hw_frames_ctx && raw->hw_frames_ctx->data) {
            auto* ctx = (AVHWFramesContext*)raw->hw_frames_ctx->data;
            const int size = av_image_get_buffer_size(ctx->sw_format, ctx->width, ctx->height, 1);
            return size > 0 ? (size_t)size : 0;
        }
        const int size = av_image_get_buffer_size((AVPixelFormat)raw->format, raw->width, raw->height, 1);
        return size > 0 ? (size_t)size : 0;
    }

    /// Nanoseconds from the clip's first frame to frame *index*, from the
    /// timestamps the frames were decoded with.
    int64_t elapsedNs(size_t index) {
        if (index == 0 || index >= playback_.size()) return 0;
        const av::Timestamp first = playback_.front().pts(), current = playback_[index].pts();
        if (!first.isValid() || !current.isValid())
            return grid_.time((int64_t)index);   // no timestamps: fall back to the output grid
        return rescaleTS(addTS(current, negateTS(first)), {1, 1000000000}).timestamp();
    }

    /// Caller holds mutex_.
    void finishLoad() {
        cache_->finish(loading_key_);
        logstream << "clip_cache: cached " << loaded_frames_ << " frame(s) of " << loading_key_;
        loading_key_.clear();
        loaded_frames_ = 0;
    }

    void load(const std::string& key) {
        if (!cache_->take(key).empty()) {
            logstream << "clip_cache: " << key << " is already cached";
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        loading_key_ = key;
        loaded_frames_ = 0;
        logged_first_ = 0;
        cache_->begin(key);
        logstream << "clip_cache: loading " << key;
        wake_.signal();
    }

    /// True when a replay frame was pushed or the thread should come back at once.
    bool replay() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!playing_) return false;
        if (index_ >= playback_.size()) {
            playing_ = false;
            playback_.clear();
            logstream << "clip_cache: finished replaying " << key_;
            return false;
        }
        const int64_t tick = start_tick_ + grid_.nearestIndex(elapsedNs(index_));
        const int64_t due = grid_.time(tick), now = avp::mixer::monotonicNs();
        if (now < due) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(due - now, grid_.time(1))));
            return true;
        }
        av::VideoFrame out = playback_[index_];
        out.setTimeBase(timeBase());
        out.setPts(av::Timestamp(tick, timeBase()));
        out.setComplete(true);
        // Never blocks under the mutex: a full queue means the consumer is not
        // keeping up, and one frame late is no better than one frame less.
        this->sink_->put(out, true);
        ++index_;
        return true;
    }

public:
    using NodeSISO::NodeSISO;

    av::Rational frameRate() override { return fps_; }
    av::Rational timeBase() override { return {fps_.getDenominator(), fps_.getNumerator()}; }

    void process() override {
        if (replay()) return;

        // The frame leaves the edge under the mutex, so it is either dropped before a
        // load is accepted or belongs to that load, never taken before and stored after.
        std::unique_lock<std::mutex> lock(mutex_);
        av::VideoFrame in;
        if (!this->source_->tryGet(in, 0)) {
            lock.unlock();
            events_->wait(IDLE_WAKE_MS);
            return;
        }
        if (loading_key_.empty()) return;   // upstream only runs to fill the cache; nothing here is on air
        if (isEofMarker(in)) {
            finishLoad();   // without a single picture the clip stays incomplete
            return;
        }
        // A picture is anything with real dimensions and a timestamp; the
        // completeness flag is not reliable across every producer.
        const bool picture = in.raw() && in.width() > 0 && in.height() > 0 && in.pts().isValid();
        if (loaded_frames_ == 0 && logged_first_ < 3) {
            ++logged_first_;
            logstream << "clip_cache: input frame " << in.width() << "x" << in.height()
                      << " pts_valid=" << in.pts().isValid() << " complete=" << in.isComplete()
                      << " -> " << (picture ? "picture" : "ignored");
        }
        if (!picture) return;
        if (!cache_->append(loading_key_, in, frameBytes(in))) {
            logstream << "clip_cache: " << loading_key_ << " does not fit the budget; not cached";
            loading_key_.clear();
            loaded_frames_ = 0;
            return;
        }
        ++loaded_frames_;
        if (loaded_frames_ % 16 == 0)
            logstream << "clip_cache: " << loaded_frames_ << " frame(s) so far of " << loading_key_;
    }

    /// A clip still loading when the node goes away never reached its end: it is
    /// dropped, not completed.
    ~ClipCacheNode() override {
        if (!loading_key_.empty() && cache_) {
            logstream << "clip_cache: torn down while loading " << loading_key_ << " after "
                      << loaded_frames_ << " frame(s); dropping the partial clip";
            cache_->forget(loading_key_);
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "play") {
            const std::string clip = value.get<std::string>();
            std::vector<av::VideoFrame> frames = cache_->take(clip);
            if (frames.empty()) throw Error("clip_cache: not cached: " + clip);
            std::lock_guard<std::mutex> lock(mutex_);
            playback_ = std::move(frames);
            key_ = clip;
            index_ = 0;
            start_tick_ = grid_.atOrBefore(avp::mixer::monotonicNs()) + 1;   // the first tick still ahead
            playing_ = true;
            wake_.signal();
            logstream << "clip_cache: replaying " << playback_.size() << " cached frame(s) of " << clip
                      << " from tick " << start_tick_;
        } else if (key == "stop") {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!playing_) return;
            playing_ = false;
            playback_.clear();
            logstream << "clip_cache: stopped replaying " << key_ << " at frame " << index_;
        } else if (key == "load") {
            load(value.get<std::string>());
        } else if (key == "forget") {
            const std::string clip = value.get<std::string>();
            std::lock_guard<std::mutex> lock(mutex_);
            if (loading_key_ == clip) {
                loading_key_.clear();
                loaded_frames_ = 0;
            }
            cache_->forget(clip);
            logstream << "clip_cache: forgot " << clip;
        } else {
            throw Error("clip_cache: unknown object key: " + key);
        }
    }

    /// "status": the cache's, plus "loading", the clip being read ("" when none: a load
    /// that ended shows there and in its clip's "complete"), "playing" and "start_tick",
    /// the output tick of the last replay's first frame, which is where that replay's
    /// output begins.
    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("clip_cache: unknown object key: " + key);
        // Locked before the cache is read: finishLoad() completes the clip and clears
        // loading_key_ under mutex_, and a reader between the two would see a load that
        // ended with its clip incomplete.
        std::lock_guard<std::mutex> lock(mutex_);
        Parameters status = cache_->status();
        status["loading"] = loading_key_;
        status["playing"] = playing_;
        status["start_tick"] = start_tick_;
        return status;
    }

    static std::shared_ptr<ClipCacheNode> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<ClipCacheNode>(nci.edges, params);
        r->cache_ = InstanceSharedObjects<avp::clipcache::ClipCache>::get(
            nci.instance, params.value("cache", std::string("clips")));
        if (params.count("fps")) r->fps_ = parseRatio(params["fps"]);
        r->grid_ = avp::mixer::TickGrid(r->fps_);
        if (params.count("budget_mb"))
            r->cache_->setBudget((size_t)params["budget_mb"].get<double>() * 1024u * 1024u);
        // End-of-stream markers from the decode chain are clip boundaries, not this
        // node's end; forwarding one would end the compositor's clip input.
        r->auto_eof_ = false;
        r->events_ = make_unique<MultiEventWait>(
            std::vector<Event*>{&r->edgeSource()->edge()->producedEvent(), &r->wake_});
        // "url" names the clip the decode chain delivers when the group holding
        // both starts, which is how a preload without a running node fills the cache.
        const std::string url = params.value("url", std::string());
        if (!url.empty()) r->load(url);
        return r;
    }
};

DECLNODE(clip_cache, ClipCacheNode);
