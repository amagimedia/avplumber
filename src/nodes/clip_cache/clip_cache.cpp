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
/// frame; "stop" ends a replay. Between clips the thread waits on its events.
///
/// A replay is stamped on the output tick grid from the tick after it was
/// armed and each frame is pushed when its tick begins, so a clocked compositor
/// downstream matches frame k to tick T0+k with its whole deadline to draw it,
/// and rejects anything stamped before the arm with one reset.
static constexpr int IDLE_END_MS = 200;

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
    int64_t last_input_ns_ = 0;
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

        av::VideoFrame in;
        if (!this->source_->tryGet(in, 0)) {
            {
                // The decode chain stops delivering when the clip runs out and no
                // marker is guaranteed to reach this far. Frames are paced at the
                // clip's own rate, tens of milliseconds apart, so a quiet fifth of
                // a second means the end.
                std::lock_guard<std::mutex> lock(mutex_);
                if (!loading_key_.empty() && loaded_frames_ > 0 &&
                    avp::mixer::monotonicNs() - last_input_ns_ >= (int64_t)IDLE_END_MS * 1000000)
                    finishLoad();
            }
            events_->wait(IDLE_END_MS);
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (loading_key_.empty()) return;   // upstream only runs to fill the cache; nothing here is on air
        // A picture is anything with real dimensions and a timestamp; the
        // completeness flag is not reliable across every producer.
        const bool picture = in.raw() && in.width() > 0 && in.height() > 0 && in.pts().isValid();
        if (loaded_frames_ == 0 && logged_first_ < 3) {
            ++logged_first_;
            logstream << "clip_cache: input frame " << in.width() << "x" << in.height()
                      << " pts_valid=" << in.pts().isValid() << " complete=" << in.isComplete()
                      << " -> " << (picture ? "picture" : "marker");
        }
        if (!picture) {
            // A marker frame. Before the first picture it is the chain starting
            // up, not the clip ending; only the second kind means the cache now
            // holds a whole clip, which is what a preload waits for.
            if (loaded_frames_ > 0) finishLoad();
            return;
        }
        last_input_ns_ = avp::mixer::monotonicNs();
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

    /// A loading pass ends when the group is torn down, which is also when the
    /// decode chain has reached the end of the clip.
    ~ClipCacheNode() override {
        if (!loading_key_.empty() && cache_) {
            logstream << "clip_cache: torn down while loading " << loading_key_ << " after "
                      << loaded_frames_ << " frame(s); completing what was read";
            cache_->finish(loading_key_);
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
        } else {
            throw Error("clip_cache: unknown object key: " + key);
        }
    }

    /// "status": the cache's, plus "playing" and "start_tick", the output tick of the
    /// last replay's first frame, which is where that replay's output begins.
    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("clip_cache: unknown object key: " + key);
        Parameters status = cache_->status();
        std::lock_guard<std::mutex> lock(mutex_);
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
