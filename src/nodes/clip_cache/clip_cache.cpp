#include "../node_common.hpp"
#include "ClipCache.hpp"
#include "../../mixer/MonotonicClock.hpp"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
}

/// Replay a short clip from GPU memory, filling the cache on its first pass.
///
/// Placed at the end of a normal decode chain, this node passes frames through
/// and keeps them. Started on its own afterwards, with the decode chain left
/// stopped, it replays the cached frames at the requested rate: no file open,
/// no decoder, no thread startup on the path of a live transition.
///
/// The clip is identified by the "url" parameter, which is what the mixer sets
/// when it arms a media wipe, so nothing upstream needs to know about caching.
class ClipCacheNode : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                      public ReportsFinishByFlag, public IReturnsObjects,
                      public IFrameRateSource, public ITimeBaseSource {
protected:
    std::shared_ptr<avp::clipcache::ClipCache> cache_;
    std::string key_;
    av::Rational fps_{60, 1};
    bool caching_ = false;              // this pass is filling the cache
    bool started_ = false;
    std::vector<av::VideoFrame> playback_;
    size_t index_ = 0;
    size_t cached_frames_ = 0;
    int64_t first_frame_ns_ = 0;
    int64_t frame_ns_ = 16666667;

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

    av::Timestamp ptsFor(size_t index) {
        return av::Timestamp((int64_t)index, timeBase());
    }

public:
    using NodeSISO::NodeSISO;

    av::Rational frameRate() override { return fps_; }
    av::Rational timeBase() override { return {fps_.getDenominator(), fps_.getNumerator()}; }

    void process() override {
        if (!started_) {
            started_ = true;
            playback_ = key_.empty() ? std::vector<av::VideoFrame>() : cache_->take(key_);
            caching_ = playback_.empty() && !key_.empty();
            cached_frames_ = 0;
            if (caching_) {
                cache_->begin(key_);
                logstream << "clip_cache: loading " << key_;
            } else if (!playback_.empty()) {
                logstream << "clip_cache: replaying " << playback_.size() << " cached frame(s) of " << key_;
            }
        }

        if (!playback_.empty()) {
            if (index_ >= playback_.size()) {
                markFinished();
                return;
            }
            const int64_t now = avp::mixer::monotonicNs();
            if (first_frame_ns_ == 0) first_frame_ns_ = now;
            const int64_t due = first_frame_ns_ + (int64_t)index_ * frame_ns_;
            if (now < due) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(due - now, frame_ns_)));
                return;
            }
            av::VideoFrame out = playback_[index_];
            out.setTimeBase(timeBase());
            out.setPts(ptsFor(index_));
            out.setComplete(true);
            this->sink_->put(out);
            ++index_;
            return;
        }

        av::VideoFrame in = this->source_->get();
        if (!in) return;
        if (caching_) {
            if (!in.isComplete() || !in.pts().isValid()) {
                // A marker frame. Before the first picture it is the chain
                // starting up, not the clip ending; only the second kind means
                // the cache now holds a whole clip, which is what a preload
                // waits for.
                if (cached_frames_ > 0) {
                    cache_->finish(key_);
                    caching_ = false;
                    logstream << "clip_cache: cached " << cached_frames_ << " frame(s) of " << key_;
                }
            } else if (!cache_->append(key_, in, frameBytes(in))) {
                caching_ = false;   // over budget: stay a passthrough for this clip
                logstream << "clip_cache: " << key_ << " does not fit the budget; not cached";
            } else {
                ++cached_frames_;
            }
        }
        if (caching_) {
            // A preload runs with nothing downstream consuming, so waiting for
            // room would stall the clip half-decoded. The frames that matter are
            // already in the cache; forwarding is best effort here. A live pass
            // has a consumer, so this put succeeds and nothing is dropped.
            this->sink_->put(in, true);
            return;
        }
        this->sink_->put(in);
    }

    /// The loading pass ends when the group is torn down, which is also when the
    /// decode chain has reached the end of the clip.
    ~ClipCacheNode() override {
        if (caching_ && cache_) cache_->finish(key_);
    }

    Parameters getObject(const std::string key) override {
        if (key == "status") return cache_->status();
        throw Error("clip_cache: unknown object key: " + key);
    }

    static std::shared_ptr<ClipCacheNode> create(NodeCreationInfo& nci) {
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<ClipCacheNode>(nci.edges, params);
        r->cache_ = InstanceSharedObjects<avp::clipcache::ClipCache>::get(
            nci.instance, params.value("cache", std::string("clips")));
        // Read at creation: the engine recreates this node for every take, which
        // is how the mixer's newly armed clip reaches it.
        r->key_ = params.value("url", std::string());
        if (params.count("fps")) r->fps_ = parseRatio(params["fps"]);
        r->frame_ns_ = (int64_t)(1000000000.0 * r->fps_.getDenominator() / r->fps_.getNumerator());
        if (params.count("budget_mb"))
            r->cache_->setBudget((size_t)params["budget_mb"].get<double>() * 1024u * 1024u);
        return r;
    }
};

DECLNODE(clip_cache, ClipCacheNode);
