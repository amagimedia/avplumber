#include "cuda_rect_compositor.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

/// Unclocked compositor: each output frame draws the active inputs' frames of one timestamp,
/// matched across inputs, so upstream owns pacing. The mixer's clocked compositor is
/// mixer_compositor, its keyer mixer_keyer.
class CudaRectOverlay : public CudaRectCompositor {
    std::vector<bool> input_eof_;
    std::vector<av::VideoFrame> held_;
    std::vector<bool> held_valid_;

public:
    explicit CudaRectOverlay(const Config &config)
        : CudaRectCompositor("cuda_rect_overlay", config),
          input_eof_(config.inputs),
          held_(config.inputs),
          held_valid_(config.inputs) {}

    void process() override {
        if (sent_eof_)
            return;

        const size_t n = this->source_edges_.size();
        if (n == 0)
            return;

        // Read active_inputs bitmask: get a representative PTS from any peeked frame
        auto active_mask = activeInputs();
        if (hasTimeline()) {
            for (size_t i = 0; i < n; ++i) {
                auto* p = this->source_edges_[i]->peek();
                if (p && !isEofMarker(*p) && frameUsable(*p)) {
                    auto opt = tlGetRaw("active_inputs", p->pts());
                    if (opt) active_mask = avp::mixer::parseSourceMask(*opt);
                    break;
                }
            }
        }
        auto isActive = [active_mask](size_t i) { return active_mask.test((int)i); };

        // "All active inputs exhausted" == EOF only if there is at least one active
        // input. With active_mask == 0 the compositor is idle (e.g. unused PVW slot):
        // falling into the EOF branch here would vacuously set sent_eof_ on the very
        // first process() call and kill the node forever.
        if (input_eof_.size() == n) {
            bool any_active = false;
            bool all_exhausted = true;
            for (size_t i = 0; i < n; ++i) {
                if (!isActive(i)) continue;
                any_active = true;
                if (!input_eof_[i]) {
                    all_exhausted = false;
                    break;
                }
            }
            if (any_active && all_exhausted) {
                av::VideoFrame eof_out;
                eof_out.setPts(NOTS);
                sent_eof_ = true;
                this->sink_->put(eof_out);
                return;
            }
        }

        // No active inputs: nothing to wait for and nothing to produce. Park the
        // thread on any source edge so setObject("active_inputs", >0) can wake it
        // once frames start flowing again.
        {
            bool any_active = false;
            for (size_t i = 0; i < n; ++i) {
                if (isActive(i)) { any_active = true; break; }
            }
            if (!any_active) {
                this->waitForInput();
                return;
            }
        }

        bool any_data = false;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (this->source_edges_[i]->peek() != nullptr) {
                any_data = true;
                break;
            }
        }
        if (!any_data) {
            this->waitForInput();
            return;
        }

        bool all_peek_eof = true;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (p == nullptr || !isEofMarker(*p))
                all_peek_eof = false;
        }
        if (all_peek_eof) {
            for (size_t i = 0; i < n; ++i) {
                if (!isActive(i)) continue;
                this->source_edges_[i]->pop();
                if (i < input_eof_.size())
                    input_eof_[i] = true;
            }
            av::VideoFrame eof_out;
            eof_out.setPts(NOTS);
            sent_eof_ = true;
            this->sink_->put(eof_out);
            return;
        }

        av::Timestamp min_ts = NOTS;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i])
                continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p || isEofMarker(*p))
                continue;
            if (!frameUsable(*p))
                continue;
            if (min_ts.isNoPts() || p->pts() < min_ts)
                min_ts = p->pts();
        }
        if (min_ts.isNoPts()) {
            this->waitForInput();
            return;
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i])
                continue;
            while (true) {
                av::VideoFrame *p = this->source_edges_[i]->peek();
                if (!p)
                    break;
                if (isEofMarker(*p)) {
                    this->source_edges_[i]->pop();
                    input_eof_[i] = true;
                    break;
                }
                if (!frameUsable(*p))
                    break;
                if (p->pts() < min_ts)
                    this->source_edges_[i]->pop();
                else
                    break;
            }
        }

        std::vector<const av::VideoFrame *> src_for_layer(n, nullptr);
        const av::VideoFrame *meta_src = nullptr;
        bool need_wait = false;

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i]) {
                if (held_valid_[i] && !held_[i].isNull())
                    src_for_layer[i] = &held_[i];
                continue;
            }
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p) {
                need_wait = true;
                break;
            }
            if (isEofMarker(*p))
                continue;
            if (!frameUsable(*p)) {
                need_wait = true;
                break;
            }
            if (p->pts() == min_ts)
                src_for_layer[i] = p;
            else if (p->pts() > min_ts) {
                if (held_valid_[i] && !held_[i].isNull())
                    src_for_layer[i] = &held_[i];
            } else
                need_wait = true;
        }
        if (need_wait) {
            this->waitForInput();
            return;
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (p && !input_eof_[i] && frameUsable(*p) && p->pts() == min_ts)
                meta_src = p;
        }
        if (!meta_src) {
            for (size_t i = 0; i < n; ++i) {
                if (src_for_layer[i]) {
                    meta_src = src_for_layer[i];
                    break;
                }
            }
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p || input_eof_[i] || !frameUsable(*p) || p->pts() != min_ts)
                continue;
            requireDrawable(*p);
            av::VideoFrame consumed = *p;
            const bool carries_metadata = p == meta_src;
            this->source_edges_[i]->pop();
            held_[i] = std::move(consumed);
            held_valid_[i] = true;
            src_for_layer[i] = &held_[i];
            // pop destroys the queue entry; retain the metadata source along
            // with the frame reference used for drawing this tick.
            if (carries_metadata) meta_src = &held_[i];
        }

        // During slot warmup, do not emit a partial black composite just because
        // one input has produced an earlier PTS than the others.  Consume fresh
        // frames into held_ until every active layer has a post-activation frame.
        // Once warmup_timeout_ms_ elapses without all layers caught up, fall through
        // to render with whatever we have (missing layers stay black) so a stuck
        // input can never park the compositor indefinitely.
        bool any_missing = false;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i) || input_eof_[i])
                continue;
            if (!src_for_layer[i]) {
                any_missing = true;
                break;
            }
        }
        if (any_missing) {
            const int64_t now_ms = wallclock.pts();
            if (warmup_timeout_ms_ <= 0) {
                this->waitForInput();
                return;
            }
            if (warmup_started_pts_ < 0) {
                warmup_started_pts_ = now_ms;
                this->waitForInput();
                return;
            }
            if (now_ms - warmup_started_pts_ < warmup_timeout_ms_) {
                this->waitForInput();
                return;
            }
            logstream << "cuda_rect_overlay: warmup timeout after "
                      << (now_ms - warmup_started_pts_) << "ms; emitting partial composite";
        }
        warmup_started_pts_ = -1;

        this->sink_->put(compose(min_ts, src_for_layer, meta_src));
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active_inputs") {
            setActiveInputs(avp::mixer::parseSourceMask(value));
            sent_eof_ = false;
            std::fill(input_eof_.begin(), input_eof_.end(), false);
            std::fill(held_valid_.begin(), held_valid_.end(), false);
            warmup_started_pts_ = -1;
            wakeInputs();
        } else {
            CudaRectCompositor::setObject(key, value);
        }
    }

    av::Rational frameRate() override {
        if (!source_edges_.empty()) {
            auto source = source_edges_.front()->findNodeUp<IFrameRateSource>();
            if (source) return source->frameRate();
        }
        return {0, 1};
    }

    static std::shared_ptr<CudaRectOverlay> create(NodeCreationInfo &nci);
};
std::shared_ptr<CudaRectOverlay> CudaRectOverlay::create(NodeCreationInfo &nci) {
    const Parameters &params = nci.params;
    // The parameters of the mixer's clocked nodes: ignoring them would silently drop a caller's
    // clock or subscriptions.
    if (params.contains("fps") || params.contains("clock_input") || params.value("aux_mode", false) ||
        params.contains("subscriptions") || params.contains("pgm_delay_frames"))
        throw Error("cuda_rect_overlay: unclocked; fps, aux_mode, clock_input, subscriptions and "
                    "pgm_delay_frames are mixer_compositor's and mixer_keyer's");
    auto node = std::make_shared<CudaRectOverlay>(parseConfig(nci, "cuda_rect_overlay"));
    node->connect(nci);
    return node;
}

DECLNODE(cuda_rect_overlay, CudaRectOverlay)
