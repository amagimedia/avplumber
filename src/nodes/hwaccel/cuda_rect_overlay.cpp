#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../mixer/primitives/compositor_layers.hpp"
#include "../../SharedTimeline.hpp"
#include "../../mixer/Playout.hpp"
#include "../../mixer/primitives/MonotonicClock.hpp"
#include "cuda_rect_draw.hpp"
#include <sstream>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using avp::mixer::CudaRectDraw;
using avp::mixer::DrawOp;
using avp::mixer::LayerSpec;

namespace {

bool frameUsable(const av::VideoFrame &f) {
    return !f.isNull() && f.isComplete() && f.raw() && f.pts().isValid();
}

} // namespace

/// Multi-input GPU compositor. Scheduling (clocked playout or timestamp matching), control
/// (`active_inputs`, `layers`, timeline) and output frame plumbing live here; layer geometry is
/// resolved by compositor_layers.hpp and the CUDA work is done by CudaRectDraw.
class CudaRectOverlay : public NodeMultiInput<av::VideoFrame>,
                        public NodeSingleOutput<av::VideoFrame>,
                        public IVideoFormatSource,
                        public IFrameRateSource,
                        public IInputReset,
                        public TimelineReader,
                        public IInputsObjects {
    AVBufferRef *out_frames_ref_ = nullptr;

    int canvas_w_ = 0;
    int canvas_h_ = 0;
    AVPixelFormat sw_fmt_ = AV_PIX_FMT_NONE;
    // Canvas color contract; unspecified preserves the node's metadata inheritance for non-mixer callers.
    AVColorTransferCharacteristic canvas_trc_ = AVCOL_TRC_UNSPECIFIED;
    CudaRectDraw draw_;

    bool hasCanvasColor() const { return canvas_trc_ != AVCOL_TRC_UNSPECIFIED; }

    void setCanvasColor(AVFrame *frame) const {
        if (!hasCanvasColor()) return;
        const bool sdr = canvas_trc_ == AVCOL_TRC_BT709;
        frame->color_trc = canvas_trc_;
        frame->color_primaries = sdr ? AVCOL_PRI_BT709 : AVCOL_PRI_BT2020;
        frame->colorspace = sdr ? AVCOL_SPC_BT709 : AVCOL_SPC_BT2020_NCL;
        frame->color_range = AVCOL_RANGE_MPEG;
    }

    std::vector<LayerSpec> default_layers_;
    mutable std::mutex layers_mutex_;
    std::string metadata_key_;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;

    std::string last_ops_desc_;
    bool sent_eof_ = false;

    std::vector<bool> input_eof_;
    std::vector<av::VideoFrame> held_;
    std::vector<bool> held_valid_;
    std::atomic<uint64_t> active_inputs_{~uint64_t{0}};
    std::atomic<uint64_t> prewarm_inputs_{0};

    // An explicit fps opts live, monotonic-PTS inputs into shared playout.
    // Unclocked callers retain the established timestamp-driven behavior.
    std::unique_ptr<avp::mixer::Playout<av::VideoFrame>> playout_;
    av::Rational frame_rate_{0, 1};
    std::atomic<uint64_t> input_generation_{0};
    std::atomic<int64_t> input_valid_from_ns_{0};
    std::atomic<bool> preserve_warm_input_{false};
    uint64_t applied_generation_ = 0;
    uint64_t applied_active_mask_ = 0;
    uint64_t applied_prewarm_mask_ = 0;

    // Bound how long we will wait for every active layer to produce a fresh
    // post-activation frame before emitting with whatever we have. <=0 disables
    // the bound (the historical "wait forever" behavior). The mixer can stall
    // indefinitely if one source is starved (e.g. a freshly switched router
    // output that has not yet been wired) and no timeout is set.
    int64_t warmup_timeout_ms_ = 0;
    int64_t warmup_started_pts_ = -1;

    void freeHwContexts() {
        draw_.unload();
        av_buffer_unref(&out_frames_ref_);
    }

    std::vector<LayerSpec> mergeLayersForTick(const av::VideoFrame *metadata_source) {
        std::vector<LayerSpec> layers;
        {
            std::lock_guard<std::mutex> lock(layers_mutex_);
            layers = default_layers_;
        }
        if (!metadata_source || !metadata_source->raw() || !metadata_source->raw()->metadata)
            return layers;
        AVDictionaryEntry *e = av_dict_get(metadata_source->raw()->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!e || !e->value)
            return layers;
        try {
            avp::mixer::applyLayerMetadata(layers, e->value);
        } catch (const std::exception &e) {
            logstream << "cuda_rect_overlay: ignoring bad per-frame metadata: " << e.what();
        }
        return layers;
    }

    bool hwSwFormatMatch(const av::VideoFrame &f) const {
        if (!f.raw() || !f.raw()->hw_frames_ctx || !f.raw()->hw_frames_ctx->data)
            return false;
        return avp::mixer::canvasAccepts(CudaRectDraw::frameSwFormat(f), sw_fmt_);
    }

    void processComposite(av::Timestamp pts, const std::vector<const av::VideoFrame *> &sources,
                          const av::VideoFrame *metadata_src) {
        draw_.ensureDevice();
        CUstream stream = draw_.stream();

        av::VideoFrame outf;
        int r = av_hwframe_get_buffer(out_frames_ref_, outf.raw(), 0);
        if (r < 0)
            throw Error(std::string("cuda_rect_overlay: av_hwframe_get_buffer failed: ") + av::error2string(r));
        outf.setComplete(true);   // av_hwframe_get_buffer set hw_frames_ctx, format and size

        std::vector<LayerSpec> layers = mergeLayersForTick(metadata_src);
        std::vector<DrawOp> ops = avp::mixer::resolveDrawOps(sources, layers, canvas_w_, canvas_h_, sw_fmt_);
        {
            // Log the resolved layer set whenever it changes (scene switches), not per tick.
            std::string desc = avp::mixer::describeDrawOps(ops);
            if (desc != last_ops_desc_) {
                last_ops_desc_ = std::move(desc);
                logstream << "cuda_rect_overlay layers:" << last_ops_desc_;
            }
        }
        setCanvasColor(outf.raw());
        // Background and every layer in one kernel launch.
        draw_.draw(stream, ops, outf.raw(), hasCanvasColor() ? outf.raw() : (metadata_src ? metadata_src->raw() : nullptr));

        if (metadata_src && metadata_src->raw()) {
            const int cpy = av_frame_copy_props(outf.raw(), metadata_src->raw());
            if (cpy < 0)
                throw Error(std::string("cuda_rect_overlay: av_frame_copy_props failed: ") + av::error2string(cpy));
        }
        setCanvasColor(outf.raw());
        if (hasCanvasColor())
            av_frame_side_data_remove_by_props(&outf.raw()->side_data, &outf.raw()->nb_side_data,
                                               AV_SIDE_DATA_PROP_COLOR_DEPENDENT);
        outf.setPts(pts);

        if (AVP_CHECK_CU(cuStreamSynchronize(stream)))
            throw Error("cuda_rect_overlay: cuStreamSynchronize failed");

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0)
            logstream << "cuda_rect_overlay: out frame=" << frame_counter_;

        ++frame_counter_;
        this->sink_->put(std::move(outf));
    }

public:
    using NodeSingleOutput<av::VideoFrame>::NodeSingleOutput;

    CudaRectOverlay(std::unique_ptr<Sink<av::VideoFrame>> &&sink, std::shared_ptr<HWAccelDevice> hw, int cw, int ch,
                    AVPixelFormat sw_fmt, std::vector<LayerSpec> layers, std::string metadata_key, int dbg_n)
        : NodeSingleOutput<av::VideoFrame>(std::move(sink)),
          canvas_w_(cw),
          canvas_h_(ch),
          sw_fmt_(sw_fmt),
          draw_(hw, CudaRectDraw::Canvas{cw, ch, sw_fmt}),
          default_layers_(std::move(layers)),
          metadata_key_(std::move(metadata_key)),
          debug_log_every_n_(dbg_n) {
        out_frames_ref_ = av_hwframe_ctx_alloc(hw->deviceContext());
        if (!out_frames_ref_)
            throw Error("cuda_rect_overlay: av_hwframe_ctx_alloc failed");
        AVHWFramesContext *fc = (AVHWFramesContext *)out_frames_ref_->data;
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = sw_fmt_;
        fc->width = canvas_w_;
        fc->height = canvas_h_;
        int err = av_hwframe_ctx_init(out_frames_ref_);
        if (err < 0)
            throw Error(std::string("cuda_rect_overlay: av_hwframe_ctx_init (output) failed: ") +
                        av::error2string(err));

        input_eof_.resize(default_layers_.size());
        held_.resize(default_layers_.size());
        held_valid_.resize(default_layers_.size());
        this->auto_eof_ = false;
    }

    ~CudaRectOverlay() override { freeHwContexts(); }

    void init(EdgeManager &edges, const Parameters &params) override {
        if (params.value("scale", false)) draw_.ensureKernels();
        (void)edges;
        (void)params;
        NodeSingleOutput<av::VideoFrame>::init(edges, params);
    }

    std::weak_ptr<Node> sourceNode() override {
        if (source_edges_.empty())
            return {};
        return source_edges_[0]->producer();
    }

    std::shared_ptr<EdgeBase> sourceEdge() override {
        if (source_edges_.empty())
            return {};
        return source_edges_[0];
    }

    void processClocked() {
        const int64_t now = avp::mixer::monotonicNs();
        uint64_t active = active_inputs_.load(std::memory_order_acquire);
        const uint64_t prewarm = prewarm_inputs_.load(std::memory_order_acquire);
        if (hasTimeline()) {
            const auto deadline = playout_->nextDeadline();
            const int64_t content_time = std::max(now - playout_->latencyNs(),
                deadline ? *deadline - playout_->latencyNs() : now);
            const auto value = tlGetRaw("active_inputs", av::Timestamp(content_time, {1, 1000000000}));
            if (value) active = parseBitmask<uint64_t>(*value);
        }
        const auto generation = input_generation_.load(std::memory_order_acquire);
        if (generation != applied_generation_ || active != applied_active_mask_ || prewarm != applied_prewarm_mask_) {
            for (size_t i = 0; i < source_edges_.size(); ++i) {
                playout_->setPrewarm(i, (prewarm & (uint64_t{1} << i)) != 0);
                playout_->setActive(i, (active & (uint64_t{1} << i)) != 0);
                if (generation != applied_generation_) {
                    const auto from = input_valid_from_ns_.load(std::memory_order_acquire);
                    playout_->resetInput(i, from ? std::optional<int64_t>(from) : std::nullopt,
                                         preserve_warm_input_.load(std::memory_order_acquire));
                }
            }
            applied_generation_ = generation;
            applied_active_mask_ = active;
            applied_prewarm_mask_ = prewarm;
            warmup_started_pts_ = wallclock.pts();
            sent_eof_ = false;
        }
        if (sent_eof_) return;
        if (!(active | prewarm)) {
            this->waitForInput();
            return;
        }
        for (size_t i = 0; i < source_edges_.size(); ++i) {
            if (!((active | prewarm) & (uint64_t{1} << i))) continue;
            // Bound ingestion as well as storage; an unpaced producer must not
            // monopolize the output thread before it reaches its deadline.
            for (size_t received = 0; received < 8; ++received) {
                auto *frame = source_edges_[i]->peek();
                if (!frame) break;
                if (isEofMarker(*frame)) {
                    playout_->endInput(i);
                    source_edges_[i]->pop();
                    break;
                }
                if (frameUsable(*frame)) {
                    if (frame->raw()->format != AV_PIX_FMT_CUDA)
                        throw Error("cuda_rect_overlay: input must be AV_PIX_FMT_CUDA");
                    if (!hwSwFormatMatch(*frame))
                        throw Error("cuda_rect_overlay: input hw sw_format mismatch node sw_format");
                    playout_->push(i, *frame, frame->pts().timestamp({1, 1000000000}));
                }
                source_edges_[i]->pop();
            }
        }
        if (playout_->finished()) {
            av::VideoFrame eof;
            eof.setPts(NOTS);
            this->sink_->put(eof);
            sent_eof_ = true;
            return;
        }
        const bool require_all = warmup_timeout_ms_ <= 0 ||
            wallclock.pts() - warmup_started_pts_ < warmup_timeout_ms_;
        const auto *decision = playout_->prepare(avp::mixer::monotonicNs(), require_all);
        if (!decision) {
            this->findSourceWithData(avp::mixer::waitMilliseconds(
                playout_->nextDeadline(), avp::mixer::monotonicNs()));
            return;
        }
        std::vector<const av::VideoFrame *> sources;
        const av::VideoFrame *metadata = nullptr;
        for (size_t i = 0; i < decision->frames.size(); ++i) {
            const auto &frame = decision->frames[i];
            const bool visible = (active & (uint64_t{1} << i)) != 0;
            sources.push_back(visible && frame ? &*frame : nullptr);
            if (visible && frame) metadata = &*frame;
        }
        // Warm inputs advance their bounded reference queues without allocating
        // an output surface or issuing any CUDA composition for an idle slot.
        if (active) processComposite(av::Timestamp(decision->index, av_inv_q(frame_rate_.getValue())), sources, metadata);
        playout_->commit();
        if (debug_log_every_n_ > 0 && frame_counter_ % debug_log_every_n_ == 0) {
            std::ostringstream stats;
            stats << "cuda_rect_overlay: frames=" << frame_counter_
                  << " missed_deadlines=" << playout_->missedDeadlines();
            for (size_t i = 0; i < sources.size(); ++i) {
                if (!(active & (uint64_t{1} << i))) continue;
                stats << " input" << i << "_reuse=" << playout_->stats(i).repeats
                      << " input" << i << "_discarded=" << playout_->stats(i).discarded;
            }
            logstream << stats.str();
        }
    }

    void process() override {
        if (playout_) {
            processClocked();
            return;
        }
        if (sent_eof_)
            return;

        const size_t n = this->source_edges_.size();
        if (n == 0)
            return;

        // Read active_inputs bitmask: get a representative PTS from any peeked frame
        uint64_t active_mask = active_inputs_.load(std::memory_order_relaxed);
        if (hasTimeline()) {
            for (size_t i = 0; i < n; ++i) {
                auto* p = this->source_edges_[i]->peek();
                if (p && !isEofMarker(*p) && frameUsable(*p)) {
                    auto opt = tlGetRaw("active_inputs", p->pts());
                    if (opt) active_mask = parseBitmask<uint64_t>(*opt);
                    break;
                }
            }
        }
        auto isActive = [active_mask](size_t i) { return (active_mask & (uint64_t{1} << i)) != 0; };

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
                this->sink_->put(std::move(eof_out));
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
            this->sink_->put(std::move(eof_out));
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
            if (p->raw()->format != AV_PIX_FMT_CUDA)
                throw Error("cuda_rect_overlay: input must be AV_PIX_FMT_CUDA");
            if (!hwSwFormatMatch(*p))
                throw Error("cuda_rect_overlay: input hw sw_format mismatch node sw_format");
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

        processComposite(min_ts, src_for_layer, meta_src);
    }

    void resetInput() override {
        if (!playout_) return;
        preserve_warm_input_.store(false, std::memory_order_release);
        input_valid_from_ns_.store(avp::mixer::monotonicNs(), std::memory_order_release);
        input_generation_.fetch_add(1, std::memory_order_release);
        for (auto &edge : source_edges_) edge->producedEvent().signal();
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active_inputs") {
            const uint64_t new_mask = parseBitmask<uint64_t>(value);
            active_inputs_.store(new_mask, std::memory_order_relaxed);
            if (!playout_) {
                sent_eof_ = false;
                std::fill(input_eof_.begin(), input_eof_.end(), false);
                std::fill(held_valid_.begin(), held_valid_.end(), false);
                warmup_started_pts_ = -1;
            }
            for (auto& edge : this->source_edges_) {
                edge->producedEvent().signal();
            }
        } else if (key == "prewarm_inputs") {
            if (!playout_) throw Error("cuda_rect_overlay: prewarm_inputs requires clocked playout");
            prewarm_inputs_.store(parseBitmask<uint64_t>(value), std::memory_order_release);
            for (auto &edge : source_edges_) edge->producedEvent().signal();
        } else if (key == "warm_reset") {
            if (!playout_) throw Error("cuda_rect_overlay: warm_reset requires clocked playout");
            const auto period = avp::mixer::TickGrid(frame_rate_).time(1);
            input_valid_from_ns_.store(avp::mixer::monotonicNs() - playout_->latencyNs() - period,
                                       std::memory_order_release);
            preserve_warm_input_.store(true, std::memory_order_release);
            input_generation_.fetch_add(1, std::memory_order_release);
            for (auto &edge : source_edges_) edge->producedEvent().signal();
        } else if (key == "layers") {
            auto new_layers = avp::mixer::parseLayersArray(value);
            std::lock_guard<std::mutex> lock(layers_mutex_);
            default_layers_ = std::move(new_layers);
        }
    }

    av::Rational frameRate() override {
        if (playout_) return frame_rate_;
        if (!source_edges_.empty()) {
            auto source = source_edges_.front()->findNodeUp<IFrameRateSource>();
            if (source) return source->frameRate();
        }
        return {0, 1};
    }

    int width() override { return canvas_w_; }
    int height() override { return canvas_h_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(sw_fmt_); }

    static std::shared_ptr<CudaRectOverlay> create(NodeCreationInfo &nci);
};
std::shared_ptr<CudaRectOverlay> CudaRectOverlay::create(NodeCreationInfo &nci) {
    EdgeManager &edges = nci.edges;
    const Parameters &params = nci.params;
    auto src_names = jsonToStringList(params["src"]);
    if (src_names.empty())
        throw Error("cuda_rect_overlay: at least one input required in src");
    if (src_names.size() > 64)
        throw Error("cuda_rect_overlay: at most 64 inputs supported");
    std::vector<LayerSpec> layers = avp::mixer::parseLayersParam(params);
    if (layers.size() != src_names.size())
        throw Error("cuda_rect_overlay: layers array length must match src count");

    if (!params.contains("hwaccel"))
        throw Error("cuda_rect_overlay: hwaccel parameter required");
    auto hw = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
    if (!hw)
        throw Error("cuda_rect_overlay: failed to resolve hwaccel");

    const int cw = params.at("width").get<int>();
    const int ch = params.at("height").get<int>();
    if (cw <= 0 || ch <= 0)
        throw Error("cuda_rect_overlay: width and height must be positive");
    const std::string sw_name = params.value("sw_format", std::string("nv12"));
    const AVPixelFormat sw_fmt = av_get_pix_fmt(sw_name.c_str());
    if (sw_fmt == AV_PIX_FMT_NONE)
        throw Error("cuda_rect_overlay: unknown sw_format");
    if (!CudaRectDraw::canvasSupported(sw_fmt))
        throw Error("cuda_rect_overlay: sw_format must be semiplanar YUV (nv12, p010le, p210le) or packed 8-bit RGB");

    auto out_edge = edges.find<av::VideoFrame>(params["dst"]);

    const std::string mdkey = params.value("metadata_key", std::string("rect_overlay_v1"));
    const int dbg = params.value("debug_log_every_n", 0);

    auto node = std::make_shared<CudaRectOverlay>(
        make_unique<EdgeSink<av::VideoFrame>>(out_edge), std::move(hw), cw, ch, sw_fmt, std::move(layers), mdkey,
        dbg);
    if (params.contains("color")) {
        const auto color = params.at("color").get<std::string>();
        node->canvas_trc_ = color == "sdr" ? AVCOL_TRC_BT709 : color == "hlg" ? AVCOL_TRC_ARIB_STD_B67 :
                            color == "pq" ? AVCOL_TRC_SMPTE2084 : AVCOL_TRC_UNSPECIFIED;
        if (!node->hasCanvasColor())
            throw Error("cuda_rect_overlay: color must be sdr, hlg or pq");
        if (node->canvas_trc_ != AVCOL_TRC_BT709 && av_pix_fmt_desc_get(sw_fmt)->comp[0].depth < 10)
            throw Error("cuda_rect_overlay: HDR canvas requires 10-bit storage");
        const float sdr_white = params.value("sdr_white", 203.f);
        const float hdr_peak = params.value("hdr_peak", 1000.f);
        if (!(sdr_white >= 1.f && sdr_white <= hdr_peak && hdr_peak >= 100.f && hdr_peak <= 10000.f))
            throw Error("cuda_rect_overlay: invalid display white/peak");
        node->draw_.setColor(node->canvas_trc_, sdr_white, hdr_peak);
    }
    node->createSourcesFromParameters(edges, params);
    out_edge->setProducer(node);
    node->initTimeline(nci);
    if (params.count("active_inputs"))
        node->active_inputs_.store(parseBitmask<uint64_t>(params["active_inputs"]), std::memory_order_relaxed);
    node->warmup_timeout_ms_ = params.value("warmup_timeout_ms", (int64_t)0);
    if (params.contains("fps")) {
        node->frame_rate_ = parseRatio(params.at("fps"));
        std::optional<double> latency_ms;
        if (params.contains("latency_ms")) latency_ms = params.at("latency_ms").get<double>();
        node->playout_ = std::make_unique<avp::mixer::Playout<av::VideoFrame>>(
            src_names.size(), avp::mixer::TickGrid(node->frame_rate_), latency_ms, avp::mixer::TimestampMode::Presentation);
        node->input_generation_.store(1);
        logstream << "cuda_rect_overlay: latency_ms=" << node->playout_->latencyNs() / 1000000.0;
    }

    return node;
}

DECLNODE(cuda_rect_overlay, CudaRectOverlay);
