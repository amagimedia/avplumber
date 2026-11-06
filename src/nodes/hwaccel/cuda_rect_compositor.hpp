#pragma once
// Base of the CUDA rect compositor nodes: cuda_rect_overlay (unclocked, timestamp matching),
// mixer_compositor (the mixer's clocked scenes, wipe and AUX buses) and mixer_keyer (the DSK).
#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../avbuffer.hpp"
#include "../../timeline.hpp"
#include "../../output_mask.hpp"
#include "../../mixer/primitives/compositor_layers.hpp"
#include "../../mixer/primitives/frame_subscription.hpp"
#include "../../mixer/primitives/source_mask.hpp"
#include "../../mixer/primitives/SourceDelivery.hpp"
#include "cuda_rect_draw.hpp"
#include "cuda_rect_frame.hpp"
#include "cuda_rect_texture.h"

#include <chrono>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

/// Multi-input GPU compositor. Each node decides which frames to draw when; this class holds
/// what they share: the canvas and its output frame pool, control (`layers`, per-frame
/// `metadata_key` layers, `active_inputs`), input checks and the drawing of one output
/// frame. Layer geometry is resolved by compositor_layers.hpp and the CUDA work is done by
/// CudaRectDraw.
class CudaRectCompositor : public NodeMultiInput<av::VideoFrame>,
                           public NodeSingleOutput<av::VideoFrame>,
                           public IVideoFormatSource,
                           public IFrameRateSource,
                           public IInputsObjects {
public:
    using CudaRectDraw = avp::mixer::CudaRectDraw;
    using DrawOp = avp::mixer::DrawOp;
    using LayerSpec = avp::mixer::LayerSpec;

    /// The creation parameters every compositor node takes, validated by parseConfig().
    struct Config {
        size_t inputs = 0;   // `src` count
        std::vector<LayerSpec> layers;
        int max_layers = 256;
        std::shared_ptr<HWAccelDevice> hw;
        CudaRectDraw::Canvas canvas;   // with the `color` contract
        std::shared_ptr<Edge<av::VideoFrame>> output;
        std::string metadata_key;
        int debug_log_every_n = 0;
        // compose() returns once the draw is queued: the next compose() waits for it to finish and
        // holds the frames it reads until then. Consumers order on the canvas frames' stream
        // (FFmpeg filters and NVENC use it; compositors wait for their producers' streams).
        bool hand_off = false;
    };

    /// `type` is the node type, the prefix of errors and logs.
    static Config parseConfig(NodeCreationInfo &nci, const char *type);

protected:
    const char *const type_;
    std::shared_ptr<Edge<av::VideoFrame>> output_edge_;
    avp::AvBufferRef out_frames_ref_;
    CudaRectDraw draw_;   // owns the canvas geometry, format and color contract

    std::vector<LayerSpec> default_layers_;
    mutable std::mutex layers_mutex_;
    // Apply the metadata source's per-frame layers under metadata_key_; an AUX bus takes its
    // layers from `composition` only.
    bool frame_layers_ = true;
    std::string metadata_key_;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    // The last compose(): queueing the draw (our work plus the CUDA library's own locking), and
    // waiting for the GPU to finish a draw and the source frames it reads: this one, or with
    // Config::hand_off the previous one.
    int64_t issue_ns_ = 0, gpu_wait_ns_ = 0;

    // Config::hand_off: the queued draw's completion and the frames it reads, held until a later
    // compose() or flush finds it finished. A source let go early would go back to its decoder or
    // browser while a kernel still reads it.
    const bool hand_off_;
    CUevent draw_done_ = nullptr;
    bool draw_pending_ = false;
    std::vector<av::VideoFrame> in_flight_;
    void finishPreviousDraw() {
        if (!draw_pending_) return;
        if (AVP_CHECK_CU(cuEventSynchronize(draw_done_)))
            throw Error(std::string(type_) + ": waiting for the previous draw failed");
        in_flight_.clear();
        draw_pending_ = false;
    }

    // The layer log: the configured layers and the draw shape it last reported. Geometry that only
    // per-frame metadata moved (face framing, every frame) is not logged; status objects carry it.
    std::vector<LayerSpec> logged_layers_;
    std::vector<avp::mixer::DrawOpShape> logged_shape_;
    bool sent_eof_ = false;

    // The pad mask does not fit a lock-free atomic, and the input masks are read once per frame: one
    // mutex guards them, like default_layers_ above, and the nodes' other per-input control state.
    mutable std::mutex masks_mutex_;
    avp::mixer::SourceMask active_inputs_ = avp::mixer::SourceMask().set();
    avp::mixer::SourceMask activeInputs() const {
        std::lock_guard<std::mutex> lock(masks_mutex_);
        return active_inputs_;
    }
    void setActiveInputs(avp::mixer::SourceMask mask) {
        std::lock_guard<std::mutex> lock(masks_mutex_);
        active_inputs_ = mask;
    }

    // The mixer's nodes subscribe their inputs to shared sources, which never wait for them.
    std::vector<std::shared_ptr<avp::mixer::FrameSubscription>> subscriptions_;

    // Bound how long we will wait for every active layer to produce a fresh
    // post-activation frame before emitting with whatever we have. <=0 disables
    // the bound (the historical "wait forever" behavior). The mixer can stall
    // indefinitely if one source is starved (e.g. a freshly switched router
    // output that has not yet been wired) and no timeout is set.
    int64_t warmup_timeout_ms_ = 0;
    int64_t warmup_started_pts_ = -1;

    static bool frameUsable(const av::VideoFrame &f) {
        return !f.isNull() && f.isComplete() && f.raw() && f.pts().isValid();
    }

    /// Wakes the render thread waiting for input, to apply a control change.
    void wakeInputs() {
        for (auto &edge : source_edges_) edge->producedEvent().signal();
    }

    /// After the end-of-stream marker there is nothing to draw: parks the render thread until
    /// stop(), an input frame or a control change wakes it, since returning from process() at
    /// once would spin it. True while the stream has ended.
    bool parkedAfterEof() {
        if (!sent_eof_) return false;
        this->waitForInput();
        return true;
    }

    // An unspecified canvas color preserves the node's metadata inheritance for non-mixer callers.
    bool hasCanvasColor() const { return draw_.canvas().transfer != AVCOL_TRC_UNSPECIFIED; }

    void setCanvasColor(AVFrame *frame) const {
        if (!hasCanvasColor()) return;
        const AVColorTransferCharacteristic trc = draw_.canvas().transfer;
        const bool sdr = trc == AVCOL_TRC_BT709;
        frame->color_trc = trc;
        frame->color_primaries = sdr ? AVCOL_PRI_BT709 : AVCOL_PRI_BT2020;
        frame->colorspace = sdr ? AVCOL_SPC_BT709 : AVCOL_SPC_BT2020_NCL;
        frame->color_range = AVCOL_RANGE_MPEG;
    }

    /// `configured_changed`: the configured layers differ from those the layer log last reported.
    std::vector<LayerSpec> mergeLayersForTick(const av::VideoFrame *metadata_source, bool &configured_changed) {
        std::vector<LayerSpec> layers;
        {
            std::lock_guard<std::mutex> lock(layers_mutex_);
            layers = default_layers_;
            configured_changed = default_layers_ != logged_layers_;
            if (configured_changed) logged_layers_ = default_layers_;
        }
        if (!frame_layers_ || !metadata_source || !metadata_source->raw() || !metadata_source->raw()->metadata)
            return layers;
        AVDictionaryEntry *e = av_dict_get(metadata_source->raw()->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!e || !e->value)
            return layers;
        try {
            avp::mixer::applyLayerMetadata(layers, e->value);
        } catch (const std::exception &e) {
            logstream << type_ << ": ignoring bad per-frame metadata: " << e.what();
        }
        return layers;
    }

    bool hwSwFormatMatch(const av::VideoFrame &f) const {
        const AVPixelFormat fmt = CudaRectDraw::frameSwFormat(f);
        return fmt != AV_PIX_FMT_NONE && avp::mixer::canvasAccepts(fmt, draw_.canvas().sw_fmt);
    }

    void requireDrawable(const av::VideoFrame &f) const {
        if (!CudaRectDraw::frameSupported(static_cast<AVPixelFormat>(f.raw()->format)))
            throw Error(std::string(type_) + ": input must be a CUDA device frame or CUarray");
        if (!hwSwFormatMatch(f))
            throw Error(std::string(type_) + ": input hw sw_format mismatch node sw_format");
    }

    /// Draws the output frame stamped `pts` and waits for the GPU (Config::hand_off: for the
    /// previous draw, before this one is queued). `sources` holds a frame per
    /// input, null where none is drawn; `metadata_src` gives the frame its properties and
    /// per-frame layers; `opacity`, when given, holds a weight per input (see LayerSpec::opacity).
    /// `base`, when given, is the source the bottom layer draws; where that layer is the frame
    /// unchanged, only the rectangles of the other layers are composed (CudaRectDraw::drawOver).
    av::VideoFrame compose(av::Timestamp pts, const std::vector<const av::VideoFrame *> &sources,
                           const av::VideoFrame *metadata_src, const std::vector<float> *opacity = nullptr,
                           const av::VideoFrame *base = nullptr) {
        const auto started = std::chrono::steady_clock::now();
        draw_.ensureDevice();
        if (hand_off_) {
            avp::timeline::Scope wait("gpu wait");
            finishPreviousDraw();
        }
        const auto waited = std::chrono::steady_clock::now();
        avp::timeline::Scope issue("issue");
        CUstream stream = draw_.stream();

        av::VideoFrame outf = avp::mixer::canvasFrame(out_frames_ref_.get(), type_);

        bool configured_changed = false;
        std::vector<LayerSpec> layers = mergeLayersForTick(metadata_src, configured_changed);
        if (opacity)
            for (size_t i = 0; i < layers.size(); ++i) {
                const size_t input = layers[i].sourceIndex(i);
                if (input < opacity->size()) layers[i].opacity = (*opacity)[input];
            }
        const CudaRectDraw::Canvas &cv = draw_.canvas();
        std::vector<DrawOp> ops = avp::mixer::resolveDrawOps(sources, layers, cv.width, cv.height, cv.sw_fmt);
        {
            // Log the resolved layer set when a scene or composition sets new layers or what is drawn
            // changes, not when per-frame metadata moves a crop.
            std::vector<avp::mixer::DrawOpShape> shape = avp::mixer::drawOpsShape(ops);
            if (configured_changed || shape != logged_shape_) {
                logged_shape_ = std::move(shape);
                logstream << type_ << " layers:" << avp::mixer::describeDrawOps(ops);
            }
        }
        setCanvasColor(outf.raw());
        // From the first launch on, a kernel may be reading the source frames. On an exception
        // the playout that owns them can go first, and a decoder reuses its surfaces at once.
        struct SynchronizeOnUnwind {
            CudaRectDraw &draw;
            CUstream stream;
            bool drawn = false;
            ~SynchronizeOnUnwind() {
                if (drawn) return;
                try { draw.synchronize(stream); }
                catch (const std::exception &e) { logstream << "compose: synchronize while unwinding: " << e.what(); }
            }
        } unwinding{draw_, stream};
        // Background and every layer in one kernel launch; over a base, only where the other layers are.
        const AVFrame *color_src = hasCanvasColor() ? outf.raw() : (metadata_src ? metadata_src->raw() : nullptr);
        if (!base || !draw_.drawOver(stream, ops, outf.raw(), color_src, base->raw()))
            draw_.draw(stream, ops, outf.raw(), color_src);

        if (metadata_src && metadata_src->raw())
            avp::mixer::copySourceProps(outf, *metadata_src, type_);
        avp::mixer::clearSourceFrames(outf.raw());
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& op = ops[i];
            if (!op.src || op.layer.opacity <= 0) continue;
            const auto bounds = [](const LayerSpec& l) {
                return avp::mixer::Rect{l.dst_x, l.dst_y, l.dst_w ? l.dst_w : l.crop_w,
                                       l.dst_h ? l.dst_h : l.crop_h};
            };
            const auto r = bounds(op.layer);
            const bool covered = std::any_of(ops.begin() + i + 1, ops.end(), [&](const DrawOp& above) {
                if (!above.src || above.layer.blend || above.layer.opacity <= 0) return false;
                const auto a = bounds(above.layer);
                return a.x <= r.x && a.y <= r.y && a.x + a.w >= r.x + r.w && a.y + a.h >= r.y + r.h;
            });
            if (!covered) avp::mixer::mergeSourceFrames(outf.raw(), op.src->raw());
        }
        setCanvasColor(outf.raw());
        if (hasCanvasColor())
            av_frame_side_data_remove_by_props(&outf.raw()->side_data, &outf.raw()->nb_side_data,
                                               AV_SIDE_DATA_PROP_COLOR_DEPENDENT);
        outf.setPts(pts);

        if (hand_off_) {
            if (!draw_done_ && AVP_CHECK_CU(cuEventCreate(&draw_done_, CU_EVENT_DISABLE_TIMING)))
                throw Error(std::string(type_) + ": cannot create the draw completion event");
            if (AVP_CHECK_CU(cuEventRecord(draw_done_, stream)))
                throw Error(std::string(type_) + ": cannot record the draw completion event");
            for (const auto &op : ops)
                if (op.src) in_flight_.push_back(*op.src);
            if (base) in_flight_.push_back(*base);
            draw_pending_ = true;
            unwinding.drawn = true;   // the held frames outlive the draw from here on
        }
        issue.close();
        const auto issued = std::chrono::steady_clock::now();
        if (!hand_off_) {
            avp::timeline::Scope wait("gpu wait");
            unwinding.drawn = true;   // an exception from here on has nothing left to wait for
            draw_.synchronize(stream);
        }
        const auto done = std::chrono::steady_clock::now();
        const auto ns = [](auto d) { return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); };
        issue_ns_ = ns(issued - waited);
        gpu_wait_ns_ = hand_off_ ? ns(waited - started) : ns(done - issued);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0)
            logstream << type_ << ": out frame=" << frame_counter_;

        ++frame_counter_;
        return outf;
    }

    /// Wires a created node into the graph (`src` edges and the output edge) and reads
    /// `active_inputs` and `warmup_timeout_ms`.
    void connect(NodeCreationInfo &nci) {
        createSourcesFromParameters(nci.edges, nci.params);
        output_edge_->setProducer(this->shared_from_this());
        if (nci.params.count("active_inputs"))
            active_inputs_ = avp::mixer::parseSourceMask(nci.params["active_inputs"]);
        warmup_timeout_ms_ = nci.params.value("warmup_timeout_ms", (int64_t)0);
    }

    /// Subscribes each input to the shared source `subscriptions` names in `src` order, paced at
    /// `fps` unless `rate_limit` is false. With `allow_unsubscribed`, an empty name leaves
    /// that input an ordinary edge.
    void subscribe(NodeCreationInfo &nci, av::Rational fps, bool allow_unsubscribed, bool rate_limit = true) {
        const auto names = jsonToStringList(nci.params.at("subscriptions"));
        if (names.size() != source_edges_.size())
            throw Error(std::string(type_) + ": subscriptions must match inputs");
        for (const auto &name : names) {
            if (name.empty() && allow_unsubscribed) {
                subscriptions_.push_back(nullptr);
                continue;
            }
            auto subscription = InstanceSharedObjects<avp::mixer::FrameSubscription>::get(nci.instance, name);
            subscription->configure(fps, rate_limit);
            subscriptions_.push_back(subscription);
        }
    }

public:
    CudaRectCompositor(const char *type, const Config &config)
        : NodeSingleOutput<av::VideoFrame>(make_unique<EdgeSink<av::VideoFrame>>(config.output)),
          type_(type),
          output_edge_(config.output),
          draw_(config.hw, config.canvas, config.max_layers,
                config.hand_off ? CudaRectDraw::TableCompletion::Event : CudaRectDraw::TableCompletion::Stream),
          default_layers_(config.layers),
          metadata_key_(config.metadata_key),
          debug_log_every_n_(config.debug_log_every_n),
          hand_off_(config.hand_off) {
        out_frames_ref_ = avp::mixer::allocCanvasFrames(*config.hw, config.canvas, type_);
        this->auto_eof_ = false;
    }

    ~CudaRectCompositor() override {
        if (draw_done_) {
            try {
                draw_.ensureDevice();
                finishPreviousDraw();
            } catch (const std::exception &e) { logstream << type_ << ": finishing the last draw: " << e.what(); }
            cuEventDestroy(draw_done_);
        }
        draw_.unload();
        out_frames_ref_.reset();
    }

    void stop() override {
        NodeMultiInput<av::VideoFrame>::stop();
        for (auto &subscription : subscriptions_) if (subscription) subscription->close();
    }

    void init(EdgeManager &edges, const Parameters &params) override {
        draw_.ensureKernels();   // fail at graph build, not on the first frame
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

    void setObject(const std::string key, const Parameters &value) override {
        if (key == "layers") {
            auto new_layers = avp::mixer::parseLayersArray(value);
            if (new_layers.size() > size_t(draw_.maxLayers())) throw Error(std::string(type_) + ": too many layers");
            std::lock_guard<std::mutex> lock(layers_mutex_);
            default_layers_ = std::move(new_layers);
        } else if (key == "composition" || key == "prewarm_inputs" || key == "warm_reset" || key == "fade_inputs") {
            // Another compositor node's control: fail rather than ignore it like an unknown key.
            throw Error(std::string(type_) + ": " + key + " is not supported by this node");
        }
    }

    int width() override { return draw_.canvas().width; }
    int height() override { return draw_.canvas().height; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(draw_.canvas().sw_fmt); }
};

inline CudaRectCompositor::Config CudaRectCompositor::parseConfig(NodeCreationInfo &nci, const char *type) {
    const std::string prefix = std::string(type) + ": ";
    const Parameters &params = nci.params;
    Config config;
    config.inputs = jsonToStringList(params["src"]).size();
    if (config.inputs == 0)
        throw Error(prefix + "at least one input required in src");
    if (config.inputs > avp::mixer::kSourceMaskBits)
        throw Error(prefix + "at most " + std::to_string(avp::mixer::kSourceMaskBits) + " inputs supported");
    config.layers = avp::mixer::parseLayersParam(params);
    config.max_layers = params.value("max_layers", 256);
    if (config.max_layers <= 0) throw Error(prefix + "max_layers must be positive");
    if (config.layers.size() > size_t(config.max_layers)) throw Error(prefix + "too many layers");
    for (size_t i = 0; i < config.layers.size(); ++i)
        if (size_t(config.layers[i].input < 0 ? int(i) : config.layers[i].input) >= config.inputs)
            throw Error(prefix + "layer input out of range");

    if (!params.contains("hwaccel"))
        throw Error(prefix + "hwaccel parameter required");
    config.hw = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
    if (!config.hw)
        throw Error(prefix + "failed to resolve hwaccel");

    config.canvas = avp::mixer::parseCanvas(params, prefix);

    config.output = nci.edges.find<av::VideoFrame>(params["dst"]);
    config.metadata_key = params.value("metadata_key", std::string("rect_overlay_v1"));
    config.debug_log_every_n = params.value("debug_log_every_n", 0);
    return config;
}
