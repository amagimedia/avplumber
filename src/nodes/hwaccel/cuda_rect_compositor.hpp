#pragma once
// Base of the CUDA rect compositor nodes: cuda_rect_overlay (unclocked, timestamp matching),
// mixer_compositor (the mixer's clocked scenes, wipe and AUX buses) and mixer_keyer (the DSK).
#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../avbuffer.hpp"
#include "../../SharedTimeline.hpp"
#include "../../mixer/primitives/compositor_layers.hpp"
#include "../../mixer/primitives/frame_subscription.hpp"
#include "../../mixer/primitives/source_mask.hpp"
#include "cuda_rect_draw.hpp"
#include "cuda_rect_texture.h"

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
/// `metadata_key` layers, `active_inputs`, timeline), input checks and the drawing of one output
/// frame. Layer geometry is resolved by compositor_layers.hpp and the CUDA work is done by
/// CudaRectDraw.
class CudaRectCompositor : public NodeMultiInput<av::VideoFrame>,
                           public NodeSingleOutput<av::VideoFrame>,
                           public IVideoFormatSource,
                           public IFrameRateSource,
                           public TimelineReader,
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

    std::string last_ops_desc_;
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

    std::vector<LayerSpec> mergeLayersForTick(const av::VideoFrame *metadata_source) {
        std::vector<LayerSpec> layers;
        {
            std::lock_guard<std::mutex> lock(layers_mutex_);
            layers = default_layers_;
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

    /// Draws the output frame stamped `pts` and waits for the GPU. `sources` holds a frame per
    /// input, null where none is drawn; `metadata_src` gives the frame its properties and
    /// per-frame layers; `opacity`, when given, holds a weight per input (see LayerSpec::opacity).
    av::VideoFrame compose(av::Timestamp pts, const std::vector<const av::VideoFrame *> &sources,
                           const av::VideoFrame *metadata_src, const std::vector<float> *opacity = nullptr) {
        draw_.ensureDevice();
        CUstream stream = draw_.stream();

        av::VideoFrame outf;
        int r = av_hwframe_get_buffer(out_frames_ref_.get(), outf.raw(), 0);
        if (r < 0)
            throw Error(std::string(type_) + ": av_hwframe_get_buffer failed: " + av::error2string(r));
        outf.setComplete(true);   // av_hwframe_get_buffer set hw_frames_ctx, format and size

        std::vector<LayerSpec> layers = mergeLayersForTick(metadata_src);
        if (opacity)
            for (size_t i = 0; i < layers.size(); ++i) {
                const size_t input = layers[i].sourceIndex(i);
                if (input < opacity->size()) layers[i].opacity = (*opacity)[input];
            }
        const CudaRectDraw::Canvas &cv = draw_.canvas();
        std::vector<DrawOp> ops = avp::mixer::resolveDrawOps(sources, layers, cv.width, cv.height, cv.sw_fmt);
        {
            // Log the resolved layer set whenever it changes (scene switches), not per tick.
            std::string desc = avp::mixer::describeDrawOps(ops);
            if (desc != last_ops_desc_) {
                last_ops_desc_ = std::move(desc);
                logstream << type_ << " layers:" << last_ops_desc_;
            }
        }
        setCanvasColor(outf.raw());
        // Background and every layer in one kernel launch.
        draw_.draw(stream, ops, outf.raw(), hasCanvasColor() ? outf.raw() : (metadata_src ? metadata_src->raw() : nullptr));

        if (metadata_src && metadata_src->raw()) {
            const int cpy = av_frame_copy_props(outf.raw(), metadata_src->raw());
            if (cpy < 0)
                throw Error(std::string(type_) + ": av_frame_copy_props failed: " + av::error2string(cpy));
            // The output owns a new canvas; an imported source texture describes
            // only that source's storage, not these rendered pixels.
            if (avp::mixer::textureFrameDesc(outf.raw()))
                av_buffer_unref(&outf.raw()->opaque_ref);
        }
        setCanvasColor(outf.raw());
        if (hasCanvasColor())
            av_frame_side_data_remove_by_props(&outf.raw()->side_data, &outf.raw()->nb_side_data,
                                               AV_SIDE_DATA_PROP_COLOR_DEPENDENT);
        outf.setPts(pts);

        if (AVP_CHECK_CU(cuStreamSynchronize(stream)))
            throw Error(std::string(type_) + ": cuStreamSynchronize failed");

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0)
            logstream << type_ << ": out frame=" << frame_counter_;

        ++frame_counter_;
        return outf;
    }

    /// Wires a created node into the graph (`src` edges, the output edge, the timeline) and reads
    /// `active_inputs` and `warmup_timeout_ms`.
    void connect(NodeCreationInfo &nci) {
        createSourcesFromParameters(nci.edges, nci.params);
        output_edge_->setProducer(this->shared_from_this());
        initTimeline(nci);
        if (nci.params.count("active_inputs"))
            active_inputs_ = avp::mixer::parseSourceMask(nci.params["active_inputs"]);
        warmup_timeout_ms_ = nci.params.value("warmup_timeout_ms", (int64_t)0);
    }

    /// Subscribes each input to the shared source `subscriptions` names in `src` order, paced at
    /// `fps`. With `allow_unsubscribed`, an empty name leaves that input an ordinary edge.
    void subscribe(NodeCreationInfo &nci, av::Rational fps, bool allow_unsubscribed) {
        const auto names = jsonToStringList(nci.params.at("subscriptions"));
        if (names.size() != source_edges_.size())
            throw Error(std::string(type_) + ": subscriptions must match inputs");
        for (const auto &name : names) {
            if (name.empty() && allow_unsubscribed) {
                subscriptions_.push_back(nullptr);
                continue;
            }
            auto subscription = InstanceSharedObjects<avp::mixer::FrameSubscription>::get(nci.instance, name);
            subscription->configure(fps);
            subscriptions_.push_back(subscription);
        }
    }

public:
    CudaRectCompositor(const char *type, const Config &config)
        : NodeSingleOutput<av::VideoFrame>(make_unique<EdgeSink<av::VideoFrame>>(config.output)),
          type_(type),
          output_edge_(config.output),
          draw_(config.hw, config.canvas, config.max_layers),
          default_layers_(config.layers),
          metadata_key_(config.metadata_key),
          debug_log_every_n_(config.debug_log_every_n) {
        out_frames_ref_.reset(av_hwframe_ctx_alloc(config.hw->deviceContext()));
        if (!out_frames_ref_)
            throw Error(std::string(type_) + ": av_hwframe_ctx_alloc failed");
        AVHWFramesContext *fc = (AVHWFramesContext *)out_frames_ref_->data;
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = config.canvas.sw_fmt;
        fc->width = config.canvas.width;
        fc->height = config.canvas.height;
        int err = av_hwframe_ctx_init(out_frames_ref_.get());
        if (err < 0)
            throw Error(std::string(type_) + ": av_hwframe_ctx_init (output) failed: " + av::error2string(err));
        this->auto_eof_ = false;
    }

    ~CudaRectCompositor() override {
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

    CudaRectDraw::Canvas &canvas = config.canvas;
    canvas.width = params.at("width").get<int>();
    canvas.height = params.at("height").get<int>();
    if (canvas.width <= 0 || canvas.height <= 0)
        throw Error(prefix + "width and height must be positive");
    const std::string sw_name = params.value("sw_format", std::string("nv12"));
    canvas.sw_fmt = av_get_pix_fmt(sw_name.c_str());
    if (canvas.sw_fmt == AV_PIX_FMT_NONE)
        throw Error(prefix + "unknown sw_format");
    if (!CudaRectDraw::canvasSupported(canvas.sw_fmt))
        throw Error(prefix + "sw_format must be semiplanar YUV (nv12, p010le, p210le) or packed 8-bit RGB");

    config.output = nci.edges.find<av::VideoFrame>(params["dst"]);
    config.metadata_key = params.value("metadata_key", std::string("rect_overlay_v1"));
    config.debug_log_every_n = params.value("debug_log_every_n", 0);

    if (params.contains("color")) {
        const auto color = params.at("color").get<std::string>();
        canvas.transfer = avp::mixer::graphicTransfer(color);
        if (canvas.transfer == AVCOL_TRC_UNSPECIFIED)
            throw Error(prefix + "color must be sdr, hlg or pq");
        if (canvas.transfer != AVCOL_TRC_BT709 && av_pix_fmt_desc_get(canvas.sw_fmt)->comp[0].depth < 10)
            throw Error(prefix + "HDR canvas requires 10-bit storage");
        canvas.sdr_white = params.value("sdr_white", avp::mixer::kGraphicSdrWhite);
        canvas.hdr_peak = params.value("hdr_peak", avp::mixer::kGraphicHdrPeak);
        if (!(canvas.sdr_white >= 1.f && canvas.sdr_white <= canvas.hdr_peak &&
              canvas.hdr_peak >= 100.f && canvas.hdr_peak <= 10000.f))
            throw Error(prefix + "invalid display white/peak");
    }
    return config;
}
