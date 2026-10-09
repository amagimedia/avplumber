#include "cuda_overlay_base.hpp"
#include "draw_bbox_items.hpp"
#include "draw_keypoint_items.hpp"
#include "draw_label_items.hpp"
#include "draw_trail_items.hpp"
#include "ml_debug_renderer.hpp"

#include <atomic>
#include <variant>
#include <vector>

#include "../../../../objs/src/nodes/neural_net/draw/ml_debug.ptx.h"

using namespace cuda_overlay;

// Everything a model says about a frame, drawn in one pass: `layers` are painted in the order
// given, each read from frame metadata by the reader of the draw node of its kind (draw_bbox,
// draw_bbox_labels, draw_keypoints, draw_trail) with that node's parameters. The frame comes out
// as it would from a chain of those nodes in the same order, for one kernel launch that visits
// only the 16x16-pixel tiles something is drawn in.
class MlDebug : public CudaOverlayBase {
    using Layer = std::variant<BBoxItems, LabelItems, KeypointItems, TrailItems>;

    std::vector<Layer> layers_;
    MlDebugRenderer renderer_;
    CUfunction kernel_ = nullptr;
    // Scratch of one frame, kept for its capacity.
    std::vector<BatchedBBox> boxes_;
    std::vector<KeypointPos> points_;
    std::vector<LineSegment> segments_;
    // What the last frame drew; see getObject.
    std::atomic<uint64_t> last_items_{0}, last_tiles_{0}, drawn_frames_{0};

    const char* nodeName() const override { return "ml_debug"; }

    void onKernelsUnloaded() override {
        kernel_ = nullptr;
        renderer_.release(cu_ctx_);
    }

    void collect(BBoxItems& layer, FramePayloads& payloads) {
        boxes_.clear();
        layer.collect(payloads, geometry(), frame_counter_, boxes_);
        for (const BatchedBBox& box : boxes_) renderer_.addBox(box);
    }

    void collect(LabelItems& layer, FramePayloads& payloads) {
        const size_t first = renderer_.labels().size();
        layer.collect(payloads, geometry(), frame_counter_, renderer_.labels(), renderer_.text());
        renderer_.addLabels(first);
    }

    void collect(KeypointItems& layer, FramePayloads& payloads) {
        points_.clear();
        layer.collect(payloads, geometry(), frame_counter_, points_);
        for (const KeypointPos& point : points_) renderer_.addDot(point, layer.radius(), layer.color());
    }

    void collect(TrailItems& layer, FramePayloads& payloads) {
        segments_.clear();
        layer.collect(payloads, geometry(), frame_counter_, segments_);
        for (const LineSegment& segment : segments_) renderer_.addSegment(segment, layer.thickness(), layer.color());
    }

    void drawOnFrame(const av::VideoFrame& input, av::VideoFrame& output) override {
        if (!kernel_ && !loadKernel(avpl_ml_debug_ptx, avpl_ml_debug_ptx_len, "kMlDebugNV12", kernel_)) {
            kernel_ = nullptr;
            throw Error("ml_debug: failed to initialize the CUDA kernel");
        }

        // A metadata key that several layers read is parsed once.
        FramePayloads payloads(input.raw());
        renderer_.begin(output.width(), output.height());
        for (Layer& layer : layers_) {
            std::visit([&](auto& items) { collect(items, payloads); }, layer);
        }
        if (!renderer_.draw(cu_ctx_, cuda_dev_ctx_->stream, kernel_, output.raw(), nodeName())) {
            throw Error("ml_debug: failed to draw");
        }
        last_items_ = renderer_.itemCount();
        last_tiles_ = renderer_.tileCount();
        if (renderer_.tileCount()) ++drawn_frames_;
    }

public:
    MlDebug(std::unique_ptr<Source<av::VideoFrame>>&& source,
            std::unique_ptr<Sink<av::VideoFrame>>&& sink,
            std::vector<Layer> layers,
            VideoParameters input_params,
            av::Rational frame_rate,
            av::Rational timebase)
        : CudaOverlayBase(std::move(source), std::move(sink)),
          layers_(std::move(layers)) {
        input_params_ = input_params;
        frame_rate_ = frame_rate;
        timebase_ = timebase;
    }

    ~MlDebug() override {
        renderer_.release(cu_ctx_);
    }

    /// `pictures` as every draw node; `draw`: the items and tiles of the last frame and how many
    /// frames had anything drawn on them.
    Parameters getObject(const std::string name) override {
        if (name == "draw") {
            return {{"items", last_items_.load()}, {"tiles", last_tiles_.load()}, {"frames", drawn_frames_.load()}};
        }
        return CudaOverlayBase::getObject(name);
    }

    static std::shared_ptr<MlDebug> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;

        auto src_edge = edges.find<av::VideoFrame>(params["src"]);
        const auto upstream = resolveUpstreamInfo(src_edge, params);

        if (!params.count("layers") || !params["layers"].is_array() || params["layers"].empty()) {
            throw Error("ml_debug: layers must be a non-empty list of {\"kind\": ..., ...}");
        }
        std::vector<Layer> layers;
        for (const auto& layer : params["layers"]) {
            if (!layer.is_object() || !layer.contains("kind") || !layer["kind"].is_string()) {
                throw Error("ml_debug: every layer needs a kind: boxes, labels, keypoints or trail");
            }
            const std::string kind = layer["kind"].get<std::string>();
            const std::string name = "ml_debug " + kind + " layer " + std::to_string(layers.size());
            if (kind == "boxes") layers.emplace_back(BBoxItems::fromParams(layer, name));
            else if (kind == "labels") layers.emplace_back(LabelItems::fromParams(layer, name));
            else if (kind == "keypoints") layers.emplace_back(KeypointItems::fromParams(layer, name));
            else if (kind == "trail") layers.emplace_back(TrailItems::fromParams(layer, name));
            else throw Error("ml_debug: unknown layer kind " + kind + ": boxes, labels, keypoints or trail");
        }

        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MlDebug>(
            edges, params, std::move(layers), upstream.input_params, upstream.frame_rate, upstream.timebase);
    }
};

DECLNODE(ml_debug, MlDebug)
