#include "../node_common.hpp"
#include "../../mixer/primitives/compositor_color.hpp"
#include "cuda_rect_frame.hpp"

extern "C" {
#include <libavutil/mathematics.h>
}

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// One frame in, one frame per output out. Each output is a canvas of its own size whose layers
/// crop, scale and place the input frame; where no layer draws, the canvas keeps its background.
/// It is the rect compositor's drawing without input matching or a clock: every input frame
/// gives one frame on each output that is due, with the input's timestamp and metadata.
///
/// An output whose canvas would only repeat the input (one unblended layer, the whole frame at
/// 1:1, the canvas's own sw_format) gets the input frame itself: nothing is drawn or copied. This
/// is decided per frame, so a source that changes size is scaled when it has to be. A CUarray
/// input is a different kind of storage from the CUDA frames the node draws; it is passed on only
/// with `pass_arrays`, for consumers that read arrays as well.
///
/// Parameters: `src`, `hwaccel`, and `outputs`, an array of
///   { "dst": edge or [edges], "width", "height", "sw_format" (nv12), "layers": [...],
///     "fps": rate (optional), "drop": bool (optional), "pass_arrays": bool (optional),
///     "allow_hdr_depth_reduction": bool (optional) }
/// `layers` are the compositor's layer objects (crop, dst rect, fit); all read the one input.
/// A layer's `filter` defaults to `auto` here (the compositor nodes default to `bilinear`):
/// bicubic above 1.3x enlarging, multisample above 2x shrinking, bilinear between.
/// With `fps` the output takes the first frame of each 1/fps slot of the input's timestamps, so
/// the choice of frames follows timestamps and not arrival order. The same frame goes to every
/// edge of an output. `drop` discards an output's frame when its edge is full instead of waiting.
/// A frame tagged PQ or HLG is an error on an 8-bit output, drawn or passed on, whatever its own
/// depth: the output would hold HDR-coded pixels that read as SDR. The error text carries
/// `hdr_source_transfer=pq` or `=hlg`; tone map before the node. `allow_hdr_depth_reduction`
/// takes the frame as it is and keeps its tags, for sources that are 8-bit HDR by intent.
///
/// When all outputs have one canvas (the same width, height and sw_format; one output always
/// does), the node declares it as its video format (IVideoFormatSource). A node below that asks
/// for the stream's size (a router, an encoder, a filter's first init) then gets the canvas and
/// not the size of the input above. The declaration is one per node, not one per edge: when the
/// canvases differ the node cannot answer for one edge and the question fails. An encoder or a
/// router below such an output needs an `assume_video_format` before it, as below every output
/// before; `filter_video` takes the format from its first frame instead. A CUarray passed on
/// with `pass_arrays` keeps its own pixel format. Frame rate and time base are not declared:
/// readers find them above this node, which is right only for an output without `fps`.
class CudaTransform : public NodeSingleInput<av::VideoFrame>,
                      public NodeMultiOutput<av::VideoFrame>,
                      public NodeDoesNotBuffer,   // no frame is kept between process() calls: a stop only has to end the loop
                      public IVideoFormatSource {
    using CudaRectDraw = avp::mixer::CudaRectDraw;
    static constexpr const char *kType = "cuda_transform";

    struct Output {
        std::unique_ptr<CudaRectDraw> draw;
        avp::AvBufferRef frames;
        std::vector<avp::mixer::LayerSpec> layers;
        std::vector<std::shared_ptr<Edge<av::VideoFrame>>> edges;
        AVRational slot = {0, 1};   // 1/fps; num 0: every frame
        int64_t last_slot = INT64_MIN;
        bool drop = false;
        bool pass_arrays = false;
        bool passing = false;   // the last frame was passed on, not drawn; logged when it changes
    };
    std::vector<Output> outputs_;

    /// Whether `out` takes the frame stamped `pts`. Slots are shifted by a quarter so a frame
    /// whose timestamp was rounded just below a slot boundary still opens that slot.
    static bool due(Output &out, const av::Timestamp &pts) {
        if (out.slot.num == 0 || !pts.isValid())
            return true;
        const int64_t quarters = av_rescale_q_rnd(pts.timestamp(), pts.timebase().getValue(),
                                                  AVRational{out.slot.num, out.slot.den * 4}, AV_ROUND_DOWN);
        const int64_t slot = (quarters + 1) >> 2;
        if (slot == out.last_slot)
            return false;
        out.last_slot = slot;
        return true;
    }

public:
    void onEofConsumed() override {
        av::VideoFrame eof = createEofMarker<av::VideoFrame>();
        for (auto &edge : this->sink_edges_)
            edge->enqueue(eof);
    }

    using NodeSingleInput<av::VideoFrame>::NodeSingleInput;

    /// The one canvas of all outputs, which is what the node can declare for any of its edges.
    /// create() rejects an empty `outputs`.
    const CudaRectDraw::Canvas &commonCanvas() const {
        const CudaRectDraw::Canvas &first = outputs_.front().draw->canvas();
        for (const Output &out : outputs_) {
            const CudaRectDraw::Canvas &cv = out.draw->canvas();
            if (cv.width != first.width || cv.height != first.height || cv.sw_fmt != first.sw_fmt)
                throw Error(std::string(kType) + ": outputs differ in size or sw_format, so the node declares "
                            "no video format; put assume_video_format between the output and its reader");
        }
        return first;
    }
    int width() override { return commonCanvas().width; }
    int height() override { return commonCanvas().height; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(commonCanvas().sw_fmt); }

    ~CudaTransform() override {
        for (auto &out : outputs_) {
            out.draw->unload();
            out.frames.reset();
        }
    }

    void init(EdgeManager &edges, const Parameters &params) override {
        for (auto &out : outputs_)
            out.draw->ensureKernels();   // fail at graph build, not on the first frame
        NodeSingleInput<av::VideoFrame>::init(edges, params);
    }

    void process() override {
        av::VideoFrame *in = this->source_->peek();
        if (in == nullptr)
            return;
        if (isEofMarker(*in)) {
            this->source_->pop();
            onEofConsumed();
            this->finished_ = true;
            return;
        }
        if (!CudaRectDraw::frameSupported(static_cast<AVPixelFormat>(in->raw()->format)))
            throw Error(std::string(kType) + ": input must be a CUDA device frame or CUarray");
        const AVPixelFormat in_fmt = CudaRectDraw::frameSwFormat(*in);

        const av::Timestamp pts = in->pts();
        const std::vector<const av::VideoFrame *> sources{in};
        std::vector<std::pair<Output *, av::VideoFrame>> drawn;
        CUstream stream = nullptr;
        for (auto &out : outputs_) {
            if (!due(out, pts))
                continue;
            const CudaRectDraw::Canvas &cv = out.draw->canvas();
            if (in_fmt == AV_PIX_FMT_NONE || !avp::mixer::canvasAccepts(in_fmt, cv.sw_fmt))
                throw Error(std::string(kType) + ": input hw sw_format mismatch output sw_format");
            if (!cv.allow_hdr_depth_reduction)
                if (const char *hdr = avp::mixer::hdrOnEightBitCanvas(in->raw()->color_trc, cv.sw_fmt))
                    throw Error(avp::mixer::hdrOnEightBitError(kType, hdr) +
                                ", or set allow_hdr_depth_reduction on the output");
            const auto ops = avp::mixer::resolveDrawOps(sources, out.layers, cv.width, cv.height, cv.sw_fmt);
            const bool pass = in_fmt == cv.sw_fmt && avp::mixer::copiesWholeFrame(ops, cv.width, cv.height) &&
                              (in->raw()->format == AV_PIX_FMT_CUDA || out.pass_arrays);
            if (pass != out.passing) {
                out.passing = pass;
                logstream << kType << ": output " << (&out - outputs_.data())
                          << (pass ? " passes the input frame on" : " draws the input frame");
            }
            if (pass) {
                drawn.emplace_back(&out, *in);   // a new reference to the same buffers
                continue;
            }
            if (!stream) {
                out.draw->ensureDevice();
                stream = out.draw->stream();
            }
            av::VideoFrame outf = avp::mixer::canvasFrame(out.frames.get(), kType);
            out.draw->draw(stream, ops, outf.raw(), in->raw());
            avp::mixer::copySourceProps(outf, *in, kType);
            outf.setPts(pts);
            drawn.emplace_back(&out, std::move(outf));
        }
        if (stream && AVP_CHECK_CU(cuStreamSynchronize(stream)))
            throw Error(std::string(kType) + ": cuStreamSynchronize failed");

        // Release the input before a put can wait: it may be a decoder surface.
        this->source_->pop();
        for (auto &item : drawn)
            for (auto &edge : item.first->edges)
                EdgeSink<av::VideoFrame>(edge).put(item.second, item.first->drop);
    }

    static std::shared_ptr<CudaTransform> create(NodeCreationInfo &nci) {
        const Parameters &params = nci.params;
        const std::string prefix = std::string(kType) + ": ";
        if (!params.contains("hwaccel"))
            throw Error(prefix + "hwaccel parameter required");
        auto hw = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!hw)
            throw Error(prefix + "failed to resolve hwaccel");
        if (!params.contains("outputs") || !params["outputs"].is_array() || params["outputs"].empty())
            throw Error(prefix + "outputs array required");

        auto in_edge = nci.edges.find<av::VideoFrame>(params["src"]);
        auto node = std::make_shared<CudaTransform>(make_unique<EdgeSource<av::VideoFrame>>(in_edge));
        for (const Parameters &spec : params["outputs"]) {
            Output out;
            CudaRectDraw::Canvas canvas = avp::mixer::parseCanvas(spec, prefix);
            if (canvas.transfer != AVCOL_TRC_UNSPECIFIED)
                throw Error(prefix + "color is not supported on an output");
            canvas.allow_hdr_depth_reduction = spec.value("allow_hdr_depth_reduction", false);
            out.layers = avp::mixer::parseLayersParam(spec, avp::mixer::ScaleFilter::Auto);
            if (out.layers.empty())
                throw Error(prefix + "an output needs at least one layer");
            for (auto &layer : out.layers) {
                if (layer.input > 0)
                    throw Error(prefix + "a layer can only read input 0");
                layer.input = 0;
            }
            out.draw = std::make_unique<CudaRectDraw>(hw, canvas, int(out.layers.size()));
            out.frames = avp::mixer::allocCanvasFrames(*hw, canvas, kType);
            if (spec.contains("fps")) {
                const av::Rational fps = parseRatio(spec.at("fps").get<std::string>());
                if (fps.getNumerator() <= 0 || fps.getDenominator() <= 0)
                    throw Error(prefix + "fps must be positive");
                out.slot = AVRational{fps.getDenominator(), fps.getNumerator()};
            }
            out.drop = spec.value("drop", false);
            out.pass_arrays = spec.value("pass_arrays", false);
            for (const std::string &name : jsonToStringList(spec.at("dst"))) {
                auto edge = nci.edges.find<av::VideoFrame>(name);
                edge->setProducer(node);
                out.edges.push_back(edge);
                node->sink_edges_.push_back(edge);
            }
            if (out.edges.empty())
                throw Error(prefix + "an output needs a dst edge");
            node->outputs_.push_back(std::move(out));
        }
        in_edge->setConsumer(node);
        return node;
    }
};

DECLNODE(cuda_transform, CudaTransform)
