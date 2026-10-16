#include "node_common.hpp"

// Forwards frames in increasing pts order without blocking its producer: a frame whose pts does
// not advance is dropped, and while the output edge is full the frame waits here, 2 ms at a
// time. The program slots and the program output each pass through one, which also gives the
// nodes after it the canvas rate and time base.
class MixerPtsGate : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                     public IFrameRateSource, public ITimeBaseSource {
    av::Rational fps_;
    av::Timestamp last_pts_ = NOTS;

public:
    MixerPtsGate(std::unique_ptr<SourceType>&& source, std::unique_ptr<SinkType>&& sink, av::Rational fps)
        : NodeSISO(std::move(source), std::move(sink)), fps_(fps) {}
    av::Rational frameRate() override { return fps_; }
    av::Rational timeBase() override { return av_inv_q(fps_.getValue()); }

    void process() override {
        auto* input = this->source_->peek(100);
        if (!input) return;
        av::VideoFrame output = *input;
        if (output.isValid() && last_pts_.isValid() && output.pts().isValid() && output.pts() <= last_pts_) {
            this->source_->pop();
            return;
        }
        if (this->sink_->put(output, true)) {
            this->source_->pop();
            if (output.isValid() && output.pts().isValid()) last_pts_ = output.pts();
        } else {
            this->edgeSink()->edge()->consumedEvent().wait(2);
        }
    }

    static std::shared_ptr<MixerPtsGate> create(NodeCreationInfo& nci) {
        const auto fps = parseRatio(nci.params.value("fps", std::string("60/1")));
        return NodeSISO::createCommon<MixerPtsGate>(nci.edges, nci.params, fps);
    }
};

DECLNODE(mixer_pts_gate, MixerPtsGate)
