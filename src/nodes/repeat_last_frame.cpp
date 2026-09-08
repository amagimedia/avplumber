#include "node_common.hpp"
#include "../mixer/MonotonicClock.hpp"

extern "C" {
#include <libavutil/mathematics.h>
}

/// Re-emit the last frame at a fixed rate while the input is idle.
///
/// Browser pages paint only when something changes, so a static page delivers
/// one frame and then nothing; a compositor that resets its inputs on scene
/// switches would then wait forever. This node forwards live frames untouched
/// and, whenever the input has been quiet for a frame period, sends the last
/// frame again with its timestamp moved to the current monotonic time
/// (the clock the DMA-BUF chain stamps its frames with). Copies share the
/// underlying buffer; nothing is duplicated on the GPU.
class RepeatLastFrame: public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    av::Rational fps_{60, 1};
    int64_t period_ns_ = 16666667;
    av::VideoFrame last_;
    bool have_last_ = false;
    int64_t next_repeat_ns_ = 0;
    uint64_t repeated_ = 0;

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        const int64_t now = avp::mixer::monotonicNs();
        int wait_ms = 200;
        if (have_last_)
            wait_ms = std::max<int>(1, int((next_repeat_ns_ - now) / 1000000));
        av::VideoFrame in;
        if (this->source_->tryGet(in, wait_ms)) {
            if (in && in.isComplete() && in.pts().isValid()) {
                last_ = in;
                have_last_ = true;
                // Only step in after a missed period, never between live frames.
                next_repeat_ns_ = avp::mixer::monotonicNs() + period_ns_ + period_ns_ / 2;
            }
            this->sink_->put(in);
            return;
        }
        if (!have_last_) return;
        const int64_t t = avp::mixer::monotonicNs();
        if (t < next_repeat_ns_) return;
        av::VideoFrame out = last_;   // new AVFrame, same buffers
        const av::Rational tb = last_.timeBase();
        out.setTimeBase(tb);
        out.setPts(av::Timestamp(av_rescale_q(t, AVRational{1, 1000000000}, tb.getValue()), tb));
        this->sink_->put(out);
        next_repeat_ns_ += period_ns_;
        if (next_repeat_ns_ < t) next_repeat_ns_ = t + period_ns_;   // do not burst after a long stall
        if (++repeated_ == 1 || repeated_ % 3600 == 0)
            logstream << "repeat_last_frame: repeated " << repeated_ << " frame(s) while the input was idle";
    }

    static std::shared_ptr<RepeatLastFrame> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<RepeatLastFrame>(edges, params);
        if (params.count("fps")) {
            r->fps_ = parseRatio(params["fps"]);
            r->period_ns_ = int64_t(1000000000.0 * r->fps_.getDenominator() / r->fps_.getNumerator());
        }
        return r;
    }
};

DECLNODE(repeat_last_frame, RepeatLastFrame);
