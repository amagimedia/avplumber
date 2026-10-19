#include "node_common.hpp"
#include "../mixer/primitives/HoldNumbering.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"

/// Re-emit the last frame at a fixed rate while the input is idle.
///
/// Browser pages paint only when something changes, so a static page delivers
/// one frame and then nothing; a compositor that resets its inputs on scene
/// switches would then wait forever. This node forwards live frames untouched
/// and, whenever the input has been quiet for a frame period, sends the last
/// frame again, stamped one frame after the last stamp it put out (the last
/// live frame's own stamp, then counting), so the copies continue the live
/// numbering instead of taking the node's emission time, which rounded into
/// the neighbouring slot whenever the train's phase sat near the half slot.
/// The schedule, the stall rule and the stamps live in HoldNumbering. Copies
/// share the underlying buffer; nothing is duplicated on the GPU.
class RepeatLastFrame: public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    avp::mixer::HoldNumbering hold_{AVRational{60, 1}};
    av::VideoFrame last_;
    uint64_t repeated_ = 0;
    uint64_t realigned_logged_ = 0;

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        const int64_t now = avp::mixer::monotonicNs();
        int wait_ms = 200;
        if (hold_.armed())
            wait_ms = std::max<int>(1, int((hold_.nextRepeatNs() - now) / 1000000));
        av::VideoFrame in;
        if (this->source_->tryGet(in, wait_ms)) {
            if (in && in.isComplete() && in.pts().isValid()) {
                last_ = in;
                hold_.live(avp::mixer::monotonicNs(), in.pts().timestamp(), in.timeBase().getValue());
            }
            this->sink_->put(in);
            return;
        }
        const int64_t t = avp::mixer::monotonicNs();
        if (!hold_.due(t)) return;
        av::VideoFrame out = last_;   // new AVFrame, same buffers
        const av::Rational tb = last_.timeBase();
        out.setTimeBase(tb);
        out.setPts(av::Timestamp(hold_.copy(t), tb));
        this->sink_->put(out);
        if (++repeated_ == 1 || repeated_ % 3600 == 0)
            logstream << "repeat_last_frame: repeated " << repeated_ << " frame(s) while the input was idle";
        const uint64_t realigned = hold_.realignments();
        if (realigned != realigned_logged_ && (realigned == 1 || realigned % 100 == 0))
            logstream << "repeat_last_frame: woken more than a period late " << realigned
                      << " time(s); the copy was re-aligned to the next slot";
        realigned_logged_ = realigned;
    }

    static std::shared_ptr<RepeatLastFrame> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::createCommon<RepeatLastFrame>(edges, params);
        if (params.count("fps"))
            r->hold_ = avp::mixer::HoldNumbering(parseRatio(params["fps"]).getValue());
        return r;
    }
};

DECLNODE(repeat_last_frame, RepeatLastFrame);
