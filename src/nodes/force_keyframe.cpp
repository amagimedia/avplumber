#include "node_common.hpp"

class ForceKeyFrame: public NodeSISO<av::VideoFrame, av::VideoFrame> {
protected:
    av::Rational interval_sec_;
    int64_t last_result_ = -(1L<<62);
public:
    virtual void process() {
        av::VideoFrame frm = this->source_->get();
        if (frm.isValid()) {
            soft_assert(frm.pts().timebase().getNumerator() && frm.pts().timebase().getDenominator(), "invalid timebase in frame");
            int result = (frm.pts().timestamp() * frm.pts().timebase().getNumerator() * interval_sec_.getDenominator()) / (frm.pts().timebase().getDenominator() * interval_sec_.getNumerator());
            if (result != last_result_) {
                //logstream << "Forcing key frame " << result << " != " << last_result_;
                frm.setPictureType(AV_PICTURE_TYPE_I);
                frm.setKeyFrame(true);
                last_result_ = result;
            } else {
                frm.setPictureType(AV_PICTURE_TYPE_NONE);
            }
        }
        this->sink_->put(frm);
    }
    ForceKeyFrame(std::unique_ptr<Source<av::VideoFrame>> &&source, std::unique_ptr<Sink<av::VideoFrame>> &&sink, const av::Rational interval_sec): NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)), interval_sec_(interval_sec) {
    }
    static std::shared_ptr<ForceKeyFrame> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        if (params.count("interval_sec") > 0) {
            av::Rational interval_sec;
            const json &param = params["interval_sec"];
            if (param.is_string()) {
                interval_sec = parseRatio(param);
            } else if (param.is_number_integer()) {
                interval_sec = av::Rational(param.get<int>(), 1);
            } else if (param.is_number_float()) {
                interval_sec = av::Rational(static_cast<int>(param.get<float>()*1000.0+0.5), 1000);
            } else {
                throw Error("Invalid data type for parameter interval_sec");
            }
            return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ForceKeyFrame>(edges, params, interval_sec);
        } else {
            throw Error("interval_sec must be specified");
        }
    }
};

DECLNODE(force_keyframe, ForceKeyFrame);
