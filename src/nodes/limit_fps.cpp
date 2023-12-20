#include "node_common.hpp"

class FPSLimiter: public NodeSISO<av::VideoFrame, av::VideoFrame>, public NodeDoesNotBuffer {
protected:
    double min_delta_ = 0;
    av::Timestamp prev_pts_ = NOTS;
public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    virtual void process() {
        av::VideoFrame frame = this->source_->get();
        do {
            if (prev_pts_.isNoPts()) {
                break;
            }
            if (frame.pts().isNoPts()) {
                logstream << "got NOPTS, dropping";
                return;
            }
            av::Timestamp diff = addTS(frame.pts(), negateTS(prev_pts_));
            if (diff.timestamp() < 0) {
                logstream << "PTS went backwards";
                break;
            }
            if (diff.seconds() < min_delta_) {
                return;
            }
        } while(false);
        prev_pts_ = frame.pts();
        this->sink_->put(frame);
    }
    static std::shared_ptr<FPSLimiter> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<FPSLimiter> r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<FPSLimiter>(edges, params);
        r->min_delta_ = 1.0f/params["max_fps"].get<double>();
        return r;
    }
};

DECLNODE(limit_fps, FPSLimiter);
