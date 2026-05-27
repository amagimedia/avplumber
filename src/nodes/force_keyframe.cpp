#include "node_common.hpp"
#include <atomic>

class ForceKeyFrame: public NodeSISO<av::VideoFrame, av::VideoFrame>,
                     public IInputsObjects,
                     public IReturnsObjects {
protected:
    av::Rational interval_sec_;
    bool interval_enabled_ = false;
    int64_t last_result_ = -(1L<<62);
    std::atomic<uint64_t> requested_generation_{0};
    std::atomic<uint64_t> forced_generation_{0};
    std::atomic<uint64_t> triggered_frames_{0};
    std::atomic<uint64_t> periodic_frames_{0};

    static av::Rational parseInterval(const json &param) {
        if (param.is_string()) {
            return parseRatio(param);
        } else if (param.is_number_integer()) {
            return av::Rational(param.get<int>(), 1);
        } else if (param.is_number_float()) {
            return av::Rational(static_cast<int>(param.get<float>() * 1000.0 + 0.5), 1000);
        }
        throw Error("Invalid data type for parameter interval_sec");
    }

    bool shouldForcePeriodic(const av::VideoFrame &frm) {
        if (!interval_enabled_) {
            return false;
        }
        soft_assert(frm.pts().timebase().getNumerator() && frm.pts().timebase().getDenominator(), "invalid timebase in frame");
        int result = (
            frm.pts().timestamp() *
            frm.pts().timebase().getNumerator() *
            interval_sec_.getDenominator()
        ) / (
            frm.pts().timebase().getDenominator() *
            interval_sec_.getNumerator()
        );
        if (result == last_result_) {
            return false;
        }
        last_result_ = result;
        periodic_frames_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool shouldForceTriggered() {
        const uint64_t requested = requested_generation_.load(std::memory_order_acquire);
        const uint64_t forced = forced_generation_.load(std::memory_order_acquire);
        if (requested == forced) {
            return false;
        }
        forced_generation_.store(requested, std::memory_order_release);
        triggered_frames_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

public:
    virtual void process() {
        av::VideoFrame* ptr = this->source_->peek();
        if (ptr == nullptr) return;
        if (isEofMarker(*ptr)) {
            // Leave the EOF marker in the queue so consumeEofIfPresent() picks it up
            // and correctly terminates the dumb node loop.
            return;
        }
        av::VideoFrame frm = *ptr;
        const bool force_triggered = shouldForceTriggered();
        const bool force_periodic = shouldForcePeriodic(frm);
        if (force_triggered || force_periodic) {
            frm.setPictureType(AV_PICTURE_TYPE_I);
            frm.setKeyFrame(true);
        } else {
            frm.setPictureType(AV_PICTURE_TYPE_NONE);
        }
        this->sink_->put(frm);
        this->source_->pop();
    }
    void setObject(const std::string key, const Parameters& value) override {
        if (key == "trigger" || key == "force" || key == "request") {
            bool enabled = true;
            if (value.is_boolean()) {
                enabled = value.get<bool>();
            } else if (value.is_number_integer()) {
                enabled = value.get<int64_t>() != 0;
            } else if (value.is_object() && value.contains("enable")) {
                enabled = value["enable"].get<bool>();
            }
            if (enabled) {
                requested_generation_.fetch_add(1, std::memory_order_acq_rel);
            }
            return;
        }
        throw Error("force_keyframe: unknown object key: " + key);
    }

    Parameters getObject(const std::string key) override {
        if (key == "status") {
            Parameters status = Parameters::object();
            const uint64_t requested = requested_generation_.load(std::memory_order_acquire);
            const uint64_t forced = forced_generation_.load(std::memory_order_acquire);
            status["requested_generation"] = requested;
            status["forced_generation"] = forced;
            status["pending"] = requested != forced;
            status["triggered_frames"] = triggered_frames_.load(std::memory_order_relaxed);
            status["periodic_frames"] = periodic_frames_.load(std::memory_order_relaxed);
            status["interval_enabled"] = interval_enabled_;
            return status;
        }
        if (key == "pending") {
            return Parameters(
                requested_generation_.load(std::memory_order_acquire) !=
                forced_generation_.load(std::memory_order_acquire)
            );
        }
        throw Error("force_keyframe: unknown object key: " + key);
    }

    ForceKeyFrame(
        std::unique_ptr<Source<av::VideoFrame>> &&source,
        std::unique_ptr<Sink<av::VideoFrame>> &&sink,
        bool interval_enabled,
        const av::Rational interval_sec
    ): NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)),
       interval_sec_(interval_sec),
       interval_enabled_(interval_enabled) {
    }

    static std::shared_ptr<ForceKeyFrame> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        bool interval_enabled = false;
        av::Rational interval_sec(0, 1);
        if (params.count("interval_sec") > 0) {
            interval_sec = parseInterval(params["interval_sec"]);
            if (interval_sec.getNumerator() <= 0 || interval_sec.getDenominator() <= 0) {
                throw Error("interval_sec must be positive");
            }
            interval_enabled = true;
        }
        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ForceKeyFrame>(
            edges,
            params,
            interval_enabled,
            interval_sec
        );
    }
};

DECLNODE(force_keyframe, ForceKeyFrame);
