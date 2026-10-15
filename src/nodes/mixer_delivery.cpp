#include "node_common.hpp"
#include "../mixer/primitives/SourceDelivery.hpp"
#include "../mixer/primitives/TickGrid.hpp"

// Observe the published, keyed program. Slot prewarm and off-air sources never reach this node.
class MixerDelivery : public NodeSISO<av::VideoFrame, av::VideoFrame>,
                      public IInputsObjects, public IReturnsObjects, public IFrameRateSource {
    avp::mixer::SourceDelivery delivery_;
    std::mutex mutex_;
    av::Rational fps_{30, 1};
    int64_t period_ns_ = 0;
    bool enabled_ = false;
    std::set<std::string> on_change_;
public:
    using NodeSISO::NodeSISO;
    av::Rational frameRate() override { return fps_; }
    void process() override {
        auto* frame = this->source_->peek(100);
        if (!frame) return;
        // Parse before publication so malformed lineage cannot silently report perfect delivery.
        auto lineage = frame->isValid() ? avp::mixer::sourceFrames(frame->raw()) : avp::mixer::SourceFrames{};
        for (auto& [name, frames] : lineage)
            if (on_change_.count(name)) for (auto& capture : frames) capture.period_ns = 0;
        if (this->sink_->put(*frame, true)) {
            if (frame->isValid() && frame->pts().isValid()) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (enabled_) delivery_.observe(frame->pts().timestamp({1, 1000000000}), period_ns_, lineage);
            }
            this->source_->pop();
        } else this->edgeSink()->edge()->consumedEvent().wait(2);
    }
    Parameters getObject(const std::string key) override {
        if (key != "per_source") throw Error("mixer_delivery: unknown object " + key);
        std::lock_guard<std::mutex> lock(mutex_);
        Parameters result = Parameters::object();
        for (const auto& [name, count] : delivery_.counts())
            result[name] = {{"frames", count.frames}, {"repeats", count.repeats},
                            {"dropped", count.dropped}, {"regressions", count.regressions}, {"static_holds", count.static_holds}};
        return result;
    }
    void setObject(const std::string key, const Parameters& value) override {
        if (key != "enabled") throw Error("mixer_delivery: unknown object " + key);
        std::lock_guard<std::mutex> lock(mutex_);
        const bool enable = value.get<bool>();
        if (enable && !enabled_) delivery_.reset();
        enabled_ = enable;
    }
    static std::shared_ptr<MixerDelivery> create(NodeCreationInfo& nci) {
        auto node = NodeSISO::createCommon<MixerDelivery>(nci.edges, nci.params);
        node->fps_ = parseRatio(nci.params.at("fps"));
        if (nci.params.contains("on_change_sources"))
            for (const auto& name : jsonToStringList(nci.params.at("on_change_sources"))) node->on_change_.insert(name);
        node->period_ns_ = avp::mixer::TickGrid(node->fps_).time(1);
        return node;
    }
};
DECLNODE(mixer_delivery, MixerDelivery)
