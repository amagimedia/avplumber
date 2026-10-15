#pragma once
#include "node_common.hpp"
#include "../output_mask.hpp"
#include "../output_subscriptions.hpp"
#include "../mixer/primitives/frame_subscription.hpp"
#include "../frame_color_tags.hpp"
#include "../mixer/primitives/SourceDelivery.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"

inline uint32_t parseOutputsMask(const Parameters& value) { return parseBitmask(value); }

// `frame_color` applies to video frames; any other data passes as it is.
template <typename T> void stampFrameColor(T&, const FrameColorTags&) {}
inline void stampFrameColor(av::VideoFrame& frame, const FrameColorTags& tags) { tags.apply(frame.raw()); }

// One input handed to several outputs: what both nodes below do with a frame, and their
// parameters and objects. How they come by the frame is theirs.
template <typename T>
class OneToManyBase : public NodeSingleInput<T>, public NodeMultiOutput<T>,
                      public IInputsObjects, public IReturnsObjects,
                      public IOutputSubscriptions {
protected:
    std::atomic<uint32_t> outputs_mask_{1};
    bool drop_ = false;
    std::atomic<int64_t> enable_from_ms_{0};
    FrameColorTags frame_color_;   // written on every frame before it is handed out
    std::vector<std::shared_ptr<avp::mixer::FrameSubscription>> subscriptions_;
    std::vector<std::string> output_names_;
    std::string source_identity_;
    av::Rational source_rate_{0, 1};
    uint64_t source_epoch_ = avp::mixer::monotonicNs();
    int64_t source_number_ = -1;

    void stampSource(T* data) {
        if constexpr (std::is_same_v<T, av::VideoFrame>) {
            if (source_identity_.empty() || !data->isValid() || !data->pts().isValid()) return;
            const auto key = std::string(avp::mixer::sourcePrefix) + source_identity_;
            if (const auto* capture = av_dict_get(data->raw()->metadata, avp::mixer::captureKey, nullptr, 0)) {
                avp::mixer::setFrameTag(data->raw(), key.c_str(), capture->value);
                return;
            }
            if (source_rate_.getNumerator() <= 0) {
                auto rate = this->edgeSource()->edge()->template findNodeUp<IFrameRateSource>();
                if (!rate) return;
                source_rate_ = rate->frameRate();
            }
            if (source_rate_.getNumerator() <= 0) return;
            const int64_t number = data->pts().timestamp(av_inv_q(source_rate_.getValue()));
            if (number < 0) return;
            if (number < source_number_) source_epoch_ = avp::mixer::monotonicNs();
            source_number_ = number;
            const auto period = av_rescale_q(1, av_inv_q(source_rate_.getValue()), AVRational{1, 1000000000});
            avp::mixer::setFrameTag(data->raw(), key.c_str(),
                avp::mixer::SourceFrame{source_epoch_, uint64_t(number), uint64_t(period)}.encode());
        }
    }

public:
    explicit OneToManyBase(std::unique_ptr<Source<T>> &&source): NodeSingleInput<T>(std::move(source)) {}

    std::map<std::string, bool> outputSubscriptions() const override {
        std::map<std::string, bool> result;
        for (size_t i = 0; i < subscriptions_.size(); ++i)
            if (subscriptions_[i]) result.emplace(output_names_[i], subscriptions_[i]->enabled());
        return result;
    }

protected:
    void handOut(T* data) {
        if (frame_color_.any()) stampFrameColor(*data, frame_color_);
        stampSource(data);

        uint32_t mask = outputs_mask_.load(std::memory_order_relaxed);
        const auto from = enable_from_ms_.load(std::memory_order_acquire);
        if (from && (!data->pts().isValid() || data->pts() < av::Timestamp(from, {1, 1000}))) mask = 0;
        bool drop = drop_;
        for (size_t i = 0; i < this->sink_edges_.size(); i++) {
            if (subscriptions_[i])
                subscriptions_[i]->publish(*data, [&] { EdgeSink<T>(this->sink_edges_[i]).put(*data, true); });
            else if (mask & (1u << i))
                EdgeSink<T>(this->sink_edges_[i]).put(*data, drop);
        }
    }

public:
    void setObject(const std::string key, const Parameters& value) override {
        if (key == "outputs")
            outputs_mask_.store(parseOutputsMask(value), std::memory_order_relaxed);
        else if (key == "enable_from") {
            enable_from_ms_.store(value.get<int64_t>(), std::memory_order_release);
            outputs_mask_.store(1, std::memory_order_release);
        }
    }

    Parameters getObject(const std::string key) override {
        if (key == "outputs")
            return Parameters(outputs_mask_.load(std::memory_order_relaxed));
        throw Error("one_to_many: unknown object key: " + key);
    }

    virtual void init(EdgeManager& edges, const Parameters& params) override {
        NodeSingleInput<T>::init(edges, params);
    }

protected:
    template <typename Self> static std::shared_ptr<Self> build(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<Self>(make_unique<EdgeSource<T>>(in_edge));
        r->createSinksFromParameters(edges, params);
        const auto names = jsonToStringList(params.at("dst"));
        r->output_names_.assign(names.begin(), names.end());
        r->source_identity_ = params.value("source_identity", std::string());
        if (r->sink_edges_.size() > 32) throw Error("one_to_many: at most 32 destinations");
        r->subscriptions_.resize(r->sink_edges_.size());
        if (params.contains("subscribed_outputs")) {
            size_t i = 0;
            for (const auto &name : names) {
                if (params.at("subscribed_outputs").contains(name))
                    r->subscriptions_[i] = InstanceSharedObjects<avp::mixer::FrameSubscription>::get(
                        nci.instance, params.at("subscribed_outputs").at(name).get<std::string>());
                ++i;
            }
        }
        in_edge->setConsumer(r);
        if (params.count("outputs"))
            r->outputs_mask_.store(parseOutputsMask(params["outputs"]), std::memory_order_relaxed);
        if (params.count("drop"))
            r->drop_ = params["drop"].get<bool>();
        if (params.count("frame_color"))
            r->frame_color_ = FrameColorTags::parse(params["frame_color"]);
        return r;
    }
};
