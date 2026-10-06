#include "node_common.hpp"
#include "../output_mask.hpp"
#include "../output_subscriptions.hpp"
#include "../mixer/primitives/frame_subscription.hpp"

static uint32_t parseOutputsMask(const Parameters& value) { return parseBitmask(value); }

template <typename T>
class OneToMany : public NodeSingleInput<T>, public NodeMultiOutput<T>,
                  public IInputsObjects, public IReturnsObjects,
                  public IOutputSubscriptions {
    std::atomic<uint32_t> outputs_mask_{1};
    bool drop_ = false;
    std::atomic<int64_t> enable_from_ms_{0};
    std::vector<std::shared_ptr<avp::mixer::FrameSubscription>> subscriptions_;
    std::vector<std::string> output_names_;

public:
    using NodeSingleInput<T>::NodeSingleInput;

    std::map<std::string, bool> outputSubscriptions() const override {
        std::map<std::string, bool> result;
        for (size_t i = 0; i < subscriptions_.size(); ++i)
            if (subscriptions_[i]) result.emplace(output_names_[i], subscriptions_[i]->enabled());
        return result;
    }

    virtual void process() override {
        T* data = this->source_->peek();
        if (!data) return;

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
        this->source_->pop();
    }

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

    static std::shared_ptr<OneToMany> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<OneToMany>(make_unique<EdgeSource<T>>(in_edge));
        r->createSinksFromParameters(edges, params);
        const auto names = jsonToStringList(params.at("dst"));
        r->output_names_.assign(names.begin(), names.end());
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
        return r;
    }
};

DECLNODE_ATD(one_to_many, OneToMany);
