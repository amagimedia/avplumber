#include "node_common.hpp"
#include "../output_mask.hpp"
#include "../output_subscriptions.hpp"
#include "../mixer/primitives/frame_subscription.hpp"
#include "../frame_color_tags.hpp"

static uint32_t parseOutputsMask(const Parameters& value) { return parseBitmask(value); }

// `frame_color` applies to video frames; any other data passes as it is.
template <typename T> static void stampFrameColor(T&, const FrameColorTags&) {}
static void stampFrameColor(av::VideoFrame& frame, const FrameColorTags& tags) { tags.apply(frame.raw()); }

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

// On a thread of its own, sleeping until the input has a frame.
template <typename T>
class OneToMany : public OneToManyBase<T> {
public:
    using OneToManyBase<T>::OneToManyBase;

    virtual void process() override {
        T* data = this->source_->peek();
        if (!data) return;
        this->handOut(data);
        this->source_->pop();
    }

    static std::shared_ptr<OneToMany> create(NodeCreationInfo& nci) {
        return OneToManyBase<T>::template build<OneToMany>(nci);
    }
};

// On an event loop (`event_loop`), run when the input signals a frame: no thread to wake
// for every frame. It must not wait for room in an output, so `drop` is required: a frame
// an output has no room for is dropped for that output.
template <typename T>
class OneToManyNonBlocking : public OneToManyBase<T>, public NonBlockingNode<OneToManyNonBlocking<T>> {
    // Frames handed out in one go before other work on the loop gets its turn.
    static constexpr int kFramesPerTurn = 8;

public:
    using OneToManyBase<T>::OneToManyBase;

    virtual void processNonBlocking(EventLoop&, bool ticks) override {
        for (int handed = 0; handed < kFramesPerTurn; ++handed) {
            T* data = this->source_->peek(0);
            if (!data) {
                if (!ticks) this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                return;
            }
            this->handOut(data);
            this->source_->pop();
        }
        if (!ticks) this->yieldAndProcess();
    }

    static std::shared_ptr<OneToManyNonBlocking> create(NodeCreationInfo& nci) {
        auto r = OneToManyBase<T>::template build<OneToManyNonBlocking>(nci);
        if (!r->drop_) throw Error("one_to_many_nb: drop must be true, an event loop cannot wait for room in an output");
        return r;
    }
};

DECLNODE_ATD(one_to_many, OneToMany);
DECLNODE_ATD(one_to_many_nb, OneToManyNonBlocking);
