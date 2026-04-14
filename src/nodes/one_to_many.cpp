#include "node_common.hpp"
#include "../SharedTimeline.hpp"

static uint32_t parseOutputsMask(const Parameters& value) { return parseBitmask(value); }

template <typename T>
class OneToMany : public NodeSingleInput<T>, public NodeMultiOutput<T>,
                  public TimelineReader, public IInputsObjects, public IReturnsObjects {
    std::atomic<uint32_t> outputs_mask_{1};
    bool drop_ = false;

public:
    using NodeSingleInput<T>::NodeSingleInput;

    virtual void process() override {
        T* data = this->source_->peek();
        if (!data) return;

        uint32_t mask = outputs_mask_.load(std::memory_order_relaxed);
        if (hasTimeline()) {
            auto opt = tlGetRaw("outputs", data->pts());
            if (opt) mask = parseOutputsMask(*opt);
        }

        for (size_t i = 0; i < this->sink_edges_.size(); i++) {
            if (mask & (1u << i))
                EdgeSink<T>(this->sink_edges_[i]).put(*data, drop_);
        }
        this->source_->pop();
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "outputs")
            outputs_mask_.store(parseOutputsMask(value), std::memory_order_relaxed);
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
        in_edge->setConsumer(r);
        r->initTimeline(nci);
        if (params.count("outputs"))
            r->outputs_mask_.store(parseOutputsMask(params["outputs"]), std::memory_order_relaxed);
        if (params.count("drop"))
            r->drop_ = params["drop"].get<bool>();
        return r;
    }
};

DECLNODE_ATD(one_to_many, OneToMany);
