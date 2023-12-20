#include "node_common.hpp"

template <typename T> class Split: public NodeSingleInput<T>, public NodeMultiOutput<T> {
protected:
    bool drop_ = false;
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual void process() {
        T* data = this->source_->peek();
        if (data==nullptr) {
            return;
        }
        for (auto &edge: this->sink_edges_) {
            if (!data->isComplete()) {
                logstream << "WARNING: split putting incomplete frame into sink!";
            }
            EdgeSink<T>(edge).put(*data, drop_);
        }
        this->source_->pop();
    }
    void setDrop(const bool drop) {
        drop_ = drop;
    }
    static std::shared_ptr<Split> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<Split>(make_unique<EdgeSource<T>>(in_edge));
        if (params.count("drop")==1) {
            r->setDrop(params["drop"].get<bool>());
        }
        r->createSinksFromParameters(edges, params);
        in_edge->setConsumer(r);
        return r;
    }
};

DECLNODE_ATD(split, Split);
