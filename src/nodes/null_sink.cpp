#include "node_common.hpp"

template<typename T> class NullSink: public NodeSingleInput<T> {
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual void process() {
        this->source_->get();
    }
    static std::shared_ptr<NullSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<T>> edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<NullSink>(make_unique<EdgeSource<T>>(edge));
        return r;
    }
};

DECLNODE_ATD(null_sink, NullSink);
