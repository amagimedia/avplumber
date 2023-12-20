#include "node_common.hpp"

template <typename T> class Firewall: public NodeSISO<T, T> {
protected:
    bool ready_ = false;
    AVTS offset_;
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        T data = this->source_->get();
        if (!data.pts().isValid()) return;
        this->sink_->put(data);
    }
    static std::shared_ptr<Firewall> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return NodeSISO<T, T>::template createCommon<Firewall>(edges, params);
    }
};

DECLNODE_ATD(firewall, Firewall);
