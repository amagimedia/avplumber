#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeMO: public NodeMultiOutput<T>, public IStoppable, public PythonNodeMixin {
public:
    void process() override {
        callProcess();
    }

    void stop() override {
        callDoStopOnce();
    }

    static std::shared_ptr<PythonNodeMO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = std::make_shared<PythonNodeMO<T>>();
        r->createSinksFromParameters(edges, params);
        return r;
    }
};

DECLNODE_ATD(python_node_mo, PythonNodeMO);
#endif
