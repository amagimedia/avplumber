#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeSO: public NodeSingleOutput<T>, public IStoppable, public PythonNodeMixin {
public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    void process() override {
        callProcess();
    }

    void stop() override {
        callDoStopOnce();
    }

    static std::shared_ptr<PythonNodeSO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto out_edge = edges.find<T>(params["dst"]);
        return std::make_shared<PythonNodeSO<T>>(make_unique<EdgeSink<T>>(out_edge));
    }
};

DECLNODE_ATD(python_node_so, PythonNodeSO);
#endif
