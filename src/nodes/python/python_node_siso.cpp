#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeSISO: public NodeSISO<T, T>, public PythonNodeMixin {
public:
    using NodeSISO<T, T>::NodeSISO;

    void process() override {
        callProcess();
    }

    void stop() override {
        std::exception_ptr stop_error = captureStop();
        NodeSingleInput<T>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
    }

    void onEofConsumed() override {
        callDoStopOnce();
        NodeSISO<T, T>::onEofConsumed();
    }

    static std::shared_ptr<PythonNodeSISO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return NodeSISO<T, T>::template createCommon<PythonNodeSISO<T>>(edges, params);
    }
};

DECLNODE_ATD(python_node_siso, PythonNodeSISO);
#endif
