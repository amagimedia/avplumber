#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeMI: public NodeMultiInput<T>, public PythonNodeMixin {
public:
    void process() override {
        callProcess();
    }

    void stop() override {
        std::exception_ptr stop_error = captureStop();
        NodeMultiInput<T>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
    }

    void onEofConsumed() override {
        callDoStopOnce();
        NodeMultiInput<T>::onEofConsumed();
    }

    static std::shared_ptr<PythonNodeMI> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = std::make_shared<PythonNodeMI<T>>();
        r->createSourcesFromParameters(edges, params);
        return r;
    }
};

DECLNODE_ATD(python_node_mi, PythonNodeMI);
#endif
