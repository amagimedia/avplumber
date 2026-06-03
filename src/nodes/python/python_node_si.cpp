#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeSI: public NodeSingleInput<T>, public PythonNodeMixin {
public:
    using NodeSingleInput<T>::NodeSingleInput;

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
        NodeSingleInput<T>::onEofConsumed();
    }

    static std::shared_ptr<PythonNodeSI> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        return std::make_shared<PythonNodeSI<T>>(make_unique<EdgeSource<T>>(in_edge));
    }
};

DECLNODE_ATD(python_node_si, PythonNodeSI);
#endif
