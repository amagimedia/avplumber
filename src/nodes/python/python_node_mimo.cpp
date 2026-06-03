#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeMIMO: public NodeMultiInput<T>, public NodeMultiOutput<T>, public PythonNodeMixin {
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
        this->emitEof();
    }

    static std::shared_ptr<PythonNodeMIMO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = std::make_shared<PythonNodeMIMO<T>>();
        r->createSourcesFromParameters(edges, params);
        r->createSinksFromParameters(edges, params);
        return r;
    }
};

DECLNODE_ATD(python_node_mimo, PythonNodeMIMO);
#endif
