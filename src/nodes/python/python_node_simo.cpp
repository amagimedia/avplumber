#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeSIMO: public NodeSingleInput<T>, public NodeMultiOutput<T>, public PythonNodeMixin {
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
        this->emitEof();
    }

    static std::shared_ptr<PythonNodeSIMO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<PythonNodeSIMO<T>>(make_unique<EdgeSource<T>>(in_edge));
        r->createSinksFromParameters(edges, params);
        in_edge->setConsumer(r);
        return r;
    }
};

DECLNODE_ATD(python_node_simo, PythonNodeSIMO);
#endif
