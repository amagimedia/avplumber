#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

template<typename T>
class PythonNodeMISO: public NodeMultiInput<T>, public NodeSingleOutput<T>, public PythonNodeMixin {
public:
    using NodeSingleOutput<T>::NodeSingleOutput;

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

    static std::shared_ptr<PythonNodeMISO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto out_edge = edges.find<T>(params["dst"]);
        auto r = std::make_shared<PythonNodeMISO<T>>(make_unique<EdgeSink<T>>(out_edge));
        r->createSourcesFromParameters(edges, params);
        out_edge->setProducer(r);
        return r;
    }
};

DECLNODE_ATD(python_node_miso, PythonNodeMISO);
#endif
