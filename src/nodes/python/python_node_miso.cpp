#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

namespace py = pybind11;

template<typename T>
class PythonNodeMISO: public NodeMultiInput<T>, public NodeSingleOutput<T>, public IPythonNode {
private:
    py::object python_node_;

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    ~PythonNodeMISO() override {
        py::gil_scoped_acquire gil;
        python_node_ = py::object();
    }

    void set_python_node(py::object python_node) override {
        py::gil_scoped_acquire gil;
        python_node_ = std::move(python_node);
    }

    void process() override {
        py::gil_scoped_acquire gil;
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            throw Error("Python node is not set");
        }
        python_node_.attr("process")();
    }

    void stop() override {
        std::exception_ptr stop_error;
        {
            py::gil_scoped_acquire gil;
            try {
                if (python_node_.ptr() == nullptr || python_node_.is_none()) {
                    throw Error("Python node is not set");
                }
                python_node_.attr("doStop")();
            } catch (...) {
                stop_error = std::current_exception();
            }
        }
        NodeMultiInput<T>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
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
