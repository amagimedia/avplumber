#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

template<typename T>
class PythonNodeSIMO: public NodeSingleInput<T>, public NodeMultiOutput<T>, public IPythonNode {
private:
    py::object python_node_;

public:
    using NodeSingleInput<T>::NodeSingleInput;

    ~PythonNodeSIMO() override {
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
        NodeSingleInput<T>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
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
