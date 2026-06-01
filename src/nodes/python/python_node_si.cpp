#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

namespace py = pybind11;

template<typename T>
class PythonNodeSI: public NodeSingleInput<T>, public IPythonNode {
private:
    py::object python_node_;
    bool stopped_ = false;

    void callDoStopOnce() {
        if (stopped_) {
            return;
        }
        stopped_ = true;
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            throw Error("Python node is not set");
        }
        python_node_.attr("doStop")();
    }

public:
    using NodeSingleInput<T>::NodeSingleInput;

    ~PythonNodeSI() override {
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
                callDoStopOnce();
            } catch (...) {
                stop_error = std::current_exception();
            }
        }
        NodeSingleInput<T>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
    }

    void onEofConsumed() override {
        py::gil_scoped_acquire gil;
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
