#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

namespace py = pybind11;

template<typename T>
class PythonNodeSISO: public NodeSISO<T, T>, public IPythonNode {
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
    using NodeSISO<T, T>::NodeSISO;

    ~PythonNodeSISO() override {
        py::gil_scoped_acquire gil;
        // Leave a null handle so member destruction after GIL release is safe.
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
        {
            py::gil_scoped_acquire gil;
            callDoStopOnce();
        }
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
