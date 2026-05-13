#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

template<typename T>
class PythonNodeSISO: public NodeSISO<T, T>, public IPythonNode {
private:
    py::object python_node_;

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

    static std::shared_ptr<PythonNodeSISO> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return NodeSISO<T, T>::template createCommon<PythonNodeSISO<T>>(edges, params);
    }
};

DECLNODE_ATD(python_node_siso, PythonNodeSISO);
#endif
