#include <pybind11/pybind11.h>
namespace py = pybind11;

extern void py_registerAVPlumber(py::module_ &m);
extern void py_registerNodeManager(py::module_ &m);
extern void py_registerEdgeManager(py::module_ &m);

PYBIND11_MODULE(_avplumber, m) {
    py_registerAVPlumber(m);
    py_registerNodeManager(m);
    py_registerEdgeManager(m);

    m.doc() = "AVPlumber is a library for managing audio and video pipelines";
}