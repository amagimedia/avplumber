#include "graph_core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/gil.h>

namespace py = pybind11;

EdgeManager EdgeManager::global_edge_manager_;

namespace {

template <typename T>
void py_registerEdge(py::module_ &m, const char *name) {
    py::class_<Edge<T>, std::shared_ptr<Edge<T>>>(m, name)
        .def(py::init<size_t>())
        .def("addWiretapCallback", [](Edge<T> &e, py::function f) {
            e.addWiretapCallback([f](const T &p) {
                py::gil_scoped_acquire gil;
                f(py::cast(p));
            });
        })
        .def_property_readonly("capacity", [](Edge<T> &e) { return e.capacity(); })
        .def("clear", &Edge<T>::clear)
        .def_property_readonly("occupied", [](Edge<T> &e) { return e.occupied(); })
        .def_property_readonly("free", [](Edge<T> &e) { return e.free(); })
        .def("enqueue", &Edge<T>::enqueue)
        .def("pop", &Edge<T>::pop);
}

}  // namespace

void py_registerEdgeManager(py::module_ &m) {
    py::class_<EdgeManager, std::shared_ptr<EdgeManager>>(m, "EdgeManager")
        .def(py::init<>())
//        .def_property_readonly("edges", [](EdgeManager &em) { return em.edges(); })
        .def("find__VideoFrame", &EdgeManager::find<av::VideoFrame>)
        .def("find__AudioSamples", &EdgeManager::find<av::AudioSamples>)
        .def("find__Packet", &EdgeManager::find<av::Packet>)
        .def("find__EglImageFrame", &EdgeManager::find<EglImageFrame>)
        .def("findAny", &EdgeManager::findAny)
        .def("planCapacity", &EdgeManager::planCapacity)
        .def("exists__VideoFrame", &EdgeManager::exists<av::VideoFrame>)
        .def("exists__AudioSamples", &EdgeManager::exists<av::AudioSamples>)
        .def("exists__Packet", &EdgeManager::exists<av::Packet>)
        .def("exists__EglImageFrame", &EdgeManager::exists<EglImageFrame>)
        //.def("printEdgesStats", &EdgeManager::printEdgesStats)
        .def("edgesStatsJson", &EdgeManager::edgesStatsJson)
        .def("resetEdgesOccupancyStats", &EdgeManager::resetEdgesOccupancyStats)
        .def("clearEdges", &EdgeManager::clearEdges)
    ;

    py_registerEdge<av::VideoFrame>(m, "Edge__VideoFrame");
    py_registerEdge<av::AudioSamples>(m, "Edge__AudioSamples");
    py_registerEdge<av::Packet>(m, "Edge__Packet");
    py_registerEdge<EglImageFrame>(m, "Edge__EglImageFrame");


    py::class_<av::Rational, std::shared_ptr<av::Rational>>(m, "Rational")
        .def(py::init<int, int>())
        .def_property("num", &av::Rational::getNumerator, &av::Rational::setNumerator)
        .def_property("den", &av::Rational::getDenominator, &av::Rational::setDenominator)
    ;

    py::class_<av::Timestamp, std::shared_ptr<av::Timestamp>>(m, "Timestamp")
        .def(py::init<int64_t, av::Rational>())
        .def_property_readonly("timestamp", [](const av::Timestamp &t) { return t.timestamp(); })
        .def_property_readonly("timebase", [](const av::Timestamp &t) { return t.timebase(); })
    ;

    py::class_<av::Packet, std::shared_ptr<av::Packet>>(m, "Packet")
        .def(py::init<>())
        .def_property_readonly("pts", [](const av::Packet &p) { return p.pts(); })
        .def_property_readonly("dts", [](const av::Packet &p) { return p.dts(); })
        .def_property_readonly("duration", [](const av::Packet &p) { return p.duration(); })
        .def_property_readonly("size", [](const av::Packet &p) { return p.size(); })
        .def_property_readonly("data", [](const av::Packet &p) { return p.data(); })
        .def_property_readonly("stream_index", [](const av::Packet &p) { return p.streamIndex(); })
        .def_property_readonly("flags", [](const av::Packet &p) { return p.flags(); })
    ;
}
