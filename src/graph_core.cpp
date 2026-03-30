#include "graph_core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <libavutil/avutil.h>

namespace py = pybind11;

EdgeManager EdgeManager::global_edge_manager_;

namespace {

template <typename T>
void py_registerEdge(py::module_ &m, const char *type_name) {
    py::class_<Edge<T>, std::shared_ptr<Edge<T>>>(m, type_name)
        .def(py::init<size_t>())
        .def("__repr__", [](Edge<T> &e) {
            return "Edge(), " + std::to_string(e.occupied()) + "/" + std::to_string(e.capacity());
        })
        .def("addWiretapCallback", [](Edge<T> &e, py::function f) {
            e.addWiretapCallback([f](const T &p) {
                py::gil_scoped_acquire gil;
                f(py::cast(p));
            });
        })
        .def("addPacketCallback", [](Edge<T> &e, py::function f) {
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
        .def("pop", &Edge<T>::pop)
        .def("peek", &Edge<T>::peek, py::return_value_policy::reference)
        .def("wait_peek", &Edge<T>::wait_peek, py::arg("timeout_ms") = -1, py::return_value_policy::reference)
    ;
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


    py::class_<av::Timestamp, std::shared_ptr<av::Timestamp>>(m, "Timestamp")
        .def(py::init<int64_t, av::Rational>())
        .def("__repr__", [](const av::Timestamp &t) {
            return "Timestamp(" + std::to_string(t.timestamp()) + ", " + std::to_string(t.timebase().getNumerator()) + "/" + std::to_string(t.timebase().getDenominator()) + ")";
        })
        .def_property_readonly("timestamp", [](const av::Timestamp &t) { return t.timestamp(); })
        .def_property_readonly("timebase", [](const av::Timestamp &t) { return t.timebase(); })
    ;

    py::class_<av::Packet, std::shared_ptr<av::Packet>>(m, "Packet")
        .def(py::init<>())
        .def_property_readonly("pts", [](const av::Packet &p) { return p.pts(); })
        .def_property_readonly("dts", [](const av::Packet &p) { return p.dts(); })
        .def_property_readonly("duration", [](const av::Packet &p) { return p.duration(); })
        .def_property_readonly("size", [](const av::Packet &p) { return p.size(); })
        .def_property_readonly("data", [](const av::Packet &p) -> py::bytes { 
            return py::bytes(reinterpret_cast<const char*>(p.data()), p.size());
        })
        .def_property_readonly("stream_index", [](const av::Packet &p) { return p.streamIndex(); })
        .def_property_readonly("flags", [](const av::Packet &p) { return p.flags(); })
    ;

    py::class_<av::VideoFrame, std::shared_ptr<av::VideoFrame>>(m, "VideoFrame")
        .def(py::init<>())
        .def("__repr__", [](const av::VideoFrame &f) {
            return "VideoFrame(" + std::to_string(f.width()) + "x" + std::to_string(f.height()) + ", " + f.pixelFormat().name() + ", " + std::to_string(f.pts().timestamp()) + ")";
        })
        .def_property_readonly("width", &av::VideoFrame::width)
        .def_property_readonly("height", &av::VideoFrame::height)
        .def_property_readonly("format", &av::VideoFrame::pixelFormat)
        .def_property_readonly("pts", &av::VideoFrame::pts)
        .def_property("keyFrame", &av::VideoFrame::isKeyFrame, &av::VideoFrame::setKeyFrame )
        .def_property("quality", &av::VideoFrame::quality, &av::VideoFrame::setQuality )
        .def_property("pictureType", &av::VideoFrame::pictureType, &av::VideoFrame::setPictureType )
        .def_property("sampleAspectRatio", &av::VideoFrame::sampleAspectRatio, &av::VideoFrame::setSampleAspectRatio )
    ;

    py::class_<av::PixelFormat, std::shared_ptr<av::PixelFormat>>(m, "PixelFormat")
        .def(py::init<>())
        .def("__repr__", [](const av::PixelFormat &pf) {
            return "PixelFormat(" + std::string(pf.name()) + ", " + std::to_string(pf.bitsPerPixel()) + "bpp, " + std::to_string(pf.planesCount()) + "planes)";
        })
        .def_property_readonly("value", [&](const av::PixelFormat &pf) { return static_cast<int>(pf); })
        .def_property_readonly("name", &av::PixelFormat::name)
        .def_property_readonly("bitsPerPixel", &av::PixelFormat::bitsPerPixel)
        .def_property_readonly("planesCount", &av::PixelFormat::planesCount)
    ;

    py::class_<av::Rational, std::shared_ptr<av::Rational>>(m, "Rational")
        .def(py::init<int, int>())
        .def("__repr__", [](const av::Rational &r) {
            return std::to_string(r.getNumerator()) + "/" + std::to_string(r.getDenominator());
        })
        .def_property("num", &av::Rational::getNumerator, &av::Rational::setNumerator)
        .def_property("den", &av::Rational::getDenominator, &av::Rational::setDenominator)
    ;

    py::native_enum<AVPictureType>(m, "AVPictureType", "enum.IntEnum")
        .value("NONE", AV_PICTURE_TYPE_NONE)
        .value("I", AV_PICTURE_TYPE_I)
        .value("P", AV_PICTURE_TYPE_P)
        .value("B", AV_PICTURE_TYPE_B)
        .value("S", AV_PICTURE_TYPE_S)
        .value("SI", AV_PICTURE_TYPE_SI)
        .value("SP", AV_PICTURE_TYPE_SP)
        .value("BI", AV_PICTURE_TYPE_BI)
        .export_values()
        .finalize()
    ;
}
