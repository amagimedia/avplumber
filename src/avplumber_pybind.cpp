#include <pybind11/pybind11.h>
#include <pybind11/gil.h>
#include <pybind11_json/pybind11_json.hpp>
#include <pybind11/native_enum.h>
#include <algorithm>
#include <cstring>
#include <vector>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
}

#include "avplumber.hpp"
#include "app_version.hpp"
#include "graph_mgmt.hpp"
#include "metadata_frame.hpp"
#include <nlohmann/json.hpp>

namespace py = pybind11;
namespace nl = nlohmann;

namespace {

    struct FrameSideData {
        int type = 0;
        std::string data;
    
        FrameSideData() = default;
        FrameSideData(int t, const std::string& d): type(t), data(d) {}
    };
    
    static bool isKnownFrameSideDataType(int t) {
        switch (t) {
            case AV_FRAME_DATA_PANSCAN:
            case AV_FRAME_DATA_A53_CC:
            case AV_FRAME_DATA_STEREO3D:
            case AV_FRAME_DATA_MATRIXENCODING:
            case AV_FRAME_DATA_DOWNMIX_INFO:
            case AV_FRAME_DATA_REPLAYGAIN:
            case AV_FRAME_DATA_DISPLAYMATRIX:
            case AV_FRAME_DATA_AFD:
            case AV_FRAME_DATA_MOTION_VECTORS:
            case AV_FRAME_DATA_SKIP_SAMPLES:
            case AV_FRAME_DATA_AUDIO_SERVICE_TYPE:
            case AV_FRAME_DATA_MASTERING_DISPLAY_METADATA:
            case AV_FRAME_DATA_GOP_TIMECODE:
            case AV_FRAME_DATA_SPHERICAL:
            case AV_FRAME_DATA_CONTENT_LIGHT_LEVEL:
            case AV_FRAME_DATA_ICC_PROFILE:
    #if defined(AV_FRAME_DATA_QP_TABLE_PROPERTIES)
            case AV_FRAME_DATA_QP_TABLE_PROPERTIES:
    #endif
    #if defined(AV_FRAME_DATA_QP_TABLE_DATA)
            case AV_FRAME_DATA_QP_TABLE_DATA:
    #endif
            case AV_FRAME_DATA_S12M_TIMECODE:
            case AV_FRAME_DATA_DYNAMIC_HDR_PLUS:
            case AV_FRAME_DATA_REGIONS_OF_INTEREST:
            case AV_FRAME_DATA_VIDEO_ENC_PARAMS:
            case AV_FRAME_DATA_SEI_UNREGISTERED:
            case AV_FRAME_DATA_FILM_GRAIN_PARAMS:
    #if defined(AV_FRAME_DATA_DETECTION_BBOXES)
            case AV_FRAME_DATA_DETECTION_BBOXES:
    #endif
    #if defined(AV_FRAME_DATA_DOVI_RPU_BUFFER)
            case AV_FRAME_DATA_DOVI_RPU_BUFFER:
    #endif
    #if defined(AV_FRAME_DATA_DYNAMIC_HDR_VIVID)
            case AV_FRAME_DATA_DYNAMIC_HDR_VIVID:
    #endif
                return true;
            default:
                return false;
        }
    }
    
    template <typename T>
    class MetadataProxy {
        T* item_ = nullptr;
        py::object owner_ = py::none();
    
        auto raw() const -> decltype(item_->raw()) {
            return item_ ? item_->raw() : nullptr;
        }
    
    public:
        MetadataProxy() = default;
        MetadataProxy(T& item, py::object owner): item_(&item), owner_(std::move(owner)) {}
    
        py::dict asDict() const {
            py::dict out;
            const auto* r = raw();
            if (!r || !r->metadata) {
                return out;
            }
    
            const AVDictionaryEntry* entry = nullptr;
            while ((entry = av_dict_get(r->metadata, "", entry, AV_DICT_IGNORE_SUFFIX)) != nullptr) {
                out[py::str(entry->key)] = py::str(entry->value ? entry->value : "");
            }
            return out;
        }
    
        void assign(const py::dict& d) {
            auto* r = raw();
            if (!r) {
                return;
            }
            av_dict_free(&r->metadata);
            for (auto item: d) {
                std::string key = py::cast<std::string>(py::str(item.first));
                std::string value = py::cast<std::string>(py::str(item.second));
                av_dict_set(&r->metadata, key.c_str(), value.c_str(), 0);
            }
        }
    
        py::object getItem(const std::string& key) const {
            const auto* r = raw();
            if (!r || !r->metadata) {
                throw py::key_error(key);
            }
            const AVDictionaryEntry* entry = av_dict_get(r->metadata, key.c_str(), nullptr, 0);
            if (!entry) {
                throw py::key_error(key);
            }
            return py::str(entry->value ? entry->value : "");
        }
    
        void setItem(const std::string& key, const py::object& value) {
            auto* r = raw();
            if (!r) {
                return;
            }
            std::string str_value = py::cast<std::string>(py::str(value));
            av_dict_set(&r->metadata, key.c_str(), str_value.c_str(), 0);
        }
    
        void delItem(const std::string& key) {
            auto* r = raw();
            if (!r || !r->metadata) {
                throw py::key_error(key);
            }
            const AVDictionaryEntry* entry = av_dict_get(r->metadata, key.c_str(), nullptr, 0);
            if (!entry) {
                throw py::key_error(key);
            }
            av_dict_set(&r->metadata, key.c_str(), nullptr, 0);
        }
    
        bool contains(const std::string& key) const {
            const auto* r = raw();
            return r && r->metadata && av_dict_get(r->metadata, key.c_str(), nullptr, 0);
        }
    
        size_t size() const {
            const auto* r = raw();
            size_t count = 0;
            if (!r || !r->metadata) {
                return count;
            }
            const AVDictionaryEntry* entry = nullptr;
            while ((entry = av_dict_get(r->metadata, "", entry, AV_DICT_IGNORE_SUFFIX)) != nullptr) {
                ++count;
            }
            return count;
        }
    };

    class MetadataFrameMetadataProxy {
        MetadataFrame* frame_ = nullptr;
        py::object owner_ = py::none();

        Parameters* metadata() const {
            return frame_ ? &frame_->metadata() : nullptr;
        }

    public:
        MetadataFrameMetadataProxy() = default;
        MetadataFrameMetadataProxy(MetadataFrame& frame, py::object owner): frame_(&frame), owner_(std::move(owner)) {}

        py::object asDict() const {
            const Parameters* md = metadata();
            if (!md || !md->is_object()) {
                return py::dict();
            }
            return py::cast(*md);
        }

        void assign(const py::dict& d) {
            if (!frame_) {
                return;
            }
            frame_->setMetadata(pyjson::to_json(d));
        }

        py::object getItem(const std::string& key) const {
            const Parameters* md = metadata();
            if (!md || !md->is_object() || !md->contains(key)) {
                throw py::key_error(key);
            }
            return py::cast((*md)[key]);
        }

        void setItem(const std::string& key, const py::object& value) {
            Parameters* md = metadata();
            if (!md) {
                return;
            }
            if (!md->is_object()) {
                *md = Parameters::object();
            }
            (*md)[key] = pyjson::to_json(value);
        }

        void delItem(const std::string& key) {
            Parameters* md = metadata();
            if (!md || !md->is_object() || !md->contains(key)) {
                throw py::key_error(key);
            }
            md->erase(key);
        }

        bool contains(const std::string& key) const {
            const Parameters* md = metadata();
            return md && md->is_object() && md->contains(key);
        }

        size_t size() const {
            const Parameters* md = metadata();
            return (md && md->is_object()) ? md->size() : 0;
        }
    };
    
    class VideoFrameSideDataProxy {
        av::VideoFrame* frame_ = nullptr;
        py::object owner_ = py::none();
    
        AVFrame* raw() const {
            return frame_ ? frame_->raw() : nullptr;
        }
    
        struct SideDataEntry {
            int type = 0;
            std::string data;
        };
    
        static SideDataEntry parseEntry(const py::handle& obj) {
            if (py::isinstance<FrameSideData>(obj)) {
                auto e = py::cast<FrameSideData>(obj);
                return { e.type, e.data };
            }
            if (!py::isinstance<py::dict>(obj)) {
                throw py::type_error("side_data item must be FrameSideData or dict with keys: type, data");
            }
            py::dict d = py::reinterpret_borrow<py::dict>(obj);
            if (!d.contains("type") || !d.contains("data")) {
                throw py::key_error("side_data item must contain keys: type, data");
            }
            SideDataEntry e;
            e.type = py::cast<int>(d["type"]);
            py::bytes b = py::cast<py::bytes>(d["data"]);
            e.data = py::cast<std::string>(b);
            return e;
        }
    
        static FrameSideData toFrameSideData(const AVFrameSideData* sd) {
            return FrameSideData(
                static_cast<int>(sd->type),
                std::string(reinterpret_cast<const char*>(sd->data), static_cast<size_t>(sd->size))
            );
        }
    
        std::vector<SideDataEntry> entries() const {
            std::vector<SideDataEntry> out;
            const AVFrame* r = raw();
            if (!r) {
                return out;
            }
            out.reserve(static_cast<size_t>(r->nb_side_data));
            for (int i = 0; i < r->nb_side_data; ++i) {
                const AVFrameSideData* sd = r->side_data[i];
                if (!sd) {
                    continue;
                }
                SideDataEntry e;
                e.type = static_cast<int>(sd->type);
                e.data.assign(reinterpret_cast<const char*>(sd->data), static_cast<size_t>(sd->size));
                out.push_back(std::move(e));
            }
            return out;
        }
    
        void applyEntries(const std::vector<SideDataEntry>& items) {
            AVFrame* r = raw();
            if (!r) {
                return;
            }
    
            // Clear all current side data.
            while (r->nb_side_data > 0 && r->side_data && r->side_data[0]) {
                av_frame_remove_side_data(r, r->side_data[0]->type);
            }
    
            // Rebuild from desired sequence.
            for (const auto& e: items) {
                AVFrameSideData* sd = av_frame_new_side_data(
                    r, static_cast<AVFrameSideDataType>(e.type), static_cast<int>(e.data.size())
                );
                if (!sd) {
                    throw py::value_error("failed to allocate AVFrameSideData");
                }
                if (!e.data.empty()) {
                    std::memcpy(sd->data, e.data.data(), e.data.size());
                }
            }
        }
    
        static size_t normalizeIndex(py::ssize_t index, size_t n, bool allow_end = false) {
            py::ssize_t i = index;
            if (i < 0) {
                i += static_cast<py::ssize_t>(n);
            }
            py::ssize_t upper = static_cast<py::ssize_t>(n) + (allow_end ? 1 : 0);
            if (i < 0 || i >= upper) {
                throw py::index_error("side_data index out of range");
            }
            return static_cast<size_t>(i);
        }
    
    public:
        VideoFrameSideDataProxy() = default;
        VideoFrameSideDataProxy(av::VideoFrame& frame, py::object owner): frame_(&frame), owner_(std::move(owner)) {}
    
        py::list asList() const {
            py::list out;
            const AVFrame* r = raw();
            if (!r) {
                return out;
            }
            for (int i = 0; i < r->nb_side_data; ++i) {
                const AVFrameSideData* sd = r->side_data[i];
                if (!sd) {
                    continue;
                }
                out.append(py::cast(toFrameSideData(sd)));
            }
            return out;
        }
    
        void assign(const py::list& values) {
            std::vector<SideDataEntry> items;
            items.reserve(py::len(values));
            for (auto obj: values) {
                items.push_back(parseEntry(obj));
            }
            applyEntries(items);
        }
    
        size_t size() const {
            const AVFrame* r = raw();
            return r ? static_cast<size_t>(r->nb_side_data) : 0;
        }
    
        FrameSideData getItem(py::ssize_t index) const {
            const AVFrame* r = raw();
            size_t n = r ? static_cast<size_t>(r->nb_side_data) : 0;
            size_t i = normalizeIndex(index, n);
            return toFrameSideData(r->side_data[i]);
        }
    
        void setItem(py::ssize_t index, const py::handle& value) {
            auto items = entries();
            size_t i = normalizeIndex(index, items.size());
            items[i] = parseEntry(value);
            applyEntries(items);
        }
    
        void delItem(py::ssize_t index) {
            auto items = entries();
            size_t i = normalizeIndex(index, items.size());
            items.erase(items.begin() + static_cast<std::ptrdiff_t>(i));
            applyEntries(items);
        }
    
        void append(const py::handle& value) {
            auto items = entries();
            items.push_back(parseEntry(value));
            applyEntries(items);
        }
    
        void insert(py::ssize_t index, const py::handle& value) {
            auto items = entries();
            size_t i = normalizeIndex(index, items.size(), true);
            items.insert(items.begin() + static_cast<std::ptrdiff_t>(i), parseEntry(value));
            applyEntries(items);
        }
    
        FrameSideData pop(py::object index_obj = py::none()) {
            auto items = entries();
            if (items.empty()) {
                throw py::index_error("pop from empty side_data");
            }
            size_t i = items.size() - 1;
            if (!index_obj.is_none()) {
                i = normalizeIndex(py::cast<py::ssize_t>(index_obj), items.size());
            }
            FrameSideData removed(items[i].type, items[i].data);
            items.erase(items.begin() + static_cast<std::ptrdiff_t>(i));
            applyEntries(items);
            return removed;
        }
    
        void clear() {
            applyEntries({});
        }
    };
    
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
            .def_property_readonly("capacity", [](Edge<T> &e) { return e.capacity(); })
            .def("clear", &Edge<T>::clear)
            .def_property_readonly("occupied", [](Edge<T> &e) { return e.occupied(); })
            .def_property_readonly("free", [](Edge<T> &e) { return e.free(); })
            .def("enqueue", [](Edge<T> &e, const T &elem) {
                py::gil_scoped_release release;
                return e.enqueue(elem);
            })
            .def("pop", &Edge<T>::pop)
            .def("wait_dequeue", [](Edge<T> &e) {
                py::gil_scoped_release release;
                T elem{};
                e.wait_dequeue(elem);
                return elem;
            })
            .def("wait_dequeue_timed_ms", [](Edge<T> &e, unsigned int msec) {
                py::gil_scoped_release release;
                T elem{};
                if (!e.wait_dequeue_timed_ms(elem, msec)) {
                    throw py::value_error("wait_dequeue_timed_ms: timeout or empty");
                }
                return elem;
            })
            // Python-friendly aliases (Edge has no get/tryGet; map to dequeue APIs)
            .def("get", [](Edge<T> &e, int timeout_ms) {
                py::gil_scoped_release release;
                T elem{};
                if (timeout_ms < 0) {
                    e.wait_dequeue(elem);
                    return elem;
                }
                if (!e.wait_dequeue_timed_ms(elem, static_cast<unsigned>(timeout_ms))) {
                    throw py::value_error("get: timeout");
                }
                return elem;
            }, py::arg("timeout_ms") = -1)
            .def("tryGet", [](Edge<T> &e, int timeout_ms) -> py::object {
                py::gil_scoped_release release;
                T elem{};
                if (!e.wait_dequeue_timed_ms(elem, static_cast<unsigned>(std::max(0, timeout_ms)))) {
                    return py::none();
                }
                return py::cast(elem);
            }, py::arg("timeout_ms") = 0)
            .def("peek", &Edge<T>::peek, py::return_value_policy::reference)
            .def("wait_peek", [](Edge<T> &e, int timeout_ms) {
                py::gil_scoped_release release;
                return e.wait_peek(timeout_ms);
            }, py::arg("timeout_ms") = -1, py::return_value_policy::reference)
        ;
    }

    static int audioPlaneCount(const av::AudioSamples& s) {
        if (!s.raw() || s.samplesCount() <= 0 || s.channelsCount() <= 0) {
            return 0;
        }
        return s.isPlanar() ? s.channelsCount() : 1;
    }

    static const uint8_t* audioPlaneData(const av::AudioSamples& s, int plane) {
        const AVFrame* raw = s.raw();
        if (!raw || plane < 0 || plane >= audioPlaneCount(s)) {
            return nullptr;
        }
        if (raw->extended_data && raw->extended_data[plane]) {
            return raw->extended_data[plane];
        }
        return plane < AV_NUM_DATA_POINTERS ? raw->data[plane] : nullptr;
    }

    static size_t audioPlaneSize(const av::AudioSamples& s) {
        const AVFrame* raw = s.raw();
        if (!raw || s.samplesCount() <= 0 || s.channelsCount() <= 0) {
            return 0;
        }
        if (raw->linesize[0] > 0) {
            return static_cast<size_t>(raw->linesize[0]);
        }
        const int channels = s.isPlanar() ? 1 : s.channelsCount();
        int line_size = 0;
        const int size = av_samples_get_buffer_size(
            &line_size,
            channels,
            s.samplesCount(),
            static_cast<AVSampleFormat>(raw->format),
            1);
        return size > 0 ? static_cast<size_t>(size) : 0;
    }
    
}  // namespace
    
PYBIND11_MODULE(_avplumber, m) {
    py::class_<AVPlumber>(m, "AVPlumber")
        .def(py::init<>())
        .def("enableControlServer", &AVPlumber::enableControlServer)
        .def("registerWithWebUI", &AVPlumber::registerWithWebUI)
        .def("executeCommandsFromString", &AVPlumber::executeCommandsFromString)
        .def("executeCommandsFromFile", &AVPlumber::executeCommandsFromFile)
        .def("setLogFile", &AVPlumber::setLogFile)
        .def("setLogCallback", [](AVPlumber &avp, py::function callback) {
            avp.setLogCallback([callback](const std::string &s) {
                py::gil_scoped_acquire acquire;
                callback(py::str(s));
            });
        })
        .def("setReady", &AVPlumber::setReady)
        .def("shutdown", &AVPlumber::shutdown)
        .def("mainLoop", &AVPlumber::mainLoop)
        .def("stopMainLoop", &AVPlumber::stopMainLoop)
        .def("heartbeat", &AVPlumber::heartbeat)
        .def_property_readonly("manager", &AVPlumber::manager)
    ;

    py::class_<NodeManager, std::shared_ptr<NodeManager>>(m, "NodeManager")
        .def(py::init<>())
        .def("addNode", [](NodeManager &nm, py::dict &parameters, bool early_create=false, bool start=false, py::object node_obj=py::none()) {
            Parameters json_parameters = pyjson::to_json(parameters);
            if (node_obj.is_none()) {
                return nm.createNode(json_parameters, early_create, start);
            }
            auto result = nm.createNode(json_parameters, false, false);
            result->setPythonNodeObject(node_obj);
            if (early_create) {
                result->createNode();
            }
            if (start) {
                result->start();
            }
            return result;
        })
        .def("deleteNode", &NodeManager::deleteNode)
        .def("node", &NodeManager::node, py::arg("name"))
        .def("node_if_exists", &NodeManager::node_if_exists, py::arg("name"))
        .def("nodes", &NodeManager::nodes, py::arg("type"))
        .def_property_readonly("edges", [](NodeManager &nm) { return nm.edges(); })
        .def("group", &NodeManager::group, py::arg("name"))
        .def_property_readonly("allNodes", [](NodeManager &nm) {
            py::dict out;
            for (auto &node: nm.allNodes()) {
                out[node.first.c_str()] = node.second;
            }
            return out;
        })
    ;

    py::class_<NodeGroup, std::shared_ptr<NodeGroup>>(m, "NodeGroup")
        .def(py::init<NodeManager*, const std::string>())
        .def("startNodes", &NodeGroup::startNodes)
        .def("stopNodes", &NodeGroup::stopNodes)
        .def("restartNodes", &NodeGroup::restartNodes)
        .def_property_readonly("sortedNodes", [](NodeGroup &ng) -> py::list {
            py::list sorted_nodes;
            for (auto &weak_node: ng.sortedNodes()) {
                auto node = weak_node.lock();
                if (node) {
                    sorted_nodes.append(node);
                }
            }
            return sorted_nodes;
        })
    ;

    py::class_<NodeWrapper, std::shared_ptr<NodeWrapper>>(m, "NodeWrapper")
        .def_property_readonly("type", [](NodeWrapper &nw) { return nw.type(); })
        .def_property_readonly("name", [](NodeWrapper &nw) { return nw.name(); })
        .def("__repr__", [](NodeWrapper &nw) {
            return "Node(" + nw.type() + " / " + nw.name() + ")";
        })
        .def_property_readonly("parameters", &NodeWrapper::parameters)
        .def("getObject", &NodeWrapper::getObject)
        .def("start", &NodeWrapper::start)
        .def("stop", &NodeWrapper::stop)
        .def("interrupt", &NodeWrapper::interrupt, py::arg("optional") = false)
        .def("stopAndWait", &NodeWrapper::stopAndWait)
        .def("join", &NodeWrapper::join)
        .def_property_readonly("isWorking", [](NodeWrapper &nw) { return nw.isWorking(); })
    ;

    using VideoFrameMetadataProxy = MetadataProxy<av::VideoFrame>;
    using AudioSamplesMetadataProxy = MetadataProxy<av::AudioSamples>;
    using VideoFrameSideDataProxyType = VideoFrameSideDataProxy;
    using MetadataFrameMetadataProxyType = MetadataFrameMetadataProxy;

    py::class_<FrameSideData>(m, "FrameSideData")
        .def(py::init<>())
        .def(py::init<int, py::bytes>(), py::arg("type"), py::arg("data"))
        .def_property("type",
            [](const FrameSideData& s) { return s.type; },
            [](FrameSideData& s, int v) { s.type = v; }
        )
        .def_property("data",
            [](const FrameSideData& s) { return py::bytes(s.data.data(), s.data.size()); },
            [](FrameSideData& s, py::bytes b) { s.data = py::cast<std::string>(b); }
        )
        .def_property_readonly("type_enum", [](const FrameSideData& s) -> py::object {
            if (isKnownFrameSideDataType(s.type)) {
                return py::cast(static_cast<AVFrameSideDataType>(s.type));
            }
            return py::none();
        })
        .def("__repr__", [](const FrameSideData& s) {
            std::string enum_name = isKnownFrameSideDataType(s.type)
                ? py::str(py::cast(static_cast<AVFrameSideDataType>(s.type))).cast<std::string>()
                : "unknown";
            return "FrameSideData(type=" + std::to_string(s.type) + ", enum=" + enum_name +
                ", data_len=" + std::to_string(s.data.size()) + ")";
        })
    ;

    py::class_<VideoFrameMetadataProxy>(m, "VideoFrameMetadataProxy")
        .def("__repr__", [](const VideoFrameMetadataProxy& md) {
            py::object dict_obj = md.asDict();
            return "VideoFrameMetadataProxy(" + py::repr(dict_obj).cast<std::string>() + ")";
        })
        .def("__len__", &VideoFrameMetadataProxy::size)
        .def("__contains__", &VideoFrameMetadataProxy::contains)
        .def("__getitem__", &VideoFrameMetadataProxy::getItem)
        .def("__setitem__", &VideoFrameMetadataProxy::setItem)
        .def("__delitem__", &VideoFrameMetadataProxy::delItem)
        .def_property_readonly("as_dict", &VideoFrameMetadataProxy::asDict)
    ;

    py::class_<AudioSamplesMetadataProxy>(m, "AudioSamplesMetadataProxy")
        .def("__repr__", [](const AudioSamplesMetadataProxy& md) {
            py::object dict_obj = md.asDict();
            return "AudioSamplesMetadataProxy(" + py::repr(dict_obj).cast<std::string>() + ")";
        })
        .def("__len__", &AudioSamplesMetadataProxy::size)
        .def("__contains__", &AudioSamplesMetadataProxy::contains)
        .def("__getitem__", &AudioSamplesMetadataProxy::getItem)
        .def("__setitem__", &AudioSamplesMetadataProxy::setItem)
        .def("__delitem__", &AudioSamplesMetadataProxy::delItem)
        .def_property_readonly("as_dict", &AudioSamplesMetadataProxy::asDict)
    ;

    py::class_<MetadataFrameMetadataProxyType>(m, "MetadataFrameMetadataProxy")
        .def("__repr__", [](const MetadataFrameMetadataProxyType& md) {
            py::object dict_obj = md.asDict();
            return "MetadataFrameMetadataProxy(" + py::repr(dict_obj).cast<std::string>() + ")";
        })
        .def("__len__", &MetadataFrameMetadataProxyType::size)
        .def("__contains__", &MetadataFrameMetadataProxyType::contains)
        .def("__getitem__", &MetadataFrameMetadataProxyType::getItem)
        .def("__setitem__", &MetadataFrameMetadataProxyType::setItem)
        .def("__delitem__", &MetadataFrameMetadataProxyType::delItem)
        .def_property_readonly("as_dict", &MetadataFrameMetadataProxyType::asDict)
    ;

    py::class_<VideoFrameSideDataProxyType>(m, "VideoFrameSideDataProxy")
        .def("__repr__", [](const VideoFrameSideDataProxyType& sd) {
            py::object list_obj = sd.asList();
            return "VideoFrameSideDataProxy(" + py::repr(list_obj).cast<std::string>() + ")";
        })
        .def("__len__", &VideoFrameSideDataProxyType::size)
        .def("__getitem__", &VideoFrameSideDataProxyType::getItem)
        .def("__setitem__", &VideoFrameSideDataProxyType::setItem)
        .def("__delitem__", &VideoFrameSideDataProxyType::delItem)
        .def("append", &VideoFrameSideDataProxyType::append)
        .def("insert", &VideoFrameSideDataProxyType::insert)
        .def("pop", &VideoFrameSideDataProxyType::pop, py::arg("index") = py::none())
        .def("clear", &VideoFrameSideDataProxyType::clear)
        .def_property_readonly("as_list", &VideoFrameSideDataProxyType::asList)
        .def("__iter__", [](const VideoFrameSideDataProxyType& sd) {
            return py::iter(sd.asList());
        })
    ;

    py::class_<EdgeManager, std::shared_ptr<EdgeManager>>(m, "EdgeManager")
        .def(py::init<>())
//        .def_property_readonly("edges", [](EdgeManager &em) { return em.edges(); })
        .def("find__VideoFrame", &EdgeManager::find<av::VideoFrame>)
        .def("find__AudioSamples", &EdgeManager::find<av::AudioSamples>)
        .def("find__Packet", &EdgeManager::find<av::Packet>)
        .def("find__EglImageFrame", &EdgeManager::find<EglImageFrame>)
        .def("find__MetadataFrame", &EdgeManager::find<MetadataFrame>)
        .def("findAny", &EdgeManager::findAny)
        .def("planCapacity", &EdgeManager::planCapacity)
        .def("exists__VideoFrame", &EdgeManager::exists<av::VideoFrame>)
        .def("exists__AudioSamples", &EdgeManager::exists<av::AudioSamples>)
        .def("exists__Packet", &EdgeManager::exists<av::Packet>)
        .def("exists__EglImageFrame", &EdgeManager::exists<EglImageFrame>)
        .def("exists__MetadataFrame", &EdgeManager::exists<MetadataFrame>)
        //.def("printEdgesStats", &EdgeManager::printEdgesStats)
        .def("edgesStatsJson", &EdgeManager::edgesStatsJson)
        .def("resetEdgesOccupancyStats", &EdgeManager::resetEdgesOccupancyStats)
        .def("clearEdges", &EdgeManager::clearEdges)
    ;

    py_registerEdge<av::VideoFrame>(m, "Edge__VideoFrame");
    py_registerEdge<av::AudioSamples>(m, "Edge__AudioSamples");
    py_registerEdge<av::Packet>(m, "Edge__Packet");
    py_registerEdge<EglImageFrame>(m, "Edge__EglImageFrame");
    py_registerEdge<MetadataFrame>(m, "Edge__MetadataFrame");


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

    py::class_<MetadataFrame, std::shared_ptr<MetadataFrame>>(m, "MetadataFrame")
        .def(py::init<>())
        .def(py::init<av::Timestamp, Parameters>(),
            py::arg("pts"),
            py::arg("metadata") = Parameters::object())
        .def(py::init([](int64_t pts, av::Rational timebase, Parameters metadata) {
            return MetadataFrame(av::Timestamp(pts, timebase), std::move(metadata));
        }),
            py::arg("pts"),
            py::arg("timebase"),
            py::arg("metadata") = Parameters::object())
        .def("__repr__", [](const MetadataFrame &f) {
            return "MetadataFrame(" + std::to_string(f.pts().timestamp()) + ", " + f.metadata().dump() + ")";
        })
        .def_property("pts", &MetadataFrame::pts, &MetadataFrame::setPts)
        .def_property("metadata",
            [](MetadataFrame &f) -> MetadataFrameMetadataProxyType {
                return MetadataFrameMetadataProxyType(f, py::cast(&f, py::return_value_policy::reference));
            },
            [](MetadataFrame &f, const py::dict &d) {
                MetadataFrameMetadataProxyType(f, py::none()).assign(d);
            }
        )
        .def_property_readonly("isComplete", &MetadataFrame::isComplete)
    ;

    py::class_<av::AudioSamples, std::shared_ptr<av::AudioSamples>>(m, "AudioSamples")
        .def(py::init<>())
        .def_property_readonly("pts", &av::AudioSamples::pts)
        .def_property_readonly("samplesCount", &av::AudioSamples::samplesCount)
        .def_property_readonly("sampleRate", &av::AudioSamples::sampleRate)
        .def_property_readonly("channelsCount", &av::AudioSamples::channelsCount)
        .def_property_readonly("channelsLayout", &av::AudioSamples::channelsLayout)
        .def_property_readonly("channelsLayoutString", &av::AudioSamples::channelsLayoutString)
        .def_property_readonly("isPlanar", &av::AudioSamples::isPlanar)
        .def_property_readonly("sampleFormat", [](const av::AudioSamples& s) {
            return static_cast<int>(s.sampleFormat().get());
        })
        .def_property_readonly("sampleFormatName", [](const av::AudioSamples& s) {
            const char* name = av_get_sample_fmt_name(static_cast<AVSampleFormat>(s.sampleFormat().get()));
            return std::string(name ? name : "unknown");
        })
        .def_property_readonly("planeSize", [](const av::AudioSamples& s) {
            return audioPlaneSize(s);
        })
        .def_property_readonly("data_ptr", [](const av::AudioSamples& s) {
            py::list out;
            const int planes = audioPlaneCount(s);
            for (int i = 0; i < planes; ++i) {
                const uint8_t* ptr = audioPlaneData(s, i);
                out.append(py::int_(ptr ? reinterpret_cast<uintptr_t>(ptr) : uintptr_t(0)));
            }
            return out;
        })
        .def_property_readonly("planes", [](const av::AudioSamples& s) {
            py::list out;
            const int planes = audioPlaneCount(s);
            const size_t plane_size = audioPlaneSize(s);
            for (int i = 0; i < planes; ++i) {
                const uint8_t* ptr = audioPlaneData(s, i);
                if (!ptr || plane_size == 0) {
                    out.append(py::bytes());
                } else {
                    out.append(py::bytes(reinterpret_cast<const char*>(ptr), plane_size));
                }
            }
            return out;
        })
        .def_property("metadata",
            [](av::AudioSamples &s) -> AudioSamplesMetadataProxy {
                return AudioSamplesMetadataProxy(s, py::cast(&s, py::return_value_policy::reference));
            },
            [](av::AudioSamples &s, const py::dict &d) {
                AudioSamplesMetadataProxy(s, py::none()).assign(d);
            }
        )
        .def("data", [](av::AudioSamples &s, int channel) -> py::bytes {
            if (channel < 0 || static_cast<int>(s.channelsCount()) <= channel)
                return py::bytes();
            const uint8_t *ptr = s.data(channel);
            if (!ptr) return py::bytes();
            const AVFrame *raw = s.raw();
            if (!raw) return py::bytes();
            int bps = av_get_bytes_per_sample(static_cast<AVSampleFormat>(raw->format));
            if (bps <= 0) return py::bytes();
            return py::bytes(reinterpret_cast<const char*>(ptr),
                             static_cast<size_t>(raw->nb_samples) * static_cast<size_t>(bps));
        }, py::arg("channel") = 0, "Return raw PCM bytes for the given channel plane.")
        .def_property_readonly("isComplete", [](const av::AudioSamples &s) -> bool {
            return s.isComplete();
        })
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
        .def_property_readonly("linesize", [](const av::VideoFrame &f) {
            py::list out;
            const AVFrame* raw = f.raw();
            for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
                out.append(raw ? raw->linesize[i] : 0);
            }
            return out;
        })
        .def_property_readonly("data_ptr", [](const av::VideoFrame &f) {
            py::list out;
            const AVFrame* raw = f.raw();
            for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
                uintptr_t ptr = (raw && raw->data[i]) ? reinterpret_cast<uintptr_t>(raw->data[i]) : uintptr_t(0);
                out.append(py::int_(ptr));
            }
            return out;
        })
        .def_property_readonly("data", [](const av::VideoFrame &f) {
            py::list out;
            const AVFrame* raw = f.raw();
            if (!raw) {
                for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
                    out.append(py::bytes());
                }
                return out;
            }

            size_t plane_sizes[4] = {0, 0, 0, 0};
            ptrdiff_t linesizes[4] = {
                raw->linesize[0], raw->linesize[1], raw->linesize[2], raw->linesize[3]
            };
            bool have_plane_sizes = (raw->height > 0) &&
                (av_image_fill_plane_sizes(plane_sizes, (AVPixelFormat)raw->format, raw->height, linesizes) >= 0);

            for (int i = 0; i < AV_NUM_DATA_POINTERS; ++i) {
                if (!raw->data[i]) {
                    out.append(py::bytes());
                    continue;
                }

                size_t size = 0;
                if (have_plane_sizes && i < 4) {
                    size = plane_sizes[i];
                } else if (raw->height > 0 && raw->linesize[i] != 0) {
                    size = static_cast<size_t>(std::abs(raw->linesize[i])) * static_cast<size_t>(raw->height);
                }
                out.append(py::bytes(reinterpret_cast<const char*>(raw->data[i]), size));
            }
            return out;
        })
        .def_property("side_data",
            [](av::VideoFrame &f) -> VideoFrameSideDataProxyType {
                return VideoFrameSideDataProxyType(f, py::cast(&f, py::return_value_policy::reference));
            },
            [](av::VideoFrame &f, const py::list &values) {
                VideoFrameSideDataProxyType(f, py::none()).assign(values);
            }
        )
        
        .def_property("keyFrame", &av::VideoFrame::isKeyFrame, &av::VideoFrame::setKeyFrame )
        .def_property("quality", &av::VideoFrame::quality, &av::VideoFrame::setQuality )
        .def_property("pictureType", &av::VideoFrame::pictureType, &av::VideoFrame::setPictureType )
        .def_property("sampleAspectRatio", &av::VideoFrame::sampleAspectRatio, &av::VideoFrame::setSampleAspectRatio )
        .def_property("metadata",
            [](av::VideoFrame &f) -> VideoFrameMetadataProxy {
                return VideoFrameMetadataProxy(f, py::cast(&f, py::return_value_policy::reference));
            },
            [](av::VideoFrame &f, const py::dict &d) {
                VideoFrameMetadataProxy(f, py::none()).assign(d);
            }
        )
    ;

    py::class_<av::PixelFormat, std::shared_ptr<av::PixelFormat>>(m, "PixelFormat")
        .def(py::init<>())
        .def("__repr__", [](const av::PixelFormat &pf) {
            return "PixelFormat(" + std::string(pf.name()) + ", " + std::to_string(pf.bitsPerPixel()) + "bpp, " + std::to_string(pf.planesCount()) + "planes)";
        })
        .def_property_readonly("value", [&](const av::PixelFormat &pf) { return static_cast<int>(pf); })
        .def_property_readonly("name", [&](const av::PixelFormat &pf) -> std::string {
            return std::string(pf.name());
        })
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

    py::native_enum<AVFrameSideDataType>(m, "AVFrameSideDataType", "enum.IntEnum")
        .value("PANSCAN", AV_FRAME_DATA_PANSCAN)
        .value("A53_CC", AV_FRAME_DATA_A53_CC)
        .value("STEREO3D", AV_FRAME_DATA_STEREO3D)
        .value("MATRIXENCODING", AV_FRAME_DATA_MATRIXENCODING)
        .value("DOWNMIX_INFO", AV_FRAME_DATA_DOWNMIX_INFO)
        .value("REPLAYGAIN", AV_FRAME_DATA_REPLAYGAIN)
        .value("DISPLAYMATRIX", AV_FRAME_DATA_DISPLAYMATRIX)
        .value("AFD", AV_FRAME_DATA_AFD)
        .value("MOTION_VECTORS", AV_FRAME_DATA_MOTION_VECTORS)
        .value("SKIP_SAMPLES", AV_FRAME_DATA_SKIP_SAMPLES)
        .value("AUDIO_SERVICE_TYPE", AV_FRAME_DATA_AUDIO_SERVICE_TYPE)
        .value("MASTERING_DISPLAY_METADATA", AV_FRAME_DATA_MASTERING_DISPLAY_METADATA)
        .value("GOP_TIMECODE", AV_FRAME_DATA_GOP_TIMECODE)
        .value("SPHERICAL", AV_FRAME_DATA_SPHERICAL)
        .value("CONTENT_LIGHT_LEVEL", AV_FRAME_DATA_CONTENT_LIGHT_LEVEL)
        .value("ICC_PROFILE", AV_FRAME_DATA_ICC_PROFILE)
#if defined(AV_FRAME_DATA_QP_TABLE_PROPERTIES)
        .value("QP_TABLE_PROPERTIES", AV_FRAME_DATA_QP_TABLE_PROPERTIES)
#endif
#if defined(AV_FRAME_DATA_QP_TABLE_DATA)
        .value("QP_TABLE_DATA", AV_FRAME_DATA_QP_TABLE_DATA)
#endif
        .value("S12M_TIMECODE", AV_FRAME_DATA_S12M_TIMECODE)
        .value("DYNAMIC_HDR_PLUS", AV_FRAME_DATA_DYNAMIC_HDR_PLUS)
        .value("REGIONS_OF_INTEREST", AV_FRAME_DATA_REGIONS_OF_INTEREST)
        .value("VIDEO_ENC_PARAMS", AV_FRAME_DATA_VIDEO_ENC_PARAMS)
        .value("SEI_UNREGISTERED", AV_FRAME_DATA_SEI_UNREGISTERED)
        .value("FILM_GRAIN_PARAMS", AV_FRAME_DATA_FILM_GRAIN_PARAMS)
#if defined(AV_FRAME_DATA_DETECTION_BBOXES)
        .value("DETECTION_BBOXES", AV_FRAME_DATA_DETECTION_BBOXES)
#endif
#if defined(AV_FRAME_DATA_DOVI_RPU_BUFFER)
        .value("DOVI_RPU_BUFFER", AV_FRAME_DATA_DOVI_RPU_BUFFER)
#endif
#if defined(AV_FRAME_DATA_DYNAMIC_HDR_VIVID)
        .value("DYNAMIC_HDR_VIVID", AV_FRAME_DATA_DYNAMIC_HDR_VIVID)
#endif
        .export_values()
        .finalize()
    ;

    m.def("version", []() {
        return std::string(APP_VERSION ? APP_VERSION : "");
    }, "Return AVPlumber version.");

    m.doc() = "AVPlumber is a library for managing audio and video pipelines";
}
