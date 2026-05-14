#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "../../graph_interfaces.hpp"
#include <pybind11/gil.h>

namespace py = pybind11;

class PythonNodeAudioToMetadata:
    public NodeSISO<av::AudioSamples, MetadataFrame>,
    public IPythonNode,
    public IFlushable {
private:
    py::object python_node_;

    void callIfPresent(const char* attr_name) {
        py::gil_scoped_acquire gil;
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            return;
        }
        if (py::hasattr(python_node_, attr_name)) {
            python_node_.attr(attr_name)();
        }
    }

public:
    using NodeSISO<av::AudioSamples, MetadataFrame>::NodeSISO;

    ~PythonNodeAudioToMetadata() override {
        py::gil_scoped_acquire gil;
        python_node_ = py::object();
    }

    void set_python_node(py::object python_node) override {
        py::gil_scoped_acquire gil;
        python_node_ = std::move(python_node);
    }

    void start() override {
        NodeSingleOutput<MetadataFrame>::start();
        callIfPresent("doStart");
    }

    void process() override {
        py::gil_scoped_acquire gil;
        if (python_node_.ptr() == nullptr || python_node_.is_none()) {
            throw Error("Python node is not set");
        }
        python_node_.attr("process")();
    }

    void flush() override {
        callIfPresent("flush_open_segment");
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
        NodeSingleInput<av::AudioSamples>::stop();
        if (stop_error) {
            std::rethrow_exception(stop_error);
        }
    }

    static std::shared_ptr<PythonNodeAudioToMetadata> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return NodeSISO<av::AudioSamples, MetadataFrame>::template createCommon<PythonNodeAudioToMetadata>(edges, params);
    }
};

DECLNODE(python_node_audio_to_metadata, PythonNodeAudioToMetadata);
#endif
