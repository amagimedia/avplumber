#ifdef PYTHON_MODULE
#include "../node_common.hpp"
#include "python_node_mixin.hpp"

class PythonNodeAudioToMetadata:
    public NodeSISO<av::AudioSamples, MetadataFrame>,
    public PythonNodeMixin,
    public IFlushable {
public:
    using NodeSISO<av::AudioSamples, MetadataFrame>::NodeSISO;

    void start() override {
        NodeSingleOutput<MetadataFrame>::start();
        callOptional("doStart");
    }

    void process() override {
        callProcess();
    }

    void flush() override {
        callOptional("flush_open_segment");
    }

    void stop() override {
        std::exception_ptr stop_error = captureStop();
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
