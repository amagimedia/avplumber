#include "node_common.hpp"
#include "../picture_buffer.hpp"

class PictureBufferSink: public NodeSingleInput<av::VideoFrame>, public ReportsFinishByFlag {
protected:
    InstanceData &instance_;
    std::string buffer_name_;
    bool once_ = true;
    using ISOs = InstanceSharedObjects<PictureBuffer>;
public:
    PictureBufferSink(std::unique_ptr<SourceType> &&source, InstanceData &instance, std::string buffer_name):
        NodeSingleInput<av::VideoFrame>(std::move(source)),
        instance_(instance),
        buffer_name_(buffer_name) {
    }
    virtual void process() {
        av::VideoFrame frm = this->source_->get();
        if (frm.isValid()) {
            if (once_) {
                logstream << "Got frame for picture buffer, putting and finishing";
            }
            ISOs::emplace(instance_, buffer_name_, ISOs::PolicyIfExists::Overwrite, frm);
            if (once_) {
                this->finished_ = true;
            }
        } else {
            this->finished_ = true;
        }
    }
    static std::shared_ptr<PictureBufferSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        std::shared_ptr<Edge<av::VideoFrame>> edge = edges.find<av::VideoFrame>(params["src"]);
        auto r = std::make_shared<PictureBufferSink>(make_unique<EdgeSource<av::VideoFrame>>(edge), nci.instance, params["buffer"]);
        if (params.count("once")) {
            r->once_ = params["once"];
        }
        return r;
    }
};

DECLNODE(picture_buffer_sink, PictureBufferSink);
