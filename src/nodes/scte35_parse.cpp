#include "node_common.hpp"
#include <libavcodec/packet.h>
#include <libklscte35/scte35.h>

class SCTE35Parser: public NodeSingleInput<av::Packet> {
protected:

public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    virtual void process() override {
        logstream << "inside scte parse";
        av::Packet pkt = this->source_->get();
        if (!pkt.isComplete()) return;
        AVPacket* frm = pkt.raw();
        uint8_t* payload = frm->data;

        logstream << "got scte packet: " << frm->size;

        scte35_splice_info_section_s s;

        scte35_splice_info_section_unpackFrom(&s, payload, frm->size);
        scte35_splice_info_section_print(&s);
    }

    static std::shared_ptr<SCTE35Parser> create(NodeCreationInfo &nci) {
        logstream << "create scte parser";
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> src_edge = edges.find<av::Packet>(params["src"]);
        return std::make_shared<SCTE35Parser>(make_unique<EdgeSource<av::Packet>>(src_edge));
    }
};

DECLNODE(parse_scte35, SCTE35Parser);
