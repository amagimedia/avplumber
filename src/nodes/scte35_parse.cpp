#include "node_common.hpp"
#include "../rest_client.hpp"
extern "C" {
#include <libavcodec/packet.h>
}
#include <libklscte35/scte35.h>

class SCTE35Parser: public NodeSingleInput<av::Packet> {
protected:
    ThreadedRESTEndpoint rest_;
    bool json_active_ = false;
    bool log_active_ = false;

public:
    using NodeSingleInput<av::Packet>::NodeSingleInput;
    virtual void process() override {
        av::Packet pkt = this->source_->get();
        if (!pkt.isComplete()) return;
        AVPacket* frm = pkt.raw();
        uint8_t* payload = frm->data;

        scte35_splice_info_section_s s;

        scte35_splice_info_section_unpackFrom(&s, payload, frm->size);

        if (s.splice_command_type == SCTE35_COMMAND_TYPE__SPLICE_INSERT) {
            if (json_active_ || log_active_) {
                Parameters p;
                p["scte_event"]["event_id"] = s.splice_insert.splice_event_id;
                p["scte_event"]["immediate"] = !!s.splice_insert.splice_immediate_flag;
                p["scte_event"]["cancel"] = !!s.splice_insert.splice_event_cancel_indicator;
                p["scte_event"]["out"] = !!s.splice_insert.out_of_network_indicator;
                if (s.splice_insert.duration_flag) {
                    p["scte_event"]["duration"] = s.splice_insert.duration.duration;
                }

                if (log_active_) {
                    logstream << p;
                }
                if (json_active_) {
                    rest_.send("", p);
                }
            }
        }
    }

    static std::shared_ptr<SCTE35Parser> create(NodeCreationInfo &nci) {
        logstream << "create scte parser";
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> src_edge = edges.find<av::Packet>(params["src"]);
        auto parser = std::make_shared<SCTE35Parser>(make_unique<EdgeSource<av::Packet>>(src_edge));
        if (params.count("url")) {
            parser->setURL(params.at("url"));
        }
        return parser;
    }

    void setURL(const std::string url) {
        rest_.setBaseURL(url);
        json_active_ = !url.empty();
        log_active_ = !json_active_;
    }
};

DECLNODE(parse_scte35, SCTE35Parser);
