#include "node_common.hpp"

class CCDataExtractor: public NodeSISO<av::VideoFrame, av::Packet>, public IPauseable {
protected:
    std::atomic_bool paused_ = false;

public:
    using NodeSISO<av::VideoFrame, av::Packet>::NodeSISO;
    virtual void process() override {
        av::VideoFrame vfrm = this->source_->get();
        if (paused_) return;
        if (!vfrm.isComplete()) return;
        AVFrame* frm = vfrm.raw();
        for (int i=0; i<frm->nb_side_data; i++) {
            auto sd = frm->side_data[i];
            int start_index = -1;
            int end_index = -1;
            if (sd->type == AV_FRAME_DATA_A53_CC) {
                if (sd->data == nullptr) continue;
                /*for (int j=0; j<sd->size; j++) {
                    if (sd->data[j] == 255) {
                        if (start_index<0) {
                            start_index = j+1;
                        } else if (end_index<0) {
                            end_index = j;
                            break;
                        }
                    }
                }
                logstream << start_index << ".." << end_index;
                if (start_index<0 || end_index<0) continue;
                std::vector<uint8_t> data(sd->data + start_index, sd->data + end_index);*/
                std::vector<uint8_t> data(sd->data, sd->data + sd->size);
                av::Packet pkt = av::Packet(data);
                pkt.setPts(vfrm.pts());
                pkt.setDts(vfrm.pts());
                this->sink_->put(pkt);
            }
        }
    }
    static std::shared_ptr<CCDataExtractor> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::VideoFrame>> src_edge = edges.find<av::VideoFrame>(params["src"]);
        std::shared_ptr<Edge<av::Packet>> dst_edge = edges.find<av::Packet>(params["dst"]);
        auto r = std::make_shared<CCDataExtractor>(make_unique<EdgeSource<av::VideoFrame>>(src_edge), make_unique<EdgeSink<av::Packet>>(dst_edge));

        return r;
    }

    void pause() override {
        paused_ = true;
    }

    void resume() override {
        paused_ = false;
    }
};

DECLNODE(extract_cc_data, CCDataExtractor);
