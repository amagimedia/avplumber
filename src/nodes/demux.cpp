#include <unordered_set>
#include <atomic>
#include "node_common.hpp"
#include "stream_routing.hpp"

class StreamDemuxer: public NodeSingleInput<av::Packet>, public NodeWithOutputs<av::Packet>, public IReportsFinish {
protected:
    StreamRoutingSink routes_;
    std::unordered_set<int> video_streams_;
    bool report_unknown_stream_ = false; // TODO: setting this variable in factory function
    bool waiting_for_keyframe_ = false;
    bool stop_on_eof_ = true;
    std::atomic_bool finished_{false};
public:
    using NodeSingleInput::NodeSingleInput;
    bool consumeEofIfPresent() override {
        return false;
    }
    bool finished() override {
        return finished_.load();
    }
    void stop() override {
        // Publish termination before waking the read: the wake-up is consumed
        // once, so the worker must not enter another blocking read afterwards.
        finished_ = true;
        NodeSingleInput<av::Packet>::stop();
    }
    virtual void process() {
        if (finished()) {
            return;
        }
        av::Packet pkt = this->source_->get();
        if (isEofMarker(pkt)) {
            // eof marker, put on each queue
            routes_.put(pkt);
            // EOF may arrive while stop() wakes a blocked read. Never undo
            // that explicit stop when keeping an interactive input seekable.
            if (stop_on_eof_) {
                finished_ = true;
            }
        } else if (routes_.routes(pkt.streamIndex())) {
            if (waiting_for_keyframe_ && video_streams_.count(pkt.streamIndex()) && pkt.isKeyPacket()) {
                waiting_for_keyframe_ = false;
            }
            if (!waiting_for_keyframe_) {
                routes_.put(pkt);
            } // else drop
        } else if (report_unknown_stream_) {
            logstream << "Unknown stream " << pkt.streamIndex() << std::endl;
        }
    }
    virtual void forEachOutput(std::function<void(Sink<av::Packet>*)> cb) {
        routes_.forEachOutput(cb);
    }
    static std::shared_ptr<StreamDemuxer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<av::Packet>(params["src"]);
        auto r = std::make_shared<StreamDemuxer>(make_unique<EdgeSource<av::Packet>>(in_edge));
        r->auto_eof_ = false;
        in_edge->setConsumer(r);
        if (params.count("wait_for_keyframe")) {
            r->waiting_for_keyframe_ = params["wait_for_keyframe"];
        }
        if (params.count("stop_on_eof")) {
            r->stop_on_eof_ = params["stop_on_eof"];
        }

        // find source of streams:
        std::shared_ptr<IStreamsInput> streams_in = in_edge->findNodeUp<IStreamsInput>();
        if (streams_in == nullptr) {
            throw Error("Demuxer not connected to stream input!");
        }

        connectStreamRoutes(edges, *streams_in, r, params, [&r](av::Stream stream, std::shared_ptr<Edge<av::Packet>> edge) {
            r->routes_.add(stream.index(), edge);
            if (stream.isVideo()) {
                r->video_streams_.insert(stream.index());
            }
        });
        return r;
    }
};

DECLNODE(demux, StreamDemuxer);
