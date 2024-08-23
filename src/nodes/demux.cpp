extern "C" {
#include <libavformat/avformat.h>
}
#include <unordered_set>
#include "node_common.hpp"

class StreamDemuxer: public NodeSingleInput<av::Packet>, public NodeWithOutputs<av::Packet> {
protected:
    std::unordered_map<int, std::shared_ptr<Edge<av::Packet>> > map_;
    std::unordered_set<int> video_streams_;
    bool report_unknown_stream_ = false; // TODO: setting this variable in factory function
    bool waiting_for_keyframe_ = false;
public:
    using NodeSingleInput::NodeSingleInput;
    void addStream(int stream_index, std::shared_ptr<Edge<av::Packet>> edge, bool is_video) {
        if (map_.count(stream_index)!=0) {
            throw Error("Adding stream requested but this stream_index already exists.");
        }
        map_[stream_index] = edge;
        if (is_video) {
            video_streams_.insert(stream_index);
        }
    }
    virtual void process() {
        av::Packet pkt = this->source_->get();
        auto iter = map_.find(pkt.streamIndex());
        if (iter != map_.end()) {
            if (iter->second != nullptr) { // null pointer means that we should ignore that packet
                if (waiting_for_keyframe_ && video_streams_.count(pkt.streamIndex()) && pkt.isKeyPacket()) {
                    waiting_for_keyframe_ = false;
                }
                if (!waiting_for_keyframe_) {
                    iter->second->enqueue(pkt);
                } // else drop
            }
        } else if (report_unknown_stream_) {
            logstream << "Unknown stream " << pkt.streamIndex() << std::endl;
        }
    }
    virtual void forEachOutput(std::function<void(Sink<av::Packet>*)> cb) {
        for (auto it: map_) {
            if (it.second==nullptr) continue;
            //std::unique_ptr<EdgeSink<av::Packet>> esink = it.second->makeSink();
            //cb(esink.get());
            EdgeSink<av::Packet> esink(it.second);
            cb(&esink);
        }
    }
    static std::shared_ptr<StreamDemuxer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<av::Packet>(params["src"]);
        auto r = std::make_shared<StreamDemuxer>(make_unique<EdgeSource<av::Packet>>(in_edge));
        in_edge->setConsumer(r);
        if (params.count("wait_for_keyframe")) {
            r->waiting_for_keyframe_ = params["wait_for_keyframe"];
        }
        
        // find source of streams:
        std::shared_ptr<IStreamsInput> streams_in = in_edge->findNodeUp<IStreamsInput>();
        if (streams_in == nullptr) {
            throw Error("Demuxer not connected to stream input!");
        }
        
        streams_in->discardAllStreams();
        
        std::string streams_filter;
        if (params.count("streams_filter")==1) {
            streams_filter = params["streams_filter"];
        }
        
        auto foundStream = [&](av::Stream stream, const std::string route_to) {
            std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(route_to);
            std::shared_ptr<InputStreamMetadata> md = edge->metadata<InputStreamMetadata>(true);
            streams_in->enableStream(stream.index());
            md->source_stream = stream;
            r->addStream(stream.index(), edge, stream.isVideo());
            edge->setProducer(r);
        };
        
        Parameters routing = params["routing"];
        for (Parameters::iterator route = routing.begin(); route != routing.end(); ++route) {
            // parse source stream specification:
            std::string sourcespec = route.key();
            std::string route_to = route.value();
            bool optional = false;
            
            if (sourcespec.length() < 1) {
                throw Error("Source specification too short");
            }
            if (sourcespec[0]=='?') {
                optional = true;
                sourcespec = sourcespec.substr(1);
                if (sourcespec.length() < 1) {
                    throw Error("Source specification too short");
                }
            }
            char typetag = sourcespec[0];
            AVMediaType mediatype;
            bool absolute_index = false;
            if ( (typetag=='v') || (typetag=='V') ) {
                mediatype = AVMEDIA_TYPE_VIDEO;
            } else if ( (typetag=='a') || (typetag=='A') ) {
                mediatype = AVMEDIA_TYPE_AUDIO;
            } else if (typetag == 'd' || typetag == 'D') {
                mediatype = AVMEDIA_TYPE_DATA;
            } else if (typetag>='0' && typetag<='9') {
                absolute_index = true;
            } else {
                throw Error(std::string("Invalid media type specification: ") + typetag);
            }
            
            bool found = false;
            if (absolute_index) {
                int sindex;
                try {
                    sindex = std::stoi(sourcespec);
                } catch (std::invalid_argument &e) {
                    throw Error("Invalid source specification: " + sourcespec);
                }
                if (sindex >= 0 && sindex < streams_in->streamsCount()) {
                    foundStream(streams_in->stream(sindex), route_to);
                    found = true;
                }
            } else {
                int sindex = 0;
                if (sourcespec.length() > 1) {
                    // override default ( = 0 ) index
                    if ( (sourcespec.length() == 2) || (sourcespec[1] != ':') ) {
                        throw Error("Invalid syntax: " + sourcespec);
                    }
                    std::string indexstr = sourcespec.substr(2);
                    try {
                        sindex = std::stoi(indexstr);
                    } catch (std::invalid_argument &e) {
                        throw Error("Invalid number " + indexstr + " in source specification: " + sourcespec);
                    }
                }
                int index_reached = 0;
                for (int i=0; i<streams_in->streamsCount(); i++) {
                    av::Stream stream = streams_in->stream(i);
                    bool matched = stream.mediaType()==mediatype;
                    if (matched && !streams_filter.empty()) {
                        int match_result = avformat_match_stream_specifier(streams_in->formatContext().raw(), stream.raw(), streams_filter.c_str());
                        if (match_result<0) {
                            throw Error("avformat_match_stream_specifier returned " + std::to_string(match_result) +
                                        " for stream index " + std::to_string(stream.index()));
                        }
                        matched = match_result > 0;
                    }
                    if (matched) {
                        if (index_reached == sindex) {
                            // found stream corresponding to specification!
                            foundStream(stream, route_to);
                            found = true;
                            break;
                        }
                        index_reached++;
                    }
                }
                
            }
            if (!found) {
                if (optional) {
                    logstream << "No optional stream " << sourcespec << " found.";
                } else {
                    throw Error("No stream " + sourcespec + " found!");
                }
            }
        }
        return r;
    }
};

DECLNODE(demux, StreamDemuxer);
