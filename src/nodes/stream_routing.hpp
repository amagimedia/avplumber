#pragma once
extern "C" {
#include <libavformat/avformat.h>
}
#include <unordered_map>
#include "node_common.hpp"

// Packet edges keyed by stream index: the outputs of a node that gives each stream of a
// container its own edge (demux, input_rec with `routing`). A packet goes to the edge of
// its stream, an EOF marker to every edge as that stream's EOF; a packet of a stream
// without an edge is dropped.
class StreamRoutingSink: public Sink<av::Packet> {
protected:
    std::unordered_map<int, std::shared_ptr<Edge<av::Packet>>> edges_;
public:
    void add(int stream_index, std::shared_ptr<Edge<av::Packet>> edge) {
        if (!edges_.emplace(stream_index, edge).second) {
            throw Error("Adding stream requested but this stream_index already exists.");
        }
    }
    bool routes(int stream_index) const {
        return edges_.count(stream_index) != 0;
    }
    virtual bool put(const av::Packet &pkt, bool drop_if_full = false) override {
        if (isEofMarker(pkt)) {
            for (auto &route: edges_) {
                route.second->enqueue(createEofPacket(route.first));
            }
            return true;
        }
        auto route = edges_.find(pkt.streamIndex());
        if (route == edges_.end()) {
            return true;
        }
        if (!drop_if_full) {
            route->second->enqueue(pkt);
            return true;
        }
        if (route->second->try_enqueue(pkt)) {
            return true;
        }
        route->second->countDrop();
        return false;
    }
    void forEachOutput(std::function<void(Sink<av::Packet>*)> cb) {
        for (auto &route: edges_) {
            EdgeSink<av::Packet> sink(route.second);
            cb(&sink);
        }
    }
};

// The index of the stream of `input` that `spec` names, or -1 when there is none.
// "v", "a", "d" name the first video, audio or data stream, "v:1" the second video stream
// and so on; a number names the stream with that index. With `streams_filter` (an FFmpeg
// stream specifier, e.g. "p:1" for a program) only the streams it matches are counted.
inline int findStream(IStreamsInput &input, const std::string &spec, const std::string &streams_filter) {
    char typetag = spec[0];
    AVMediaType mediatype = AVMEDIA_TYPE_UNKNOWN;
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
    const int streams_count = input.streamsCount();
    if (absolute_index) {
        int sindex;
        try {
            sindex = std::stoi(spec);
        } catch (std::invalid_argument &e) {
            throw Error("Invalid source specification: " + spec);
        }
        return (sindex >= 0 && sindex < streams_count) ? sindex : -1;
    }
    int sindex = 0;
    if (spec.length() > 1) {
        // override default ( = 0 ) index
        if ( (spec.length() == 2) || (spec[1] != ':') ) {
            throw Error("Invalid syntax: " + spec);
        }
        std::string indexstr = spec.substr(2);
        try {
            sindex = std::stoi(indexstr);
        } catch (std::invalid_argument &e) {
            throw Error("Invalid number " + indexstr + " in source specification: " + spec);
        }
    }
    int index_reached = 0;
    for (int i=0; i<streams_count; i++) {
        av::Stream stream = input.stream(i);
        bool matched = stream.mediaType()==mediatype;
        if (matched && !streams_filter.empty()) {
            int match_result = avformat_match_stream_specifier(input.formatContext().raw(), stream.raw(), streams_filter.c_str());
            if (match_result<0) {
                throw Error("avformat_match_stream_specifier returned " + std::to_string(match_result) +
                            " for stream index " + std::to_string(stream.index()));
            }
            matched = match_result > 0;
        }
        if (matched) {
            if (index_reached == sindex) {
                return i;
            }
            index_reached++;
        }
    }
    return -1;
}

// Connects the routes of `params["routing"]` (stream specifier => edge name; a "?" in front
// of the specifier marks a route that may have no stream) to `producer`: `input` keeps only
// the routed streams enabled, each routed stream's edge carries it as InputStreamMetadata
// for the decoders, and `on_route` receives the stream and its edge. A missing optional
// route leaves an edge of the packet type whose producer is finished, so its consumers see
// EOF instead of waiting forever; a missing required route throws. `params["streams_filter"]`
// is passed to findStream().
inline void connectStreamRoutes(EdgeManager &edges, IStreamsInput &input, std::shared_ptr<Node> producer, const Parameters &params,
                                const std::function<void(av::Stream, std::shared_ptr<Edge<av::Packet>>)> &on_route) {
    input.discardAllStreams();

    std::string streams_filter;
    if (params.count("streams_filter")==1) {
        streams_filter = params["streams_filter"];
    }

    const Parameters &routing = params["routing"];
    for (Parameters::const_iterator route = routing.begin(); route != routing.end(); ++route) {
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
        int index = findStream(input, sourcespec, streams_filter);
        if (index >= 0) {
            av::Stream stream = input.stream(index);
            std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(route_to);
            edge->metadata<InputStreamMetadata>(true)->source_stream = stream;
            input.enableStream(index);
            edge->setProducer(producer);
            on_route(stream, edge);
        } else if (optional) {
            logstream << "No optional stream " << sourcespec << " found.";
            edges.find<av::Packet>(route_to)->finishProducer();
        } else {
            throw Error("No stream " + sourcespec + " found!");
        }
    }
}
