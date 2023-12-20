#include "node_common.hpp"
#include "../MultiEventWait.hpp"

class StreamMuxer: public NodeSingleOutput<av::Packet>, public IStoppable, public IMuxer, public NodeDoesNotBuffer {
private:
    struct StreamInfo {
        int stream_index = -1;
        //av::Stream stream;
        std::shared_ptr<Edge<av::Packet>> edge;
        AVTS idle_since = AV_NOPTS_VALUE; // timebase: milliseconds
        bool idle = false; // flags that this input was idle in latest iteration of process()
        bool warned = false;
        av::Timestamp prev_dts = NOTS;
        av::Rational stream_tb {0, 0};
        AVTS shift = 0;
        size_t shifted_for = 0; // unit: packets count
    };
    std::vector<StreamInfo> streams_;
    Event stop_event_;
    AVTS sync_wait_max_ms_ = 2500;
    std::unique_ptr<MultiEventWait> event_wait_;
    bool fix_timestamps_ = false;
    av::Timestamp global_shift_ = {0, {1, 1}};
    void calculateGlobalShift() {
        bool severe = false;
        for (StreamInfo &s: streams_) {
            if (s.shifted_for > 10) {
                severe = true;
                break;
            }
        }
        if (!severe) {
            return;
        }
        av::Timestamp max_shift = {0, {1, 1}};
        av::Rational max_tb = {0, 0};
        for (StreamInfo &s: streams_) {
            if (s.prev_dts.isNoPts()) {
                continue;
            }
            av::Rational tb = s.prev_dts.timebase();
            av::Timestamp shift = {s.shift, tb};
            logstream << "Stream " << s.stream_index << " was shifted by " << shift;
            if (shift > max_shift) {
                max_shift = shift;
            }
            if ( (max_tb.getNumerator()==0) || (max_tb < tb) ) {
                max_tb = tb;
            }
            s.shift = 0;
            s.shifted_for = 0;
        }
        av::Timestamp target_shift = addTS(max_shift, global_shift_);
        global_shift_ = rescaleTS(target_shift, max_tb); // use least precision timebase to prevent A-V desync
        if (rescaleTS(global_shift_, target_shift.timebase()) < target_shift) {
            // do not allow rounding down
            global_shift_ = {global_shift_.timestamp()+1, max_tb};
        }
        logstream << "Shifting everything by " << global_shift_;
    }
public:
    StreamMuxer(std::unique_ptr<Sink<av::Packet>> &&sink): NodeSingleOutput<av::Packet>(std::move(sink)) {
    }
    void addStream(int stream_index, std::shared_ptr<Edge<av::Packet>> edge) {
        streams_.emplace_back();
        streams_.back().stream_index = stream_index;
        streams_.back().edge = edge;
    }
    void prepare() {
        std::vector<Event*> events(streams_.size()+1);
        for (size_t i=0; i<streams_.size(); i++) {
            std::shared_ptr<Edge<av::Packet>> edge = streams_[i].edge;
            events[i] = &(edge->producedEvent());
            edge->setConsumer(this->shared_from_this());
        }
        events[streams_.size()] = &stop_event_;
        event_wait_ = make_unique<MultiEventWait>(events);
    }
    virtual void stop() {
        stop_event_.signal();
    }
    virtual void process() {
        // find earliest packet (least DTS) in streams:
        av::Timestamp least_ts = NOTS;
        StreamInfo* least_ts_si = nullptr;
        unsigned candidates = 0;
        for (StreamInfo &s: streams_) {
            s.idle = true;
            av::Packet *pkt = s.edge->peek();
            if (pkt==nullptr) continue;
            av::Timestamp pkt_ts = pkt->dts();
            if (pkt_ts.isNoPts()) pkt_ts = pkt->pts();
            if (pkt_ts.isNoPts()) {
                // packet without PTS
                // drop as invalid
                s.edge->pop();
                continue;
            }
            s.idle = false;
            s.idle_since = AV_NOPTS_VALUE;
            s.warned = false;
            candidates++;
            if ( least_ts.isNoPts() || (pkt_ts < least_ts) ) {
                least_ts = pkt_ts;
                least_ts_si = &s;
            }
        }
        bool have_all = (candidates == streams_.size());
        bool should_emit = false;
        if (least_ts_si==nullptr) {
            // no packet available in queue, wait for it
            should_emit = false;
            event_wait_->wait();
        } else if (sync_wait_max_ms_>0 && !have_all) {
            AVTS ts = least_ts.timestamp(av::Rational(1, 1000));
            AVTS max_wait = -1;
            for (StreamInfo &s: streams_) {
                if (s.idle) {
                    if (s.idle_since==AV_NOPTS_VALUE) {
                        s.idle_since = ts;
                        if (max_wait > sync_wait_max_ms_) {
                            logstream << "BUG: max_wait = " << max_wait;
                        }
                        max_wait = sync_wait_max_ms_;
                    } else {
                        AVTS diff = ts-s.idle_since;
                        if (diff < 0) {
                            logstream << "Warning: Time went backwards in muxer: [stream_index]ms: [" << s.stream_index << "]" << s.idle_since <<
                                " -> [" << least_ts_si->stream_index << "]" << ts;
                            diff = 0;
                        }
                        AVTS to_wait = sync_wait_max_ms_ - diff;
                        if (to_wait <= 0) {
                            // we waited too long -> pretend that this stream is not idle
                            candidates++;
                            if (!s.warned) {
                                logstream << "Warning: sync wait timeout exceeded: " << diff << "ms, stream " << s.stream_index;
                                s.warned = true;
                            }
                        } else {
                            // find maximum wait time and store it in max_wait
                            if (to_wait > max_wait) {
                                max_wait = to_wait;
                            }
                        }
                    }
                }
            }
            if (candidates == streams_.size()) {
                should_emit = true;
            } else {
                if ( (max_wait<=0) || (max_wait > sync_wait_max_ms_) ) {
                    logstream << "BUG: max_wait = " << max_wait << "ms";
                    if (max_wait > sync_wait_max_ms_) {
                        max_wait = sync_wait_max_ms_;
                    }
                }
                // wait for next packet
                // if timeout occured (no more packets arrived), emit without re-checking
                should_emit = !(event_wait_->wait(max_wait));
            }
        } else { // have_all==true
            should_emit = true;
        }
        if (should_emit) {
            // set stream index & emit packet
            StreamInfo &s = *least_ts_si;
            av::Packet *pkt = s.edge->peek();
            if (pkt!=nullptr) {
                if (s.stream_index >= 0 && pkt->dts().isValid()) {
                    if (fix_timestamps_) {
                        // we do it here. otherwise, in output node, avcpp will do it (FormatContext::writePacket) and sabotage our forcing of increasing DTSes
                        pkt->setTimeBase(s.stream_tb);
                        
                        if (!s.prev_dts.isNoPts()) {
                            /*if (pkt->dts().timebase() != s.prev_dts.timebase()) {
                                logstream << "Timebase changed in muxer in stream " << s.stream_index << ": " << s.prev_dts << " -> " << pkt->dts();
                                throw Error("Timebase changed in muxer! Nothing to do here.");
                            }*/
                            if (global_shift_.timestamp() != 0) {
                                pkt->setDts(addTSSameTB(pkt->dts(), rescaleTS(global_shift_, pkt->dts().timebase())));
                                if (pkt->pts().isValid()) {
                                    pkt->setPts(addTSSameTB(pkt->pts(), rescaleTS(global_shift_, pkt->pts().timebase())));
                                }
                            }
                            if (pkt->dts().timestamp() <= s.prev_dts.timestamp()) {
                                logstream << "Non-increasing DTSes in stream " << s.stream_index << ": " << s.prev_dts << " -> " << pkt->dts() << ", fixing.";
                                AVTS newts = s.prev_dts.timestamp()+1;
                                s.shift = newts - pkt->dts().timestamp();
                                s.shifted_for++;
                                pkt->setDts({ newts, s.prev_dts.timebase() });
                            } else {
                                s.shift = 0;
                                s.shifted_for = 0;
                            }
                        }
                        if (pkt->pts().isValid() && (pkt->pts() < pkt->dts())) {
                            logstream << "PTS < DTS, " << pkt->pts() << " < " << pkt->dts() << ", fixing.";
                            pkt->setPts(pkt->dts());
                        }
                        s.prev_dts = pkt->dts();
                        calculateGlobalShift();
                    }
                    pkt->setStreamIndex(s.stream_index);
                    //logstream << "mux out: stream " << s.stream_index << ", PTS = " << pkt->pts() << std::endl;
                    sink_->put(*pkt);
                } else {
                    logstream << "Dropping packet which would go to stream index " << s.stream_index << " dts " << pkt->dts();
                }
                s.edge->pop();
            }
        }
    }
    virtual void initFromFormatContext(av::FormatContext &octx) {
        for (StreamInfo &s: streams_) {
            std::shared_ptr<IEncoder> enc = s.edge->findNodeUp<IEncoder>();
            if (enc==nullptr) {
                throw Error("Muxer init failed: No encoder above in chain!");
            }
            av::Codec &codec = enc->encodingCodec();
            av::Stream deststream = octx.addStream(codec);
            s.stream_index = deststream.index();
            enc->setOutput(deststream, octx);
        }
    }
    virtual void initFromFormatContextPostOpenPreWriteHeader(av::FormatContext &octx) {
        for (StreamInfo &s: streams_) {
            std::shared_ptr<IEncoder> enc = s.edge->findNodeUp<IEncoder>();
            if (enc==nullptr) {
                throw Error("Muxer post-open-pre-writeheader init failed: No encoder above in chain!");
            }
            av::Stream deststream = octx.stream(s.stream_index);
            enc->openEncoder(deststream);
        }
    }
    virtual void initFromFormatContextPostOpen(av::FormatContext &octx) {
        for (StreamInfo &s: streams_) {
            std::shared_ptr<IEncoder> enc = s.edge->findNodeUp<IEncoder>();
            if (enc==nullptr) {
                throw Error("Muxer post-open init failed: No encoder above in chain!");
            }
            av::Stream deststream = octx.stream(s.stream_index);
            s.stream_tb = deststream.timeBase();
            enc->setOutputPostOpen(deststream, octx);
        }
    }
    static std::shared_ptr<StreamMuxer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> out_edge = edges.find<av::Packet>(params["dst"]);
        auto r = std::make_shared<StreamMuxer>(make_unique<EdgeSink<av::Packet>>(out_edge));
        if (params.count("fix_timestamps")) {
            r->fix_timestamps_ = params["fix_timestamps"];
        }
        if (params.count("ts_sort_wait")) {
            r->sync_wait_max_ms_ = params["ts_sort_wait"].get<float>() * 1000.0;
        }
        for (std::string sname: params["src"]) {
            std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(sname);
            r->addStream(-1, edge);
        }
        r->prepare();
        out_edge->setProducer(r);
        return r;
    }
};

DECLNODE(mux, StreamMuxer);
