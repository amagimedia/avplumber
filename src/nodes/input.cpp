#include "node_common.hpp"
#include <libavutil/channel_layout.h>

class StreamInput: public NodeSingleOutput<av::Packet>, public IStreamsInput, public ReportsFinishByFlag,
                   public IStoppable, public IInterruptible, public IReturnsObjects {
protected:
    av::FormatContext ictx_;
    std::atomic_bool should_end_ {false};
    AVTS wait_start_;
    AVTS wait_max_ = AV_NOPTS_VALUE;
    av::Timestamp shift_ = NOTS;
    bool eof_mode_drain_ = false;
    bool eof_sent_ = false;
    av::Timestamp stop_delay_ = NOTS;
    av::Timestamp node_stop_ts_ = NOTS;
    Parameters streams_object_, programs_object_;
    void sendEofOnce() {
        if (!eof_sent_) {
            this->sink_->put(createEofPacket());
            eof_sent_ = true;
        }
    }
    void closeInput(bool warn = true) {
        try {
            ictx_.close();
        } catch (std::exception &e) {
            if (warn) {
                logstream << "WARNING: closing input failed: " << e.what();
            }
        }
    }
public:
    StreamInput(std::unique_ptr<Sink<av::Packet>> &&sink): NodeSingleOutput<av::Packet>(std::move(sink)) {
        ictx_.setInterruptCallback([this]() -> int {
            if ( (wait_max_!=AV_NOPTS_VALUE) && ((wallclock.pts() - wait_start_) > wait_max_) ) {
                logstream << "Timeout " << wait_max_ << " exceeded";
                this->finished_ = true;
                //closeInput();
                return 1;
            }
            if (should_end_) {
                //closeInput();
                return 1;
            }
            // closeInput() shouldn't be needed here because it's called in destructor
            // and in future maybe it will be called when null packet is got in process()
            return 0;
        });
    }
    av::FormatContext& ctx() {
        return ictx_;
    }
    virtual av::FormatContext& formatContext() {
        return ictx_;
    }
    virtual size_t streamsCount() {
        return ictx_.streamsCount();
    }
    virtual av::Stream stream(size_t id) {
        return ictx_.stream(id);
    }
    virtual void discardAllStreams() {
        for (size_t i=0; i<ictx_.streamsCount(); i++) {
            ictx_.stream(i).raw()->discard = AVDISCARD_ALL;
        }
    }
    virtual void enableStream(size_t index) {
        ictx_.stream(index).raw()->discard = AVDISCARD_DEFAULT;
    }
    virtual void process() {
        if (node_stop_ts_.isValid()) {
            if (wallclock.absolute_ts() >= node_stop_ts_) {
                logstream << "stop_delay elapsed, finishing input";
                this->finished_ = true;
            } else {
                wallclock.sleepms(10);
            }
            return;
        }
        wait_start_ = wallclock.pts();
        av::Packet pkt = ictx_.readPacket();
        if (pkt.isNull()) {
            logstream << "Got null packet";
            if (stop_delay_.isValid()) {
                sendEofOnce();
                node_stop_ts_ = addTS(wallclock.absolute_ts(), stop_delay_);
                logstream << "EOF reached, delaying input finish by " << stop_delay_;
                return;
            }
            if (eof_mode_drain_) {
                sendEofOnce();
                this->finished_ = true;
                return;
            }
            this->finished_ = true;
            //closeInput(true);
            // do not close input right now, otherwise segfaults happen because decoder tries to use demuxer data which is freed
            // TODO: check whether creating stream-independent decoder will help
        } else {
            if (!pkt.isComplete()) {
                logstream << "Got incomplete packet, dropping";
                return;
            }
            if (pkt.dts().isNoPts() && pkt.pts().isNoPts()) {
                logstream << "Got packet without PTS & DTS, dropping";
                return;
            }
            if (should_end_ || this->finished_) return;
        }
        //logstream << "PKT OUT";
        #if 0
        if (!shift_) {
            shift_ = addTS(negateTS(pkt.dts()), av::Timestamp(10, {1,1}));
            logstream << "First PTS " << pkt.pts() << " DTS " << pkt.dts();
            logstream << "Set input shift to " << shift_;
        }
        pkt.setDts(addTS(pkt.dts(), shift_));
        pkt.setPts(addTS(pkt.pts(), shift_));
        #endif
        this->sink_->put(pkt);
    }
    virtual void stop() {
        logstream << "Setting should_end_ to true";
        should_end_ = true;
        this->finished_ = true;
    }
    virtual void interrupt() {
        stop();
    }
    void setTimeout(int64_t timeout) {
        if (timeout<0) {
            wait_max_ = AV_NOPTS_VALUE;
            return;
        }
        wait_start_ = wallclock.pts();
        wait_max_ = timeout * wallclock.timeBase().den / wallclock.timeBase().num;
        logstream << "Set wait_max_ to " << wait_max_ << "s";
        //ictx_.setSocketTimeout(timeout);
    }
    virtual ~StreamInput() {
        #if 0 // see comment in process() "Got null packet"
        if (ictx_.isOpened()) {
            logstream << "BUG: input context still opened in destructor, closing";
            closeInput(true);
        }
        #else
        closeInput(true);
        #endif
    }
    static std::shared_ptr<StreamInput> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        auto r = std::make_shared<StreamInput>(make_unique<EdgeSink<av::Packet>>(edge));
        return r;
    }
    virtual void init(EdgeManager &edges, const Parameters &params) {
        NodeSingleOutput<av::Packet>::init(edges, params);
        av::InputFormat ifmt;
        if (params.count("format") > 0) {
            ifmt.setFormat(params["format"]);
        }
        av::Dictionary opts;
        if (params.count("options") > 0) {
            opts = parametersToDict(params["options"]);
        }
        int timeout = 5;
        if (params.count("timeout") > 0) {
            timeout = (int)params["timeout"];
        }
        int initial_timeout = timeout;
        if (params.count("initial_timeout") > 0) {
            initial_timeout = (int)params["initial_timeout"];
        }
        setTimeout(initial_timeout);
        ictx_.openInput(params["url"], opts, ifmt);
        ictx_.findStreamInfo();
        logstream << "Opened URL " << params["url"] << " . Streams:";
        for (unsigned i=0; i<ictx_.streamsCount(); i++) {
            av::Stream stream = ictx_.stream(i);
            logstream << i << ": " << ( stream.isVideo() ? "video" : (stream.isAudio() ? "audio" : "???") ) << " tb " << stream.timeBase();
        }
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        edge->setProducer(this->shared_from_this());
        setTimeout(timeout);
        // Default drain: on EOF/null packet, enqueue an EOF marker so downstream can flush (VOD / file).
        eof_mode_drain_ = true;
        if (params.count("eof_mode") > 0) {
            std::string eof_mode = params["eof_mode"];
            if (eof_mode == "drain") {
                eof_mode_drain_ = true;
            } else if (eof_mode == "none") {
                eof_mode_drain_ = false;
            } else {
                throw Error("Unknown eof_mode " + eof_mode + " (expected drain or none)");
            }
        }
        if (params.count("stop_delay") > 0) {
            stop_delay_ = av::Timestamp(params["stop_delay"].get<int64_t>(), {1, 1000});
        }

        for (size_t i=0; i<ictx_.streamsCount(); i++) {
            Parameters obj;
            av::Stream stream = ictx_.stream(i);
            obj["index"] = i;
            obj["type"] = mediaTypeToString(stream.mediaType());
            AVCodecParameters &cpar = *stream.raw()->codecpar;
            obj["codec"] = avcodec_get_name(cpar.codec_id);
            if (stream.isVideo()) {
                obj["fps"] = std::to_string(stream.frameRate().getNumerator()) + '/' + std::to_string(stream.frameRate().getDenominator());
                obj["width"] = cpar.width;
                obj["height"] = cpar.height;
                obj["pixel_format"] = av::PixelFormat((AVPixelFormat)cpar.format).name();
                obj["field_order"] = fieldOrderToString(cpar.field_order);
                obj["sar"] = std::to_string(cpar.sample_aspect_ratio.num) + '/' + std::to_string(cpar.sample_aspect_ratio.den);
            } else if (stream.isAudio()) {
                obj["sample_rate"] = cpar.sample_rate;
                char chlayout[64] = {0};
#if API_NEW_CHANNEL_LAYOUT
                av_channel_layout_describe(&cpar.ch_layout, chlayout, 63);
                obj["channels_count"] = cpar.ch_layout.nb_channels;
#else
                av_get_channel_layout_string(chlayout, 63, cpar.channels, cpar.channel_layout);
                obj["channels_count"] = cpar.channels;
#endif
                obj["channel_layout"] = chlayout;
                obj["sample_format"] = av::SampleFormat((AVSampleFormat)cpar.format).name();

                Parameters metadata_obj;
                av::Dictionary dict = av::Dictionary(stream.raw()->metadata, false);
                for (auto &item : dict) {
                    metadata_obj[item.key()] = item.value();
                }
                
                if (!metadata_obj.empty()) {
                    obj["metadata"] = metadata_obj;
                }
            }
            streams_object_.push_back(obj);
        }
        for (size_t i=0; i<ictx_.raw()->nb_programs; i++) {
            AVProgram *program = ictx_.raw()->programs[i];
            Parameters obj;
            obj["index"] = i;
            Parameters streams;
            for (size_t j=0; j<program->nb_stream_indexes; j++) {
                streams.push_back(program->stream_index[j]);
            }
            obj["streams"] = streams;
            programs_object_.push_back(obj);
        }
    }
    virtual Parameters getObject(const std::string name) {
        if (name=="streams") {
            return streams_object_;
        } else if (name=="programs") {
            return programs_object_;
        } else {
            throw Error("Unknown object to get");
        }
    }
};

DECLNODE(input, StreamInput);
