#include "node_common.hpp"
#include <mutex>
#include <iostream>
#include <fstream>

#pragma pack(push)
#pragma pack(1)
struct SeekTableEntry {
    int64_t timestamp_ms;
    uint64_t bytes;
};

struct TSOffsetEntry {
    int64_t changed_at;
    int64_t input_ts_diff;
    int64_t wallclock_diff;
    int64_t output_ts_diff;
};

#pragma pack(pop)

class StreamInput: public NodeSingleOutput<av::Packet>, public IStreamsInput, public ReportsFinishByFlag,
                   public IStoppable, public IInterruptible, public IReturnsObjects, public ISeekAt {
protected:
    av::FormatContext ictx_;
    std::atomic_bool should_end_ {false};
    AVTS wait_start_;
    AVTS wait_max_ = AV_NOPTS_VALUE;

    std::string seek_table_url_;
    std::mutex seek_table_mutex_;
    std::vector<SeekTableEntry> seek_table_;
    std::mutex seek_at_mutex_;
    std::list<std::pair<av::Timestamp, StreamTarget>> seek_at_table_;

    std::string ts_offsets_url_;
    std::mutex ts_offsets_mutex_;
    std::vector<TSOffsetEntry> ts_offsets_;

    StreamTarget seek_target_;
    bool need_seek_ = false;
    bool auto_resume_after_seek_ = false;
    std::mutex seek_mutex_;
    Event seek_resume_;
    float preseek_ = 0;
    int video_stream_ = -1;

    av::Timestamp shift_ = NOTS;
    Parameters streams_object_, programs_object_;
    void closeInput(bool warn = true) {
        try {
            ictx_.close();
        } catch (std::exception &e) {
            if (warn) {
                logstream << "WARNING: closing input failed: " << e.what();
            }
        }
    }
private:
    void resolveSeekTarget(StreamTarget& st) {
        auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

        if (seek_table_.empty()) {
            // no seek table available, only seeks by time
            return;
        }

        SeekTableEntry ste;

        int64_t req_ts = st.ts.timestamp();

        auto it = std::lower_bound(seek_table_.cbegin(), seek_table_.cend(), req_ts, [](const SeekTableEntry& e, int64_t value) {
            return e.timestamp_ms < value;
        });

        if (it == seek_table_.cend()) {
            it = std::prev(seek_table_.cend());
        }

        st.ts = NOTS;
        st.bytes = it->bytes;
    }

    virtual void fixInputTimestamp(StreamTarget& ts) override
    {
        auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);

        if (!ts_offsets_.empty()) {
            // correct timestamp
            if (ts.wallclock) {
                // convert wallclock ts -> output ts
                int64_t new_ts = rescaleTS(ts.ts, {1, 1000}).timestamp();
                auto it = std::upper_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), new_ts, [](int64_t value, const TSOffsetEntry& e) {
                    return value < e.changed_at - e.wallclock_diff;
                });
                if (it != ts_offsets_.cbegin()) {
                    it = std::prev(it);
                }

                new_ts += it->wallclock_diff;
                ts.wallclock = false;
                ts.ts = av::Timestamp(new_ts, {1, 1000});
            } else {
                // convert input ts -> output ts
                int64_t new_ts = rescaleTS(ts.ts, {1, 1000}).timestamp();
                auto it = std::upper_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), new_ts, [](int64_t value, const TSOffsetEntry& e) {
                    return value <= e.changed_at - e.input_ts_diff;
                });
                if (it != ts_offsets_.cbegin()) {
                    it = std::prev(it);
                }

                new_ts += it->input_ts_diff;
                ts.ts = av::Timestamp(new_ts, {1, 1000});
            }
        }
    }

    void setFrameTimestamps(av::VideoFrame& frm, const av::Timestamp& ts_in, const av::Timestamp& ts_out, const av::Timestamp& ts_wallclock) {
        AVFrame* frame = frm.raw();

        auto set_ts = [frame](const char* metadata_name, const av::Timestamp& ts) {
            std::string value;
            if (ts.isValid()) {
                long t = ts.seconds() * 1000;
                int ms = t % 1000; t /= 1000;
                int s = t % 60; t /= 60;
                int m = t % 60; t /= 60;
                int h = t % 24; t /= 24;
                std::string date;
                if (t > 0) {
                    std::stringstream s;
                    time_t tt = (time_t)ts.seconds();
                    tm tm;
                    s << std::put_time(gmtime_r(&tt, &tm), "%Y-%m-%d ");
                    date = s.str();
                }

                char result[64];
                sprintf(result, "%s%02d:%02d:%02d.%03d", date.c_str(), h, m, s, ms);
                value = result;
            } else {
                value = "unknown";
            }
            av_dict_set(&frame->metadata, metadata_name, value.c_str(), 0);
        };

        set_ts("input_ts", ts_in);
        set_ts("output_ts", ts_out);
        set_ts("wallclock_ts", ts_wallclock);
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
    virtual void setFrameMetadataTimestamps(av::VideoFrame& frame) override {
        auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);

        if (!ts_offsets_.empty()) {
            av::Timestamp input_ts = frame.pts();
            av::Timestamp output_ts = frame.pts();
            av::Timestamp wallclock_ts = frame.pts();

            auto it = std::upper_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), rescaleTS(output_ts, {1, 1000}).timestamp(), [](int64_t value, const TSOffsetEntry& e) {
                return value <= e.changed_at;
            });

            if (it != ts_offsets_.cbegin()) {
                it = std::prev(it);
            }

            int64_t ts_diff = 0;
            if (it != ts_offsets_.cend()) {
                input_ts = addTS(input_ts, negateTS(av::Timestamp(it->input_ts_diff, {1, 1000})));
                wallclock_ts = addTS(wallclock_ts, negateTS(av::Timestamp(it->wallclock_diff, {1, 1000})));
            }

            setFrameTimestamps(frame, input_ts, output_ts, wallclock_ts);
        }
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
    virtual void seekAndPause(StreamTarget target) {
        auto lock = std::lock_guard<decltype(seek_mutex_)>(seek_mutex_);
        seek_target_ = target;
        resolveSeekTarget(seek_target_);
        auto_resume_after_seek_ = false;
        need_seek_ = true;
    }
    virtual void seek(StreamTarget target) {
        auto lock = std::lock_guard<decltype(seek_mutex_)>(seek_mutex_);
        seek_target_ = target;
        resolveSeekTarget(seek_target_);
        auto_resume_after_seek_ = true;
        need_seek_ = true;
    }
    virtual void resumeAfterSeek() {
        seek_resume_.signal();
    }
    virtual void seekAtAdd(const StreamTarget& when, const StreamTarget& target) override {
        auto lock = std::lock_guard<decltype(seek_at_mutex_)>(seek_at_mutex_);
        seek_at_table_.push_back(std::make_pair(when.ts, target));
    }
    virtual void seekAtClear() override {
        auto lock = std::lock_guard<decltype(seek_at_mutex_)>(seek_at_mutex_);
        seek_at_table_.clear();
    }
    void loadSeekTable() {
        if (seek_table_url_.empty())
            return;
        std::ifstream f(seek_table_url_, std::ios::binary);
        if (f) {
            auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);
            size_t start = sizeof(SeekTableEntry) * seek_table_.size();
            f.seekg(0, std::ios::end);
            size_t count = static_cast<size_t>(f.tellg()) - start;
            f.seekg(start);
            std::vector<char> buffer(count);
            f.read(buffer.data(), count);
            count /= sizeof(SeekTableEntry);
            seek_table_.reserve(seek_table_.size() + count);
            for (int idx = 0; idx < count; ++idx) {
                SeekTableEntry* entry = (SeekTableEntry*)(buffer.data()) + idx;
                seek_table_.push_back(*entry);
            }
        }
    }
    void loadTimestampOffsets() {
        if (ts_offsets_url_.empty())
            return;
        std::ifstream f(ts_offsets_url_, std::ios::binary);
        if (f) {
            auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);
            size_t start = sizeof(TSOffsetEntry) * ts_offsets_.size();
            f.seekg(0, std::ios::end);
            size_t count = static_cast<size_t>(f.tellg()) - start;
            f.seekg(start);
            std::vector<char> buffer(count);
            f.read(buffer.data(), count);
            count /= sizeof(TSOffsetEntry);
            ts_offsets_.reserve(ts_offsets_.size() + count);
            for (int idx = 0; idx < count; ++idx) {
                TSOffsetEntry* entry = (TSOffsetEntry*)(buffer.data()) + idx;
                ts_offsets_.push_back(*entry);
            }
        }
    }
    virtual void process() {
        bool seeked = false;
        {
            auto lock = std::lock_guard<decltype(seek_mutex_)>(seek_mutex_);
            if (need_seek_) {
                //ictx_.flush();
                //avformat_flush(ictx_.raw());
                if (seek_target_.ts.isValid()) {
                    // seek by timestamp, no seek table used
                    av::Rational tb = (video_stream_>=0) ? ictx_.stream(video_stream_).timeBase() : av::Rational(AV_TIME_BASE_Q);
                    AVTS preseek = std::round(preseek_ * float(tb.getDenominator()) / float(tb.getNumerator()));
                    // TODO: preseek may seek too far before needed timestamp, discard non-key frames before first keyframe in such case
                    AVTS ts = seek_target_.ts.timestamp(tb) - preseek;
                    /*ictx_.seek(ts, video_stream_, AVSEEK_FLAG_BACKWARD);*/
                    int ret = avformat_seek_file(ictx_.raw(), video_stream_, INT64_MIN, ts, ts + int(0.04f/tb.getDouble()+0.5f), 0);
                    //logstream << "video_stream_ " << video_stream_ << " timestamp " << ts;
                    //int ret = av_seek_frame(ictx_.raw(), video_stream_, ts, AVSEEK_FLAG_BACKWARD);
                    if (ret < 0) {
                        logstream << "av seek returned " << ret;
                    }
                } else {
                    logstream << "video_stream_ " << video_stream_ << " seek to position: " << seek_target_.bytes;
                    ictx_.seek(seek_target_.bytes, -1, AVSEEK_FLAG_BYTE);
                }
                need_seek_ = false;
                seeked = true;
            }
        }
        if (seeked && !auto_resume_after_seek_) {
            seek_resume_.wait();
        }
        wait_start_ = wallclock.pts();
        av::Packet pkt = ictx_.readPacket();
        if (pkt.isNull()) {
            this->finished_ = true;
            logstream << "Got null packet";
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

        // check if we have some planned seek
        auto lock = std::lock_guard<decltype(seek_at_mutex_)>(seek_at_mutex_);
        if (!seek_at_table_.empty()) {
            auto e = seek_at_table_.front();
            if (pkt.pts() >= e.first) {
                seek_at_table_.pop_front();
                seek(e.second);
            }
        }
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

        if (params.count("preseek")) {
            preseek_ = params["preseek"];
        }

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

        for (size_t i=0; i<ictx_.streamsCount(); i++) {
            Parameters obj;
            av::Stream stream = ictx_.stream(i);
            obj["index"] = i;
            obj["type"] = mediaTypeToString(stream.mediaType());
            AVCodecParameters &cpar = *stream.raw()->codecpar;
            obj["codec"] = avcodec_get_name(cpar.codec_id);
            if (stream.isVideo()) {
                if (video_stream_<0) {
                    video_stream_ = i;
                }
                obj["fps"] = std::to_string(stream.frameRate().getNumerator()) + '/' + std::to_string(stream.frameRate().getDenominator());
                obj["width"] = cpar.width;
                obj["height"] = cpar.height;
                obj["pixel_format"] = av::PixelFormat((AVPixelFormat)cpar.format).name();
                obj["field_order"] = fieldOrderToString(cpar.field_order);
                obj["sar"] = std::to_string(cpar.sample_aspect_ratio.num) + '/' + std::to_string(cpar.sample_aspect_ratio.den);
            } else if (stream.isAudio()) {
                obj["sample_rate"] = cpar.sample_rate;
                char chlayout[64] = {0};
                av_get_channel_layout_string(chlayout, 63, cpar.channels, cpar.channel_layout);
                obj["channel_layout"] = chlayout;
                obj["channels_count"] = cpar.channels;
                obj["sample_format"] = av::SampleFormat((AVSampleFormat)cpar.format).name();
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

        if (params.count("seek_table") > 0) {
            seek_table_url_ = params["seek_table"];
            if (seek_table_url_.empty()) {
                seek_table_url_ = params["url"];
                seek_table_url_ += "+seek";
            }
        }
        if (params.count("ts_offsets") > 0) {
            ts_offsets_url_ = params["ts_offsets"];
            if (ts_offsets_url_.empty()) {
                ts_offsets_url_ = params["url"];
                ts_offsets_url_ += "+history";
            }
        }
        loadSeekTable();
        loadTimestampOffsets();
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
