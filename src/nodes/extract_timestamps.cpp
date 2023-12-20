#include "node_common.hpp"
#include "../instance_shared.hpp"

class TimestampExtractorTeam: public InstanceShared<TimestampExtractorTeam> {
protected:
    av::Timestamp shift_ = {0, {1,1}};
    bool has_shift_ = false;
    std::mutex busy_;
public:
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    av::Timestamp shiftTimestamp(av::Timestamp ts) {
        return addTS(ts, shift_);
    }
    bool ready() {
        return has_shift_;
    }
    void gotTimeCode(av::Timestamp extracted, av::Timestamp original) {
        // maybe TODO smoothing
        shift_ = addTS(extracted, negateTS(original));
        has_shift_ = true;
    }
};

template <typename Child, typename T> class ExtractTimestamps: public NodeSISO<T, T> {
protected:
    bool ready_ = false;
    std::shared_ptr<TimestampExtractorTeam> team_;
    bool passthrough_before_available_ = false;
    bool drop_before_available_ = false;
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        T* data_ptr = this->source_->peek();
        if (!data_ptr) return;
        T &data = *data_ptr;
        bool ready;
        {
            auto lock = team_->getLock();
            static_cast<Child*>(this)->extractTimestamp(data);
            ready = team_->ready() || passthrough_before_available_;
            if (ready) {
                data.setPts(team_->shiftTimestamp(data.pts()));
                // TODO handle DTS (in av::Packet)
            }
        }
        if (ready) {
            this->sink_->put(data);
            this->source_->pop();
        } else if (drop_before_available_) {
            this->source_->pop();
        } else {
            wallclock.sleepms(33); // TODO there are better ways of handling this, such as Event
        }
    }
    static std::shared_ptr<Child> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<T, T>::template createCommon<Child>(edges, params);
        if (params.count("team")) {
            r->team_ = InstanceSharedObjects<TimestampExtractorTeam>::get(nci.instance, params["team"]);
        } else {
            r->team_ = std::make_shared<TimestampExtractorTeam>(); // not really a team, but anyway
        }
        if (params.count("passthrough_before_available")) {
            r->passthrough_before_available_ = params["passthrough_before_available"];
        }
        if (params.count("drop_before_available")) {
            r->drop_before_available_ = params["drop_before_available"];
        }
        r->useParams(params);
        return r;
    }
};

class ExtractVideoTimestamps: public ExtractTimestamps<ExtractVideoTimestamps, av::VideoFrame> {
protected:
    enum class TimeCodeSource: int_fast8_t {
        NONE = 0,
        S12M_1 = 1,
        S12M_2 = 2,
        S12M_3 = 3,
        GOP = 4
    };
    struct TimeCodes {
        uint32_t s12m[4] = {0, 0, 0, 0};
        uint32_t gop = 0;
        bool have_gop = false;
        int s12mCount() {
            return std::min(s12m[0], 3u);
        }
        bool hasS12M(int index) {
            return index <= s12mCount();
        }
        bool has(TimeCodeSource type) {
            if (type==TimeCodeSource::GOP) return have_gop;
            return hasS12M((int)type);
        }
        av::Timestamp extract(TimeCodeSource type, AVRational rate, bool liveu) {
            if (type==TimeCodeSource::GOP) {
                // adapted from libavutil/timecode.c function av_timecode_make_mpeg_tc_string
                return hmsfToTs(gop>>19 & 0x1f,             // 5-bit hours
                               gop>>13 & 0x3f,              // 6-bit minutes
                               gop>>6  & 0x3f,              // 6-bit seconds
                               gop     & 0x3f,              // 6-bit frames
                               rate);
            } else {
                // adapted from libavutil/timecode.c function av_timecode_make_smpte_tc_string2
                uint32_t tcsmpte = s12m[(int)type];
                unsigned hh   = bcd2uint(tcsmpte     & 0x3f);    // 6-bit hours
                unsigned mm   = bcd2uint(tcsmpte>>8  & 0x7f);    // 7-bit minutes
                unsigned ss   = bcd2uint(tcsmpte>>16 & 0x7f);    // 7-bit seconds
                unsigned ff   = bcd2uint(tcsmpte>>24 & (liveu ? 0x7f : 0x3f));    // 6 or 7-bit frames
                //unsigned drop = tcsmpte & 1<<30 && !prevent_df;  // 1-bit drop if not arbitrary bit

                if ((!liveu) && (av_cmp_q(rate, (AVRational) {30, 1}) == 1)) {
                    ff <<= 1;
                    if (av_cmp_q(rate, (AVRational) {50, 1}) == 0)
                        ff += !!(tcsmpte & 1 << 7);
                    else
                        ff += !!(tcsmpte & 1 << 23);
                }
                return hmsfToTs(hh, mm, ss, ff, rate);
            }
        }
    private:
        static av::Timestamp hmsfToTs(AVTS hours, AVTS minutes, AVTS seconds, AVTS frames, AVRational fps) {
            return {(hours*3600 + minutes*60 + seconds)*fps.num/fps.den + frames, {fps.den, fps.num}};
        }
        static unsigned bcd2uint(uint8_t bcd) {
            unsigned low  = bcd & 0xf;
            unsigned high = bcd >> 4;
            if (low > 9 || high > 9)
                return 0;
            return low + 10*high;
        }
    };
    static constexpr size_t max_tc_sources_ = 4;
    TimeCodeSource tc_sources_[max_tc_sources_] = {TimeCodeSource::S12M_1, TimeCodeSource::NONE, TimeCodeSource::NONE, TimeCodeSource::NONE};
    AVRational fps_;
    bool liveu_ = false;
public:
    using ExtractTimestamps<ExtractVideoTimestamps, av::VideoFrame>::ExtractTimestamps;
    void extractTimestamp(av::VideoFrame &frame) {
        AVFrame* frm = frame.raw();
        if (!frm) return;
        TimeCodes timecodes;
        for (int i=0; i<frm->nb_side_data; i++) {
            AVFrameSideData *sd = frm->side_data[i];
            if (sd->type == AV_FRAME_DATA_S12M_TIMECODE && sd->size == 16) {
                uint32_t *tc_data = (uint32_t*)sd->data;
                for (int j=0; j<4; j++) {
                    timecodes.s12m[j] = tc_data[j];
                }
            } else if (sd->type == AV_FRAME_DATA_GOP_TIMECODE && sd->size >= 8) {
                timecodes.gop = *(int64_t*)(sd->data);
                timecodes.have_gop = true;
            }
        }

        for (int i=0; i<max_tc_sources_; i++) {
            TimeCodeSource tcs = tc_sources_[i];
            if (tcs==TimeCodeSource::NONE) break;
            if (timecodes.has(tcs)) {
                av::Timestamp tc_ts = timecodes.extract(tcs, fps_, liveu_);
                //logstream << "extracted timestamp from timecode: " << tc_ts;
                team_->gotTimeCode(tc_ts, frame.pts());
                break;
            }
        }
    }
    void useParams(const Parameters &params) {
        if (params.count("timecodes")==1) {
            for (int i=0; i<max_tc_sources_; i++) {
                tc_sources_[i] = TimeCodeSource::NONE;
            }
            auto tcstrs = jsonToStringList(params["timecodes"]);
            auto it = tcstrs.begin();
            int i = 0;
            while (it != tcstrs.end()) {
                std::string &tcstr = *it;
                TimeCodeSource tcs = TimeCodeSource::NONE;
                if (tcstr == "S12M.1" || tcstr == "S12M") {
                    tcs = TimeCodeSource::S12M_1;
                } else if (tcstr == "S12M.2") {
                    tcs = TimeCodeSource::S12M_2;
                } else if (tcstr == "S12M.3") {
                    tcs = TimeCodeSource::S12M_3;
                } else if (tcstr == "GOP") {
                    tcs = TimeCodeSource::GOP;
                }
                if (i>=max_tc_sources_) {
                    throw Error("Too many timecode sources (timecodes parameter), maximum allowed is " + std::to_string(max_tc_sources_));
                }
                tc_sources_[i] = tcs;
                i++;
                it++;
            }
        }

        std::string get_fr_from = "fps";
        if (params.count("frame_rate_source")) {
            get_fr_from = params["frame_rate_source"];
        }
        if (get_fr_from=="fps") {
            std::shared_ptr<IFrameRateSource> frsrc = this->template findNodeUp<IFrameRateSource>();
            if (frsrc) {
                fps_ = frsrc->frameRate();
            } else {
                throw Error("Unknown fps!");
            }
        } else if (get_fr_from=="timebase") {
            std::shared_ptr<ITimeBaseSource> tbsrc = this->template findNodeUp<ITimeBaseSource>();
            if (tbsrc) {
                AVRational tb = tbsrc->timeBase().getValue();
                fps_.num = tb.den;
                fps_.den = tb.num;
            } else {
                throw Error("Unknown timebase!");
            }
        } else {
            throw Error("invalid frame_rate_source");
        }
        if (params.count("liveu")) {
            liveu_ = params["liveu"];
        }
        if (fps_.num==0 || fps_.den==0) {
            throw Error("Invalid frame rate " + std::to_string(fps_.num) + '/' + std::to_string(fps_.den));
        }
    }
};

template <typename T> class ExtractTimestampsSlave: public ExtractTimestamps<ExtractTimestampsSlave<T>, T> {
public:
    using ExtractTimestamps<ExtractTimestampsSlave<T>, T>::ExtractTimestamps;
    void extractTimestamp(T&) {
        // NOOP
    }
    void useParams(const Parameters &) {
        // NOOP
    }
};

DECLNODE(extract_timestamps, ExtractVideoTimestamps);
DECLNODE_ATD(extract_timestamps_slave, ExtractTimestampsSlave);
