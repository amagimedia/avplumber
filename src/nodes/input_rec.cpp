#include "node_common.hpp"
#include <mutex>
#include <iostream>
#include <fstream>

#include "../InputSeekTeam.hpp"
#include "../PauseControlTeam.hpp"
#include "../SpeedControlTeam.hpp"

using namespace std::chrono_literals;

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

enum class ETimestampSource {
    ts_None,
    ts_Input,
    ts_Wallclock
};

class RecordingInput: public NodeSingleOutput<av::Packet>, public IStreamsInput, public ReportsFinishByFlag, public IPlaybackControl,
                   public IStoppable, public IInterruptible, public IReturnsObjects, public IInputsObjects, public ISeekAt, public ISpeed {
protected:
    av::FormatContext ictx_;
    std::atomic_bool should_end_ {false};
    bool input_url_set_ = false;
    AVTS wait_start_;
    AVTS wait_max_ = AV_NOPTS_VALUE;
    av::Timestamp node_stop_ts_ = NOTS;
    av::Timestamp stop_delay_ = {0, {1, 1}};
    av::Timestamp first_video_ts_;
    StreamTarget start_ts_ = StreamTarget::from_frames_absolute(0);
    StreamTarget stop_ts_ = StreamTarget::end();
    std::shared_ptr<PauseControlTeam> pause_team_;
    bool paused_read_ = false;
    std::shared_ptr<SpeedControlTeam> speed_team_;
    std::atomic_int speed_skip_frames_ = 0;
    std::atomic_int speed_skipped_frames_ = 0;

    std::string seek_table_url_;
    std::mutex seek_table_mutex_;
    std::vector<SeekTableEntry> seek_table_;
    std::mutex seek_at_mutex_;
    std::list<std::pair<av::Timestamp, StreamTarget>> seek_at_table_;
    std::shared_ptr<InputSeekTeam> team_;
    std::thread seek_read_thread_;
    Event seek_thread_terminate_;
    Event seek_thread_ready_;
    IPlaybackControl::EPlaybackDirection play_direction_ = IPlaybackControl::EPlaybackDirection::pd_Forward;
    int64_t last_stream_position_ = -1;
    int64_t live_delay_ = 1'000;
    ETimestampSource timestamp_source_ = ETimestampSource::ts_None;
    bool loop_ = false;

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

    void seekThreadFun() {
        bool first_run = true;
        do {
            loadSeekTable();
            loadTimestampOffsets();
            if (first_run) {
                seek_thread_ready_.signal();
                first_run = false;
            }
        } while (!seek_thread_terminate_.wait(1000));
    }

private:
    void resolveSeekTarget(StreamTarget& st) {
        if (st.isStop()) {
            return;
        }

        if (st.isEmpty()) {
            // flush only
            return;
        }

        auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

        if (seek_table_.empty()) {
            // no seek table available, only seeks by time
            return;
        }

        if (st.isLive()) {
            uint64_t t = seek_table_.crbegin()->timestamp_ms;
            if (t < live_delay_) {
                t = seek_table_.begin()->timestamp_ms;
            } else {
                t -= live_delay_;
            }
            st.ts = av::Timestamp(t, {1, 1000});
            st.type = StreamTarget::ETargetType::tt_Timestamp;
        }

        if (st.isEnd()) {
            uint64_t t = seek_table_.crbegin()->timestamp_ms;
            st.ts = av::Timestamp(t, {1, 1000});
            st.type = StreamTarget::ETargetType::tt_Timestamp;
        }

        if (st.isFrameAbsolute()) {
            int64_t frame = st.frame_number;
            if (frame < 0)
                frame = 0;
            if (frame > seek_table_.size())
                frame = seek_table_.size() - 1;

            st.ts = NOTS;
            st.bytes = seek_table_[frame].bytes;
            st.type = StreamTarget::ETargetType::tt_Bytes;
            return;
        }

        int64_t frame_ms = -1;

        if (st.isTimestamp() || st.isWallclock()) {
            switch (timestamp_source_) {
                case ETimestampSource::ts_Wallclock:
                    {
                        auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);

                        if (!ts_offsets_.empty()) {
                            int64_t new_ts = rescaleTS(st.ts, {1, 1000}).timestamp();
                            auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), new_ts, [](const TSOffsetEntry& e, int64_t value) {
                                return e.changed_at - e.wallclock_diff < value;
                            });
                            if (it == ts_offsets_.cend()) {
                                it = std::prev(it);
                            }
                            if ((it != ts_offsets_.cbegin()) && (it->changed_at - it->wallclock_diff > new_ts)) {
                                it = std::prev(it);
                            }

                            frame_ms = new_ts + it->wallclock_diff;
                        }
                    }
                    break;
                case ETimestampSource::ts_Input:
                    {
                        auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);

                        if (!ts_offsets_.empty()) {
                            int64_t new_ts = rescaleTS(st.ts, {1, 1000}).timestamp();
                            auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), new_ts, [](const TSOffsetEntry& e, int64_t value) {
                                return e.changed_at - e.input_ts_diff < value;
                            });
                            if (it == ts_offsets_.cend()) {
                                it = std::prev(it);
                            }
                            if ((it != ts_offsets_.cbegin()) && (it->changed_at - it->input_ts_diff > new_ts)) {
                                it = std::prev(it);
                            }

                            frame_ms = new_ts + it->input_ts_diff;
                        }
                    }
                    break;
                default:
                    if (st.isTimestamp()) {
                        frame_ms = rescaleTS(st.ts, {1, 1000}).timestamp();
                    } else if (st.isWallclock()) {
                        // invalid request, jump to the beginning of file
                        frame_ms = 0;
                    }
                    break;
            }
        }

        SeekTableEntry ste;

        auto it = std::lower_bound(seek_table_.cbegin(), seek_table_.cend(), frame_ms, [](const SeekTableEntry& e, int64_t value) {
            return e.timestamp_ms < value;
        });
        if (it == seek_table_.cend()) {
            it = std::prev(it);
        }
        int64_t diff = abs(it->timestamp_ms - frame_ms);
        for (int i = 0; i < 5; ++i) {
            if (it != seek_table_.begin()) {
                int64_t diff2 = abs(std::prev(it)->timestamp_ms - frame_ms);
                if (diff2 < diff) {
                    diff = diff2;
                    --it;
                } else {
                    break;
                }
            }
        }

        st.ts = NOTS;
        st.bytes = it->bytes;
        st.type = StreamTarget::ETargetType::tt_Bytes;
    }

    virtual void fixInputTimestamp(StreamTarget& st) override
    {
        switch (st.type) {
            case StreamTarget::ETargetType::tt_Wallclock:
                {
                    switch (timestamp_source_) {
                        case ETimestampSource::ts_Input:
                        case ETimestampSource::ts_Wallclock:
                            // just do nothing, timestamp is ok
                            break;
                        default:
                            // no timestamp source set
                            // fix target to the very beginning of file
                            st.ts = av::Timestamp(0, {1, 1});
                            break;
                    }
                }
                break;
            case StreamTarget::ETargetType::tt_Timestamp:
                {
                    auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);

                    if (!ts_offsets_.empty()) {
                        int64_t new_ts = rescaleTS(st.ts, {1, 1000}).timestamp();
                        auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), new_ts, [](const TSOffsetEntry& e, int64_t value) {
                            return e.changed_at < value;
                        });
                        if (it == ts_offsets_.cend()) {
                            it = std::prev(it);
                        }
                        if (it->changed_at > new_ts) {
                            it = std::prev(it);
                        }
                        switch (timestamp_source_) {
                            case ETimestampSource::ts_Input:
                                new_ts -= it->input_ts_diff;
                                break;
                            case ETimestampSource::ts_Wallclock:
                                new_ts -= it->wallclock_diff;
                                break;
                        }

                        st.ts = av::Timestamp(new_ts, {1, 1000});
                    }
                }
                break;
            case StreamTarget::ETargetType::tt_Live:
                // do nothing;
                return;
            case StreamTarget::ETargetType::tt_End:
                // do nothing;
                return;
            case StreamTarget::ETargetType::tt_Stop:
                // do nothing;
                return;
            case StreamTarget::ETargetType::tt_Bytes:
                // do nothing;
                return;
        }
    }

    void setFrameTimestamps(av::VideoFrame& frm, int64_t frame_index, const av::Timestamp& ts_v, const av::Timestamp& ts_in, const av::Timestamp& ts_out, const av::Timestamp& ts_wallclock) {
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

        set_ts("video_ts", ts_v);
        set_ts("video_pts", frm.pts());
        set_ts("input_ts", ts_in);
        set_ts("output_ts", ts_out);
        set_ts("wallclock_ts", ts_wallclock);
        av_dict_set(&frame->metadata, "wallclock", std::to_string(ts_wallclock.timestamp({1, 1000})).c_str(), 0);
        if (frame_index >= 0) {
            av_dict_set(&frame->metadata, "frame_no", std::to_string(frame_index).c_str(), 0);
            auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);
            if (frame_index < seek_table_.size()) {
                av_dict_set(&frame->metadata, "frame_ts", std::to_string(seek_table_[frame_index].timestamp_ms).c_str(), 0);
            }
        }
    }
public:
    RecordingInput(std::unique_ptr<Sink<av::Packet>> &&sink): NodeSingleOutput<av::Packet>(std::move(sink)) {
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
        av::Timestamp video_ts;
        av::Timestamp input_ts;
        av::Timestamp output_ts;
        av::Timestamp wallclock_ts;

        {
            // get timestamps offset
            auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);
            if (!ts_offsets_.empty()) {
                switch (timestamp_source_) {
                    case ETimestampSource::ts_Input:
                        {
                            input_ts = frame.pts();
                            uint64_t v = rescaleTS(input_ts, {1, 1000}).timestamp();
                            auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), v, [](const TSOffsetEntry& e, int64_t value) {
                                return e.changed_at - e.input_ts_diff < value;
                            });
                            if (it == ts_offsets_.cend()) {
                                it = std::prev(it);
                            }
                            if ((it != ts_offsets_.cbegin()) && ((it->changed_at - it->input_ts_diff) > v)) {
                                it = std::prev(it);
                            }
                            video_ts = addTS(input_ts, av::Timestamp(it->input_ts_diff, {1, 1000}));
                            wallclock_ts = addTS(video_ts, av::Timestamp(-it->wallclock_diff, {1, 1000}));
                            output_ts = addTS(video_ts, av::Timestamp(-it->output_ts_diff, {1, 1000}));
                        }
                        break;
                    case ETimestampSource::ts_Wallclock:
                        {
                            wallclock_ts = frame.pts();
                            uint64_t v = rescaleTS(wallclock_ts, {1, 1000}).timestamp();
                            auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), v, [](const TSOffsetEntry& e, int64_t value) {
                                return e.changed_at - e.wallclock_diff < value;
                            });
                            if (it == ts_offsets_.cend()) {
                                it = std::prev(it);
                            }
                            if ((it != ts_offsets_.cbegin()) && ((it->changed_at - it->wallclock_diff) > v)) {
                                it = std::prev(it);
                            }
                            video_ts = addTS(wallclock_ts, av::Timestamp(it->wallclock_diff, {1, 1000}));
                            input_ts = addTS(video_ts, av::Timestamp(-it->input_ts_diff, {1, 1000}));
                            output_ts = addTS(video_ts, av::Timestamp(-it->output_ts_diff, {1, 1000}));
                        }
                        break;
                    default:
                        {
                            video_ts = frame.pts();
                            uint64_t v = rescaleTS(wallclock_ts, {1, 1000}).timestamp();
                            auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), v, [](const TSOffsetEntry& e, int64_t value) {
                                return e.changed_at < value;
                            });
                            if (it == ts_offsets_.cend()) {
                                it = std::prev(it);
                            }
                            if ((it != ts_offsets_.cbegin()) && (it->changed_at > v)) {
                                it = std::prev(it);
                            }
                            output_ts = addTS(video_ts, av::Timestamp(-it->output_ts_diff, {1, 1000}));
                            input_ts = addTS(video_ts, av::Timestamp(-it->input_ts_diff, {1, 1000}));
                            wallclock_ts = addTS(video_ts, av::Timestamp(-it->wallclock_diff, {1, 1000}));
                        }
                        break;
                }
            }
        }

        {
            // get frame index
            int64_t frame_index = -1;
            {
                auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

                if (!seek_table_.empty()) {
                    int64_t req_ts = rescaleTS(video_ts, {1, 1000}).timestamp();
                    auto it = std::lower_bound(seek_table_.cbegin(), seek_table_.cend(), req_ts, [](const SeekTableEntry& e, int64_t value) {
                        return e.timestamp_ms < value;
                    });
                    if (it == seek_table_.cend()) {
                        it = std::prev(it);
                    }
                    if ((it != seek_table_.cbegin()) && (it->timestamp_ms > req_ts)) {
                        it = std::prev(it);
                    }

                    frame_index = static_cast<int64_t>(std::distance(seek_table_.cbegin(), it));
                }
            }

            setFrameTimestamps(frame, frame_index, video_ts, input_ts, output_ts, wallclock_ts);
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
        StreamTarget when_fixed = when;
        StreamTarget target_fixed = target;
        fixInputTimestamp(when_fixed);
        fixInputTimestamp(target_fixed);
        auto lock = std::lock_guard<decltype(seek_at_mutex_)>(seek_at_mutex_);
        seek_at_table_.push_back(std::make_pair(when_fixed.ts, target_fixed));
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
            size_t start;
            {
                auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);
                start = sizeof(SeekTableEntry) * seek_table_.size();
            }
            f.seekg(0, std::ios::end);
            if (f.tellg() == std::streampos(-1))
                return;
            size_t count = static_cast<size_t>(f.tellg()) - start;
            if (count < sizeof(SeekTableEntry)) {
                // no new data
                return;
            }
            f.seekg(start);
            std::vector<char> buffer(count);
            f.read(buffer.data(), count);
            count /= sizeof(SeekTableEntry);
            {
                auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);
                seek_table_.reserve(seek_table_.size() + count);
                for (int idx = 0; idx < count; ++idx) {
                    SeekTableEntry* entry = (SeekTableEntry*)(buffer.data()) + idx;
                    seek_table_.push_back(*entry);
                }
                logstream << "seek table loaded, current video length: " << seek_table_.crbegin()->timestamp_ms << "ms";
            }
        }
    }
    void loadTimestampOffsets() {
        if (ts_offsets_url_.empty())
            return;
        std::ifstream f(ts_offsets_url_, std::ios::binary);
        if (f) {
            size_t start;
            {
                auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);
                start = sizeof(TSOffsetEntry) * ts_offsets_.size();
            }
            f.seekg(0, std::ios::end);
            size_t count = static_cast<size_t>(f.tellg()) - start;
            if (count < sizeof(SeekTableEntry)) {
                // no new data
                return;
            }
            f.seekg(start);
            std::vector<char> buffer(count);
            f.read(buffer.data(), count);
            count /= sizeof(TSOffsetEntry);
            {
                auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);
                ts_offsets_.reserve(ts_offsets_.size() + count);
                for (int idx = 0; idx < count; ++idx) {
                    TSOffsetEntry* entry = (TSOffsetEntry*)(buffer.data()) + idx;
                    ts_offsets_.push_back(*entry);
                }
            }
        }
    }
    void doStop() {
        logstream << "video_stream_ " << video_stream_ << " stopping in " << stop_delay_;
        node_stop_ts_ = addTS(wallclock.absolute_ts(), stop_delay_);
    }
    virtual void process() {
        bool seeked = false;

        if (!input_url_set_)
            return;

        if (node_stop_ts_.isValid()) {
            if (node_stop_ts_ <= wallclock.absolute_ts()) {
                stop();
            }
            return;
        }

        {
            auto lock = std::lock_guard<decltype(seek_mutex_)>(seek_mutex_);
            if (need_seek_) {
                //ictx_.flush();
                //avformat_flush(ictx_.raw());
                if (seek_target_.isTimestamp()) {
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
                } else if (seek_target_.isBytes()) {
                    logstream << "video_stream_ " << video_stream_ << " seek to position: " << seek_target_.bytes;
                    last_stream_position_ = seek_target_.bytes;
                    ictx_.seek(seek_target_.bytes, -1, AVSEEK_FLAG_BYTE);
                } else if (seek_target_.isStop()) {
                    doStop();
                }
                need_seek_ = false;
                seeked = true;
            } else {
                if (play_direction_ == IPlaybackControl::EPlaybackDirection::pd_Backward) {
                    auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);
                    if (!seek_table_.empty()) {
                        auto it = std::lower_bound(seek_table_.cbegin(), seek_table_.cend(), last_stream_position_, [](const SeekTableEntry& e, int64_t value) {
                            return e.bytes < value;
                        });
                        for (int i = 0; i <= (paused_read_ ? 0 : speed_skip_frames_.load()); ++i) {
                            if (it != seek_table_.cbegin()) {
                                it = std::prev(it);
                            }
                        }
                        last_stream_position_ = it->bytes;
                        ictx_.seek(it->bytes, -1, AVSEEK_FLAG_BYTE);
                    }
                }
            }
        }
        if (seeked && !auto_resume_after_seek_) {
            seek_resume_.wait();
        }
        if (!paused_read_) {
            if (pause_team_ && pause_team_->isPaused()) {
                paused_read_ = seeked;
                if (!paused_read_) {
                    // graph is paused, wait for another seek
                    std::this_thread::sleep_for(0us);
                    return;
                }
            }
        }
        wait_start_ = wallclock.pts();
        av::Packet pkt = ictx_.readPacket();

        if (play_direction_ == IPlaybackControl::EPlaybackDirection::pd_Backward) {
            // read only video frames, discard all other frames
            // read packet until video packet found
            int rep = 50;
            while (!pkt.isNull() && rep--) {
                if (pkt.streamIndex() == video_stream_)
                    break;
                pkt = ictx_.readPacket();
            }
        }

        if (pkt.isNull() && loop_ && (play_direction_ == EPlaybackDirection::pd_Forward)) {
            seek(start_ts_);
            return;
        }
        if (pkt.isNull()) {
            // we are at the end os recording
            //logstream << "end of video reached";
            std::this_thread::sleep_for(5ms);
            return;
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

        if (!pkt.isNull()) {
            if (paused_read_ && (pkt.streamIndex() == video_stream_) && pkt.isKeyPacket()) {
                // video frame read, do not read more packets
                paused_read_ = false;
            }

            if (!paused_read_ && (speed_skip_frames_ > 0) && (play_direction_ == EPlaybackDirection::pd_Forward) && (pkt.streamIndex() == video_stream_)) {
                if (pkt.isKeyPacket()) {
                    if (speed_skipped_frames_++ < speed_skip_frames_) {
                        // skip this video frame
                        return;
                    }
                    speed_skipped_frames_ = 0;
                } else {
                    // this doesn't work with non-intra-only frames stream, disable frame skipping
                    speed_skip_frames_ = 0;
                }
            }

            if (first_video_ts_.timestamp() != 0) {
                pkt.setDts(addTS(pkt.dts(), negateTS(first_video_ts_)));
                pkt.setPts(addTS(pkt.pts(), negateTS(first_video_ts_)));
            }

            // adjust ts
            auto lock = std::lock_guard<decltype(ts_offsets_mutex_)>(ts_offsets_mutex_);
            if (!ts_offsets_.empty() && (timestamp_source_ != ETimestampSource::ts_None)) {
                int64_t pts = rescaleTS(pkt.pts(), {1, 1000}).timestamp();
                auto it = std::lower_bound(ts_offsets_.cbegin(), ts_offsets_.cend(), pts, [](const TSOffsetEntry& e, int64_t value) {
                    return e.changed_at < value;
                });
                if (it == ts_offsets_.cend())
                    it--;
                if (it->changed_at > pts)
                    it--;

                int64_t pts_diff = 0;
                switch (timestamp_source_) {
                    case ETimestampSource::ts_Input:
                        pts_diff -= it->input_ts_diff;
                        break;
                    case ETimestampSource::ts_Wallclock:
                        pts_diff -= it->wallclock_diff;
                        break;
                }
                if (pts_diff != 0) {
                    pkt.setDts(addTS(pkt.dts(), av::Timestamp(pts_diff, {1, 1000})));
                    pkt.setPts(addTS(pkt.pts(), av::Timestamp(pts_diff, {1, 1000})));
                }
            }
        }
        this->sink_->put(pkt);

        // check if we have to stop/loop video
        if (play_direction_ == EPlaybackDirection::pd_Forward) {
            if (stop_ts_.ts.isValid() && (pkt.pts() >= stop_ts_.ts)) {
                if (loop_) {
                    seek(start_ts_);
                } else {
                    this->sink_->put(createEofPacket(video_stream_));
                    doStop();
                }
            }
        } else {
            if (start_ts_.ts.isValid() && (pkt.pts() <= start_ts_.ts)) {
                if (loop_) {
                    seek(stop_ts_);
                } else {
                    this->sink_->put(createEofPacket(video_stream_));
                    doStop();
                }
            }
        }

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
    void speedChanged() override {
        speed_skip_frames_ = std::max(int(fabs(speed_team_->getSpeed()) + 0.5) - 1, 0);
        speed_skipped_frames_ = 0;
    }
    virtual ~RecordingInput() {
        if (seek_read_thread_.joinable()) {
            seek_thread_terminate_.signal();
            seek_read_thread_.join();
        }
        #if 0 // see comment in process() "Got null packet"
        if (ictx_.isOpened()) {
            logstream << "BUG: input context still opened in destructor, closing";
            closeInput(true);
        }
        #else
        closeInput(true);
        #endif
    }
    void setPlaybackDirection(IPlaybackControl::EPlaybackDirection dir) override {
        if (play_direction_ != dir) {
            play_direction_ = dir;
            last_stream_position_ = avio_tell(ictx_.raw()->pb);
        }
    }
    IPlaybackControl::EPlaybackDirection getPlaybackDirection() override {
        return play_direction_;
    }
    size_t getFrameNumber(size_t start_frame, const av::Timestamp& offset) override {
        auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

        if (seek_table_.empty() || (start_frame >= seek_table_.size())) {
            throw Error("no seek table available");
        }
        int64_t ms = seek_table_.at(start_frame).timestamp_ms + rescaleTS(offset, {1, 1000}).timestamp();
        auto it = std::lower_bound(seek_table_.cbegin(), seek_table_.cend(), ms, [](const SeekTableEntry& e, size_t value) {
            return e.timestamp_ms < value;
        });

        if (it == seek_table_.end()) {
            throw Error("timestamp outside input duration");
        }

        // find closest frame
        if (it->timestamp_ms > ms) {
            int64_t diff = it->timestamp_ms - ms;
            for (int i = 0; i < 5; ++i) {
                if (it != seek_table_.begin()) {
                    int64_t diff2 = abs(std::prev(it)->timestamp_ms - ms);
                    if (diff2 < diff) {
                        diff = diff2;
                        --it;
                    } else {
                        break;
                    }
                }
            }
        } else {
            if (it->timestamp_ms < ms) {
                int64_t diff = ms - it->timestamp_ms;
                for (int i = 0; i < 5; ++i) {
                    if (it != seek_table_.begin()) {
                        int64_t diff2 = abs(std::next(it)->timestamp_ms - ms);
                        if (diff2 < diff) {
                            diff = diff2;
                            ++it;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        return it - seek_table_.begin();
    }

    void offsetStreamTargetByFrames(StreamTarget& ts, const int64_t frames) override {
        auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

        if (seek_table_.size() < abs(frames)) {
            throw Error("no seek table available");
        }

        int64_t duration = std::copysign(seek_table_.at(abs(frames)).timestamp_ms - seek_table_.cbegin()->timestamp_ms, frames);

        switch (ts.type) {
            case StreamTarget::ETargetType::tt_Wallclock:
                ts.ts = addTS(ts.ts, av::Timestamp(duration, {1, 1000}));
                break;
            default:
                throw Error("not implemented StreamTarget type");
        }
    }

    static std::shared_ptr<RecordingInput> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        auto r = std::make_shared<RecordingInput>(make_unique<EdgeSink<av::Packet>>(edge));
        if (params.count("team")) {
            r->team_ = InstanceSharedObjects<InputSeekTeam>::get(nci.instance, params["team"]);
            r->team_->addSeekTarget(r);
        }
        if (params.count("pause_team")) {
            r->pause_team_ = InstanceSharedObjects<PauseControlTeam>::get(nci.instance, params["pause_team"]);
        }
        if (params.count("speed_team")) {
            r->speed_team_ = InstanceSharedObjects<SpeedControlTeam>::get(nci.instance, params["speed_team"]);
            r->speed_team_->addNode(std::dynamic_pointer_cast<ISpeed>(r->shared_from_this()));
        }
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

        std::shared_ptr<Edge<av::Packet>> edge = edges.find<av::Packet>(params["dst"]);
        edge->setProducer(this->shared_from_this());

        input_url_set_ = !params["url"].get<std::string>().empty();
        if (!input_url_set_) {
            logstream << "Input node started without input url";
            return;
        }

        ictx_.openInput(params["url"], opts, ifmt);
        ictx_.findStreamInfo();
        logstream << "Opened URL " << params["url"] << " . Streams:";
        for (unsigned i=0; i<ictx_.streamsCount(); i++) {
            av::Stream stream = ictx_.stream(i);
            logstream << i << ": " << ( stream.isVideo() ? "video" : (stream.isAudio() ? "audio" : "???") ) << " tb " << stream.timeBase();
        }
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
                    first_video_ts_ = stream.startTime();
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
#if API_NEW_CHANNEL_LAYOUT
                av_channel_layout_describe(&cpar.ch_layout, chlayout, 63);
                obj["channels_count"] = cpar.ch_layout.nb_channels;
#else
                av_get_channel_layout_string(chlayout, 63, cpar.channels, cpar.channel_layout);
                obj["channels_count"] = cpar.channels;
#endif
                obj["channel_layout"] = chlayout;
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
        if (!seek_table_url_.empty() || !ts_offsets_url_.empty()) {
            seek_read_thread_ = start_thread("seek table read", [this]() {
                this->seekThreadFun();
            });
            seek_thread_ready_.wait();
        }
        if (params.count("live_delay") > 0) {
            live_delay_ = int64_t(1000 * params["live_delay"].get<double>());
        }
        if (params.count("loop") > 0) {
            loop_ = params["loop"];
        }
        if (params.count("timestamp_source") > 0) {
            std::string ts_source = params["timestamp_source"];
            if (ts_source == "input") {
                timestamp_source_ = ETimestampSource::ts_Input;
            } else if (ts_source == "wallclock") {
                timestamp_source_ = ETimestampSource::ts_Wallclock;
            } else {
                timestamp_source_ = ETimestampSource::ts_None;
            }
        }
        if (params.count("start_ts") > 0) {
            std::string s = params["start_ts"];
            start_ts_ = StreamTarget::from_string(s);
            fixInputTimestamp(start_ts_);
            seek(start_ts_);
        }
        if (params.count("stop_ts") > 0) {
            std::string stop = params["stop_ts"];
            stop_ts_ = StreamTarget::from_string(stop);
            fixInputTimestamp(stop_ts_);
        }
        if (params.count("stop_delay") > 0) {
            stop_delay_ = av::Timestamp(params["stop_delay"].get<int64_t>(), {1, 1000});
        }
    }
    virtual Parameters getObject(const std::string name) {
        if (name=="streams") {
            return streams_object_;
        } else if (name=="programs") {
            return programs_object_;
        } else if (name == "stream-limits" ) {
            Parameters res;
            auto lock = std::lock_guard<decltype(seek_table_mutex_)>(seek_table_mutex_);

            int64_t rec_length;

            if (seek_table_.empty()) {
                auto duration = rescaleTS(ictx_.duration(), {1, 1000});
                rec_length = duration.timestamp();
            } else {
                if (seek_table_.empty()) {
                    auto duration = rescaleTS(ictx_.duration(), {1, 1000});
                    rec_length = duration.timestamp();
                } else {
                    rec_length = seek_table_.crbegin()->timestamp_ms;
                }
            }
            if (start_ts_.isValidTimestamp()) {
                int64_t st = start_ts_.ts.timestamp();
                res["start"] = st;
            }
            if (stop_ts_.isValidTimestamp()) {
                int64_t st = stop_ts_.ts.timestamp();
                res["stop"] = st;
            }
            StreamTarget t = StreamTarget::from_timestamp({0, {1, 1}});
            fixInputTimestamp(t);
            res["video_start"] = t.ts.timestamp();
            res["duration"] = rec_length;
            res["loop"] = loop_;
            return res;
        } else {
            throw Error("Unknown object to get");
        }
    }
    virtual void setObject(const std::string name, const Parameters& p) {
        if (name == "stream-limits") {
            if (p.count("start") > 0) {
                if (p["start"].is_null()) {
                    start_ts_ = StreamTarget();
                } else {
                    std::string s = p["start"];
                    StreamTarget ts = StreamTarget::from_string(s);
                    fixInputTimestamp(ts);
                    start_ts_ = ts;
                }
            }
            if (p.count("stop") > 0) {
                if (p["stop"].is_null()) {
                    stop_ts_ = StreamTarget::end();
                } else {
                    std::string s = p["stop"];
                    StreamTarget ts = StreamTarget::from_string(s);
                    fixInputTimestamp(ts);
                    stop_ts_ = ts;
                }
            }
            if (p.count("loop") > 0) {
                loop_ = p["loop"].get<bool>();
            }
        }
    }
};

DECLNODE(input_rec, RecordingInput);
