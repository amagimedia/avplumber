#include "node_common.hpp"
#include "../instance_shared.hpp"
#include "../picture_buffer.hpp"
#include "../rest_client.hpp"

#include <avcpp/codeccontext.h>
#include <avcpp/videorescaler.h>
#include <ios>
#include <memory>

#pragma pack(push)
#pragma pack(1)
struct HistoryTableEntry {
    int64_t changed_at;
    int64_t input_pts_offset;
    int64_t wallclock_offset;
    int64_t output_pts_offset;
};
#pragma pack(pop)

class PTSCorrectorCommon: public InstanceShared<PTSCorrectorCommon> {
protected:
    std::recursive_mutex busy_;
    av::Timestamp timeshift_ = NOTS;
    av::Rational timebase_ = {0, 1};
    bool lock_timeshift_ = false;
    av::Timestamp clk_ = NOTS;
    AVTS clk_wallclock_ = AV_NOPTS_VALUE;
    av::Timestamp last_discontinuity_ = NOTS;
    av::Timestamp wallclock_offset_ = NOTS;
    std::ofstream timeshift_history_file_text_;
    std::ofstream timeshift_history_file_;
    ThreadedRESTEndpoint rest_;
    bool reporting_ = false;
public:
    av::Timestamp start_ts_ = {10, {1,1}};
    void wallclockOffsetChanged(av::Timestamp offset) {
        wallclock_offset_ = offset;
    }
    void reportTimeshiftChange() {
        long changed_at = rescaleTS(addTS(rtcTS(), negateTS(start_ts_)), {1, 1000}).timestamp();
        
        long input_pts_offset = rescaleTS(addTS(timeshift_, negateTS(start_ts_)), {1, 1000}).timestamp();
        
        long wallclock_offset = 0;
        if (wallclock_offset_.isValid()) {
            wallclock_offset = rescaleTS(addTS(wallclock_offset_, negateTS(start_ts_)), {1, 1000}).timestamp();
        }
        long output_pts_offset = rescaleTS(start_ts_, {1, 1000}).timestamp();
        
        if (timeshift_history_file_text_.is_open()) {
            timeshift_history_file_text_ << changed_at << " " << input_pts_offset << " " << wallclock_offset << " " << output_pts_offset << "\n";
            timeshift_history_file_text_.flush();
        }
        if (timeshift_history_file_.is_open()) {
            HistoryTableEntry entry { changed_at, input_pts_offset, wallclock_offset, output_pts_offset };
            timeshift_history_file_.write(reinterpret_cast<char*>(&entry), sizeof(entry));
            timeshift_history_file_.flush();
        }
        if (reporting_) {
            Parameters jobj;
            jobj["changed_at"] = changed_at;
            jobj["input_pts_offset"] = input_pts_offset;
            jobj["wallclock_offset"] = wallclock_offset;
            jobj["output_pts_offset"] = output_pts_offset;
            std::string json_str = jobj.dump();
            rest_.send("", json_str);
        }
    }
    void openHistoryFileText(const std::string path) {
        timeshift_history_file_text_.open(path, std::ios_base::app);
    }
    void openHistoryFile(const std::string path) {
        timeshift_history_file_.open(path, std::ios_base::app);
    }
    void addTimebase(av::Rational tb) {
        if ( (timebase_.getNumerator()==0) || (timebase_<tb) ) {
            timebase_ = tb;
        }
    }
    bool hasTimeshift() {
        return timeshift_.isValid();
    }
    void firstTS(const av::Timestamp ts) {
        // argument = TS encountered by stream's PTS corrector
        logstream << "First TS: " << ts << "; global corrector's TB: " << timebase_ << std::endl;
        if (!ts.isValid()) {
            throw Error("NOPTS supplied as first PTS");
        }
        timeshift_ = {rtcTS(false).timestamp(timebase_) - ts.timestamp(timebase_), timebase_};
        //reportTimeshiftChange();
    }
    void setTS(const av::Timestamp ts) {
        if ( (!clk_.isValid()) || (ts > clk_) ) {
            clk_ = rescaleTS(ts, timebase_);
            clk_wallclock_ = wallclock.pts();
        }
    }
    bool hasTS() {
        return (clk_wallclock_ != AV_NOPTS_VALUE) && clk_.isValid();
    }
    av::Timestamp rtcTS(const bool warn_if_empty = true) {
        if (!hasTS()) {
            if (warn_if_empty) {
                logstream << "Warning: Global corrector doesn't have clock source. Bootstrapping with start_ts_.";
            }
            setTS(start_ts_);
        }
        //double diff_sec = (wallclock.ts() - clk_wallclock_).seconds();
        AVTS clkdiff = wallclock.pts() - clk_wallclock_;
        double diff_sec = clkdiff * av::Rational(wallclock.timeBase()).getDouble();
        //logstream << "RTC ahead of clk by " << diff_sec << "s";
        if (diff_sec < 2.0) {
            return clk_;
        } else {
            //return clk_ + (wallclock.ts() - clk_wallclock_);
            //logstream << "Returning rtcTS: " << clk_.timestamp() << " + " << (int)(diff_sec / clk_.timebase().getDouble()) << " tb=" << clk_.timebase();
            return { clk_.timestamp() + (AVTS)(diff_sec / clk_.timebase().getDouble()), clk_.timebase() };
        }
    }
    av::Timestamp startTS() {
        return start_ts_;
    }
    av::Timestamp startTS(const av::Rational tb) {
        return rescaleTS(start_ts_, tb);
        //return {start_ts_.timestamp(tb), tb};
    }
    av::Timestamp timeshift() {
        if (!timeshift_.isValid()) {
            throw Error("Refusing to return NOPTS in PTSCorrectorCommon::timeshift()");
        }
        return timeshift_;
    }

    void setTimeshift(const av::Timestamp ts, bool report) {
        if (lock_timeshift_ && timeshift_.isValid()) {
            logstream << "ignoring setTimeshift - correction group locked " << timeshift_ << " -> " << ts;
            return;
        }
        logstream << "setTimeshift " << timeshift_ << " -> " << ts;
        timeshift_ = rescaleTS(ts, timebase_);
        if (report) {
            reportTimeshiftChange();
        }
    }
    void lockTimeshift() {
        lock_timeshift_ = true;
    }
    av::Timestamp timeshift(const av::Rational tb) {
        return rescaleTS(timeshift_, tb);
    }
    double timeshiftDiff(const av::Timestamp cmpto) {
        //return fabs( ((double)(timeshift_.timestamp(timebase_)-cmpto.timestamp(timebase_))) * timebase_.getDouble() );
        return fabs(addTS(timeshift_, negateTS(cmpto)).seconds());
    }
    void nowDiscontinuity() {
        last_discontinuity_ = rtcTS();
    }
    bool wasDiscontinuityRecently() {
        return addTS(rtcTS(), negateTS(last_discontinuity_)).seconds() < 2.0;
    }
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    decltype(busy_)& mutex() {
        return busy_;
    }
    void setReportingURL(const std::string url) {
        rest_.setBaseURL(url);
        reporting_ = !url.empty();
        if (reporting_) {
            rest_.setMinimumInterval(1);
        }
    }
};

template <typename T> struct CorrMediaSpecific {
};

template<> struct CorrMediaSpecific<av::AudioSamples> {
protected:
    uint8_t* zero_data_ = nullptr;
    static constexpr size_t max_len_ = 1024;
    size_t prev_size_ = 0;
    int64_t channel_layout_ = -1;
    size_t channel_count_ = 0;
    int sample_rate_ = -1;
    static constexpr av::SampleFormat::Alignment align_ = av::SampleFormat::Alignment::AlignDefault;
    av::SampleFormat sample_format_ {AV_SAMPLE_FMT_NONE};
public:
    void gotFrame(av::AudioSamples &frm) {
        if ( (frm.channelsLayout()>=0) && (frm.sampleRate()>0) && (frm.sampleFormat().get()!=AV_SAMPLE_FMT_NONE) ) {
            setAudioParameters(frm.channelsLayout(), frm.sampleRate(), frm.sampleFormat());
        }
    }
    void normalizeFrame(av::AudioSamples &frm) {
        int tbsr = frm.timeBase().getDenominator()/frm.timeBase().getNumerator();
        if ( tbsr != frm.sampleRate() ) {
            //logstream << "Warning: Strange audio frame: " << tbsr << "Hz in timebase, " << frm.sampleRate() << "Hz in audio data.";
            // this problem is common with RTMP, not spamming console with it.
            frm.setTimeBase({ 1, frm.sampleRate() });
        }
    }
    static av::Timestamp getDelta(av::AudioSamples &frm) {
        return av::Timestamp(frm.samplesCount(), frm.timeBase());
    }
    static constexpr bool is_video = false;
    av::AudioSamples getBackup(const av::Timestamp req_len) {
        av::Timestamp len = req_len;
        if (sample_format_.get()==AV_SAMPLE_FMT_NONE) {
            throw Error("CorrMediaSpecific<AudioSamples> not initialized!");
        }
        if (!len.isValid()) {
            throw Error("Cannot create silent audio frame of unknown length!");
        }

        int sr = (int)(len.timebase().getDenominator()/len.timebase().getNumerator());
        if (sr != sample_rate_) {
            logstream << "Warning: Sample rate changed in backup audio generator: " << sample_rate_ << " -> " << sr;
            sample_rate_ = sr;
        }

        if (len.timestamp() > max_len_) len = {max_len_, len.timebase()};

        av::AudioSamples r(sample_format_, len.timestamp(), channel_layout_, sample_rate_, align_);
        if (sample_format_.isPlanar()) {
            size_t size1ch = sample_format_.requiredBufferSize(1, len.timestamp(), align_);
            for (size_t i=0; i<channel_count_; i++) {
                uint8_t* ptr = r.data(i);
                std::fill(ptr, ptr+size1ch, 0);
            }
        } else {
            size_t size = sample_format_.requiredBufferSize(channel_count_, len.timestamp(), align_);
            uint8_t* ptr = r.data(0);
            std::fill(ptr, ptr+size, 0);
        }
        r.setTimeBase(len.timebase());
        r.setComplete(true);

        if (!r) {
            throw Error("Refusing to return bad frame in audio getBackup");
        }
        return r;
    }
    void setAudioParameters(const decltype(channel_layout_) channel_layout, int sample_rate, av::SampleFormat sample_format) {
        channel_layout_ = channel_layout;
        channel_count_ = av_get_channel_layout_nb_channels(channel_layout);
        sample_format_ = sample_format;
        sample_rate_ = sample_rate;
        size_t size = sample_format_.requiredBufferSize(channel_count_, max_len_, align_);
        if ( (zero_data_ == nullptr) || (size > prev_size_) ) {
            if (zero_data_ != nullptr) {
                delete[] zero_data_;
            }
            zero_data_ = new uint8_t[size];
            std::fill(zero_data_, zero_data_+size, 0);
            prev_size_ = size;
        }
    }
    CorrMediaSpecific(const Parameters&, InstanceData&) {
    }
    ~CorrMediaSpecific() {
        if (zero_data_ != nullptr) {
            delete[] zero_data_;
        }
    }
    void start() {
    }
    void reloadBackupFrame() {
    }
};

template<> struct CorrMediaSpecific<av::VideoFrame> {
protected:
    /*av::Timestamp last_ts_;
    bool ready_ = false;*/
    av::Rational frame_rate_ = {0, 1};
    av::VideoFrame backup_frame_;
    av::PixelFormat pref_pix_fmt_ { AV_PIX_FMT_NONE };
    bool ignore_pref_pix_fmt_ = false;
    int max_width_ = -1;
    int max_height_ = -1;
    InstanceData &app_instance_;
    std::string pict_buf_name_;
public:
    void gotFrame(av::VideoFrame&) {
    }
    void normalizeFrame(av::VideoFrame&) {
    }
    av::VideoFrame getBackup(const av::Timestamp) {
        return backup_frame_; // clone() really necessary?
    }
    av::Timestamp getDelta(av::VideoFrame &frm) {
        (void)frm;

        if (frame_rate_.getNumerator()==0) {
            throw Error("CorrMediaSpecific<VideoFrame> not initialized or frame rate invalid!");
        }
        // 1 frame with inverse of numerator and denominator
        return {1, {frame_rate_.getDenominator(), frame_rate_.getNumerator()}};
    }
    void setFrameRate(const av::Rational fr) {
        logstream << "corr: set frame rate to " << fr << std::endl;
        frame_rate_ = fr;
    }
    static constexpr bool is_video = true;
    CorrMediaSpecific(const Parameters& params, InstanceData &instance): app_instance_(instance) {
        if (params.count("backup_image")==1) {
            std::string bupfile = params["backup_image"];
            av::FormatContext ictx;
            ictx.openInput(bupfile);
            ictx.findStreamInfo();
            av::Stream stream;
            int stream_index;
            for (int i=0; i<ictx.streamsCount(); i++) {
                av::Stream st = ictx.stream(i);
                if (!st.isVideo()) continue;
                stream = st;
                stream_index = i;
                break;
            }
            if (!stream.isValid()) {
                throw Error("Backup image file doesn't contain image stream");
            }
            av::VideoDecoderContext vdec(stream);
            vdec.open();
            while(true) {
                av::Packet pkt = ictx.readPacket();
                if (!pkt) break;
                if (pkt.streamIndex() != stream_index) continue;
                backup_frame_ = vdec.decode(pkt);
                break;
            }
            ictx.close();
        } else if (params.count("backup_picture_buffer")==1) {
            pict_buf_name_ = params["backup_picture_buffer"];
            reloadBackupFrame();
        } else {
            throw Error("backup_image or backup_picture_buffer must be specified!");
        }
        if (!backup_frame_.isValid()) {
            throw Error("Couldn't read backup frame!");
        }
    }
    void reloadBackupFrame() {
        if (pict_buf_name_.empty()) {
            return;
        }
        std::shared_ptr<PictureBuffer> pictbuf = InstanceSharedObjects<PictureBuffer>::get(app_instance_, pict_buf_name_);
        backup_frame_ = pictbuf->getFrame();
    }
    void setPreferredPixelFormat(av::PixelFormat pix_fmt) {
        if (pix_fmt == AV_PIX_FMT_NONE) return;
        if ( (pref_pix_fmt_ != AV_PIX_FMT_NONE) && (pix_fmt != pref_pix_fmt_) ) {
            logstream << "Tried to set different preferred pixel formats: " << pref_pix_fmt_ << ", " << pix_fmt << ". Ignoring all preferences.";
            ignore_pref_pix_fmt_ = true;
        }
        pref_pix_fmt_ = pix_fmt;
        logstream << "Set preferred pixel format to " << pix_fmt;
    }
    void setPreferredResolution(int width, int height) {
        if (width > max_width_) max_width_ = width;
        if (height > max_height_) max_height_ = height;
    }
    void start() {
        av::PixelFormat dst_pix_fmt;
        int dst_width, dst_height;
        bool convert = false;
        if ( (pref_pix_fmt_ != AV_PIX_FMT_NONE) && (!ignore_pref_pix_fmt_) && (pref_pix_fmt_ != backup_frame_.pixelFormat()) ) {
            dst_pix_fmt = pref_pix_fmt_;
            logstream << "Will convert backup frame pixel format to " << dst_pix_fmt;
            convert = true;
        } else {
            dst_pix_fmt = backup_frame_.pixelFormat();
        }
        if ( (max_width_ > 0) && (backup_frame_.width() > max_width_) ) {
            dst_width = max_width_;
            logstream << "Will rescale backup frame to " << dst_width << "px width";
            convert = true;
        } else {
            dst_width = backup_frame_.width();
        }
        if ( (max_height_ > 0) && (backup_frame_.height() > max_height_) ) {
            dst_height = max_height_;
            logstream << "Will rescale backup frame to " << dst_height << "px height";
            convert = true;
        } else {
            dst_height = backup_frame_.height();
        }
        if ( convert && (dst_pix_fmt!=AV_PIX_FMT_NONE) && (dst_width>0) && (dst_height>0) ) {
            logstream << "Converting backup frame from " << backup_frame_.width() << "x" << backup_frame_.height() << " " << backup_frame_.pixelFormat() << " to " <<
                dst_width << "x" << dst_height << " " << dst_pix_fmt;
            av::VideoRescaler rescaler(dst_width, dst_height, dst_pix_fmt,
                                       backup_frame_.width(), backup_frame_.height(), backup_frame_.pixelFormat(), av::SwsFlagLanczos);
            backup_frame_ = rescaler.rescale(backup_frame_, av::throws());
        }
    }
};

template <typename T> class PTSCorrectorNode: public NodeSISO<T, T>, public ISentinel, public IPreferredFormatReceiver {
protected:
    std::shared_ptr<PTSCorrectorCommon> corr_;
    av::Rational timebase_;
    av::Timestamp next_ts_ = NOTS;
    av::Timestamp local_timeshift_ = NOTS;
    av::Timestamp last_no_card_pts_ = NOTS;
    CorrMediaSpecific<T> mspec_;
    T last_frame_;
    //bool first_ = true;
    //const int frame_get_limit_ms_ = 500;
    double max_stalled_sec_ = 1.0;
    double max_freeze_sec_ = 5.0;
    const double max_streams_diff_ = 0.001;
    bool lock_timeshift_ = false;
    unsigned int card_frames_count_ = 0;
    bool last_success_ = true;
    bool try_without_filling_ = false;
    bool sink_full_ = false;
    bool in_correction_ = false;
    bool prev_in_correction_ = false;
    bool write_history_ = false;
    std::atomic<uint64_t> card_status_ {0}; // to avoid unnecessary use of mutexes, both current card state and last change timestamp is stored in a single value
    // is card boolean is the LSB
    // timestamp is the rest

    bool track_wallclock_ = false;
    float max_wallclock_drift_ = 0.5;
    float wallclock_drift_grace_period_ = 3;
    av::Timestamp wallclock_offset_drifted_since_ = NOTS;
    av::Timestamp wallclock_offset_ = NOTS;

    enum class FrameSource: int {
        None = -1,
        Backup = 0,
        Input = 1
    };
    FrameSource last_source_ = FrameSource::None;
    bool freezable() {
        return mspec_.is_video;
    }
    std::string frameSourceToString(const FrameSource fs) {
        if (fs==FrameSource::None) return "none";
        if (fs==FrameSource::Backup) return "backup frames";
        if (fs==FrameSource::Input) return "input frames";
        return "";
    }
    void setFrameSource(const FrameSource newsrc) {
        if (newsrc != last_source_) {
            if (last_source_ != FrameSource::None) {
                logstream << "Switching source: " << frameSourceToString(last_source_) << " -> " << frameSourceToString(newsrc);
            } else {
                logstream << "source: " << frameSourceToString(newsrc);
            }
            last_source_ = newsrc;
            if (newsrc == FrameSource::Backup) {
                card_frames_count_ = 0;
                mspec_.reloadBackupFrame();
            }
        }
    }
    T getBackup(const av::Timestamp req_len, const av::Timestamp frame_pts) {
        //logstream << "CORR BUP! ";
        setCard(true);
        if ( freezable() && last_no_card_pts_.isValid() && ( addTS(frame_pts, negateTS(last_no_card_pts_)).seconds() < max_freeze_sec_ ) && last_frame_.isComplete() ) {
            return last_frame_;
        } else {
            card_frames_count_++;
            if ((card_frames_count_ % 64) == 0) {
                mspec_.reloadBackupFrame();
            }
            return mspec_.getBackup(req_len);
        }
    }
    bool isDiscontinuity(av::Timestamp ts) {
        if (!next_ts_.isValid()) {
            logstream << "First frame, no discontinuity." << std::endl;
            return false;
        }
        if ( (next_ts_.timebase() != timebase_) || (ts.timebase() != timebase_) ) {
            logstream << "Error: inequal timebases: corrector " << timebase_ << ", next_ts " << next_ts_.timebase() << ", PTS " << ts.timebase() << ". Treating as discontinuity.";
            return true;
        }
        if (ts.timestamp() != next_ts_.timestamp()) {
            double diff = fabs((ts.timestamp() - next_ts_.timestamp()) * timebase_.getDouble());
            if (diff < max_streams_diff_) {
                // ignore minor clock fluctuations
                return false;
            }
            logstream << "Discontinuity: " << next_ts_ << " -> " << ts << " diff = " << diff << "s" << std::endl;
            return true;
        } else {
            return false;
        }
        //return (next_ts_.isValid()) && (ts != next_ts_);
    }
    bool outputFrame(T &frm, const av::Timestamp ts, bool drop_if_full = false) {
        frm.setTimeBase(timebase_); // make sure output timebase is correct
        frm.setPts(ts);
        mspec_.gotFrame(frm);
        av::Timestamp prev_next_ts = next_ts_;
        av::Timestamp delta = mspec_.getDelta(frm);

        next_ts_ = addTS(ts, delta);
        if (next_ts_.timestamp() < 0) {
            logstream << "Warning: Setting next_ts to negative value: " << next_ts_ << ", PTS = " << ts << ", length = " << delta;
        }
        //logstream << "corr out: stream " << frm.streamIndex() << " PTS = " << ts << " = " << ts.seconds() << " (" << frm.pts() << " in frame), next_ts_ = " << next_ts_ << std::endl;
        bool success = this->sink_->put(frm, drop_if_full);
        if (success) {
            if (!prev_next_ts) {
                logstream << "outputFrame: PTS: " << ts << " = " << ts.seconds() << "s, set next_ts_ " << prev_next_ts << " -> " << next_ts_ << " = " << next_ts_.seconds() << "s";
            }
            return true;
        } else {
            // failure, rollback!
            next_ts_ = prev_next_ts;
            return false;
        }
    }
    void setCard(const bool is_card) {
        if (is_card != bool(card_status_ & 1)) {
            uint64_t ts = next_ts_.isValid() ? rescaleTS(addTS(next_ts_, negateTS(corr_->startTS())), av::Rational(1, 1000)).timestamp() : 0;
            card_status_ = is_card | (ts << 1);
        }
    }
    void handleWallclockOffset(const av::Timestamp output_pts, const av::Timestamp now_wallclock) {
        if (!track_wallclock_) return;
        av::Timestamp new_offset = addTS(output_pts, negateTS(now_wallclock));
        bool report = false;
        if (wallclock_offset_.isValid()) {
            float diff = abs(addTS(new_offset, negateTS(wallclock_offset_)).seconds());
            if (diff > max_wallclock_drift_) {
                if (wallclock_offset_drifted_since_.isNoPts()) {
                    wallclock_offset_drifted_since_ = now_wallclock;
                } else {
                    float drifted_for = addTS(now_wallclock, negateTS(wallclock_offset_drifted_since_)).seconds();
                    if (drifted_for > wallclock_drift_grace_period_) {
                        report = true;
                    }
                }
            }
        } else {
            report = true;
        }
        if (report) {
            wallclock_offset_ = new_offset;
            corr_->wallclockOffsetChanged(new_offset);
            corr_->reportTimeshiftChange();
        }
    }

public:
    virtual void setPreferredPixelFormat(av::PixelFormat pix_fmt) {
        setPreferredPixelFormatIfPossible<>(pix_fmt);
    }
    virtual void setPreferredResolution(int width, int height) {
        setPreferredResolutionIfPossible<>(width, height);
    }
    virtual std::pair<bool, uint64_t> getCardStatus() {
        uint64_t card_status = card_status_;
        return {card_status & 1, card_status >> 1};
    }
    virtual void start() {
        mspec_.start();
    }
    virtual void process() {
        #define updateLocalTimeshiftPreCorr() { \
            local_timeshift_ = addTSSameTB(local_timeshift_, next_ts_, negateTS(ts)); \
        }
        //T frm = this->source_->get(time_limit_ms_);
        int get_limit_ms;
        if (last_success_) {
            get_limit_ms = (int)(max_stalled_sec_*1000.0);
        } else {
            get_limit_ms = sink_full_ ? 150 : 22;
        }
        T* pfrm = this->source_->peek(get_limit_ms);
        if (pfrm != nullptr) {
            T &frm = *pfrm;
            av::Timestamp frame_wallclock = wallclock.absolute_ts();
            if (frm.isComplete() && frm.pts().isValid()) {
                // success, we have frame
                setFrameSource(FrameSource::Input);
                mspec_.normalizeFrame(frm);
                av::Timestamp ts = frm.pts();
                //logstream << "corr in: stream " << frm.streamIndex() << " PTS = " << ts << std::endl;
                if (ts.timebase() != timebase_) {
                    logstream << "Warning: timebase changed " << timebase_ << " -> " << ts.timebase() << " in the middle of stream! This may cause discontinuity, A/V desync and other weird things." << std::endl;
                write_history_ = true;
                timebase_ = ts.timebase();
                    next_ts_ = rescaleTS(next_ts_, timebase_);
                    logstream << "Rescaling because of timebase change: Set next_ts_ = " << next_ts_;
                    local_timeshift_ = rescaleTS(local_timeshift_, timebase_);
                }

                bool suspend_output = false;
                av::Timestamp duration_filled = { 0, {1, 1000} };
                { // begin lock
                    auto lock = corr_->getLock();

                    if (corr_->hasTimeshift()) {
                        // timeshift is known
                        // (this is not first frame)

                        // Synchronize local_timeshift_:
                        bool desync = false;
                        double timeshift_diff = 0;
                        if (local_timeshift_.isValid()) {
                            timeshift_diff = corr_->timeshiftDiff(local_timeshift_);
                            desync = (timeshift_diff > max_streams_diff_);
                        } else {
                            logstream << "Initializing sentinel. First PTS local in this stream: " << ts;
                            logstream << "We have timebase: " << timebase_ << ", global corrector has timeshift: " << corr_->timeshift();
                            local_timeshift_ = corr_->timeshift(timebase_);
                            logstream << "Initially set timeshift to " << local_timeshift_ << std::endl;
                        }
                        if (desync) {
                            logstream << "Timeshift difference: " << timeshift_diff << std::endl;
                        }
                        assert(local_timeshift_.timebase() == timebase_);
                        assert(ts.timebase() == timebase_);
                        //logstream << "( " << std::setprecision(8) << local_timeshift_.seconds() << " = " << local_timeshift_ << " )";
                        av::Timestamp newts = addTSSameTB(ts, local_timeshift_);
                        //logstream << "pre-sync newts = " << newts << " = " << ts << " + " << local_timeshift_;
                        bool disco_before_sync = isDiscontinuity(newts);
                        if (disco_before_sync) {
                            corr_->nowDiscontinuity();
                        }
                        if (desync) {
                            // first frame
                            // or discontinuity
                            // or desync lasts too long
                            // it's good moment to sync local_timeshift_ to global timeshift
                            av::Timestamp newshift = corr_->timeshift(timebase_);
                            logstream << "Syncing: " << local_timeshift_ << " -> " << newshift << std::endl;
                            local_timeshift_ = newshift;
                            newts = addTSSameTB(ts, newshift);
                        }
                        //logstream << "  PTS " << ts << " -> " << newts << std::endl;
                        ts = newts;

                        bool disco_after_sync = isDiscontinuity(ts);
                        if (disco_after_sync) {
                            if (disco_before_sync) {
                                logstream << "Discontinuity detected (before & after sync)";
                            } else {
                                corr_->nowDiscontinuity();
                                logstream << "Syncing caused discontinuity";
                            }
                        } else {
                            if (disco_before_sync) {
                                logstream << "Strange: syncing fixed discontinuity";
                            } // else no discontinuity at all
                        }


                        // Correct PTS:
                        if (disco_after_sync) {
                            //logstream << "pre-correct ts = " << ts;
                            if (ts < next_ts_) {
                                // PTS jumped backwards
                                logstream << "PTS jumped backwards " << next_ts_ << " -> " << ts;
                                updateLocalTimeshiftPreCorr();
                                // we can safely correct it
                                ts = next_ts_;
                            } else {
                                // PTS jumped forward
                                av::Timestamp rtc = corr_->rtcTS();
                                bool should_fill = !try_without_filling_;
                                if (should_fill) {
                                    if (disco_before_sync) {
                                        // we have discontinuity right now
                                        logstream << "Filling enabled because of input discontinuity";
                                    } else if (corr_->wasDiscontinuityRecently()) {
                                        logstream << "Filling enabled because of discontinuity in other stream";
                                    }
                                } else {
                                    suspend_output = true;
                                    try_without_filling_ = false;
                                }
                                if (should_fill) {
                                    logstream << "PTS jumped forward, filling with backup frames. " << next_ts_ << " -> " << ts;
                                    // fill stream with backup frames
                                    // fast-forward next_ts_ until we reach current PTS
                                    int lim = 5;
                                    while (next_ts_ < ts) {
                                        av::Timestamp bupts = next_ts_;
                                        av::Timestamp buplen = addTSSameTB(rescaleTS(ts, timebase_), rescaleTS(negateTS(next_ts_), timebase_));
                                        T bup = getBackup(buplen, bupts);
                                        logstream << "Generating backup frame to fill at " << bupts << " " << bup.pts();
                                        setFrameTimestamps(bup, frm.pts(), bupts, frame_wallclock);
                                        if (outputFrame(bup, bupts, true)) {
                                            duration_filled = addTS(duration_filled, mspec_.getDelta(bup));
                                        } else {
                                            // output overflow, stop emitting
                                            lim = 0;
                                        }
                                        if ( (--lim) <= 0 ) {
                                            corr_->setTS(bupts);
                                            suspend_output = true;
                                            break;
                                        }
                                    }
                                    // align PTS to next_ts_
                                    try_without_filling_ = true;
                                } else {
                                    logstream << "PTS jumped forward " << next_ts_ << " -> " << ts;
                                    // shift it backwards
                                }
                                updateLocalTimeshiftPreCorr();
                                ts = next_ts_;
                            }
                            // Synchronize corr_ to local_timeshift_
                            // TODO: less naive way of doing it
                            in_correction_ = disco_before_sync || disco_after_sync || desync;
                            corr_->setTimeshift(local_timeshift_, prev_in_correction_ && !in_correction_ && write_history_);
                        } else if (desync) {
                            // change global timeshift if desync
                            // even if there's no discontinuity
                            in_correction_ = disco_before_sync || disco_after_sync || desync;
                            corr_->setTimeshift(local_timeshift_, prev_in_correction_ && !in_correction_ && write_history_);
                        }
                        prev_in_correction_ = in_correction_;
                    } else {
                        // timeshift is unknown
                        // (this is first frame globally)
                        logstream << "Initializing sentinel. First PTS globally: " << ts;
                        corr_->firstTS(ts);
                        local_timeshift_ = corr_->timeshift(timebase_);
                        ts = addTSSameTB(ts, local_timeshift_);
                    }
                    if (!suspend_output) {
                        corr_->setTS(ts);
                    }
                } // end lock
                if (!suspend_output) {
                    setCard(false);
                    handleWallclockOffset(ts, frame_wallclock);
                    setFrameTimestamps(frm, frm.pts(), ts, frame_wallclock);
                    outputFrame(frm, ts); // we don't need overflow prevention logic here because we're outside the lock - we can block without causing Bad Things(TM)
                    last_no_card_pts_ = ts;
                    if (freezable()) last_frame_ = frm;
                    if (!this->source_->pop()) {
                        throw Error("pop() failed! (should never happen)");
                    }
                    last_success_ = true;
                    //logstream << "Setting last_success_ to true";
                } else {
                    double sec = duration_filled.seconds();
                    logstream << "Filled " << sec << "s. Output suspended, not outputting frame.";
                    int ms = static_cast<int>(sec*1000.0/2.0);
                    if (ms<1) ms = 1;
                    wallclock.sleepms(ms);
                }
            } else {
                // empty or invalid frame
                // it usually means EOF / switching inputs
                // TODO: force PTS correction within next frame
                // (if it is really needed . . .)
            }
        }
        if (pfrm==nullptr) {
            // timeout getting frame
            setFrameSource(FrameSource::Backup);
            auto lock = corr_->getLock();
            av::Timestamp rtc = corr_->rtcTS();
            if (!next_ts_.isValid()) {
                next_ts_ = corr_->startTS(timebase_);
                logstream << "Notice: setting next_ts_ from start TS = " << next_ts_ << std::endl;
            }
            double stalled_sec = addTS(rtc, negateTS(next_ts_)).seconds();
            //logstream << "Stalled for " << stalled_sec << " = " << rtc << " - " << next_ts_ << ", max " << max_stalled_sec_;
            if ( (!last_success_) || (stalled_sec>max_stalled_sec_) ) {
                sink_full_ = false;
                //logstream << "TIMEOUT ";
                // timeout exceeded
                // generate some backup frames...
                int lim = 5;
                while (next_ts_ < rtc) {
                    //logstream << "Generating backup frame because of timeout at " << next_ts_;
                    av::Timestamp diff = addTSSameTB(rescaleTS(rtc, timebase_), negateTS(rescaleTS(next_ts_, timebase_)));
                    if (diff.timestamp() <= 0) {
                        break;
                    }
                    T frm = getBackup(diff, next_ts_);
                    // setTS sets sync-point between PTS and wallclock
                    // we don't want it here because no meaningful data is received
                    // so we should synchronize to last real sync point
                    //corr_->setTS(next_ts_);
                    //logstream << "Outputting backup frame PTS = " << frm.pts() << ", delta/length = " << mspec_.getDelta(frm);
                    setFrameTimestamps(frm, av::NoPts, next_ts_, av::NoPts);
                    if (!outputFrame(frm, next_ts_, true)) {
                        sink_full_ = true;
                        break;
                    }
                    if ((--lim)<=0) break;
                }
            }
            last_success_ = false;
        }
        #undef updateLocalTimeshiftPreCorr
    }
    PTSCorrectorNode(std::unique_ptr<Source<T>> &&source,
                     std::unique_ptr<Sink<T>> &&sink,
                     decltype(corr_) corr,
                     const Parameters &params,
                     const double max_stalled_sec,
                     const double max_freeze_sec,
                     const bool forward_start_shift,
                     const double max_streams_diff,
                     InstanceData &instance):
        NodeSISO<T, T>(std::move(source), std::move(sink)),
        corr_(corr),
        mspec_(params, instance),
        max_stalled_sec_(max_stalled_sec),
        max_freeze_sec_(max_freeze_sec),
        max_streams_diff_(max_streams_diff) {
        std::shared_ptr<ITimeBaseSource> tbmd = this->template findNodeUp<ITimeBaseSource>();
        if (tbmd) {
            timebase_ = tbmd->timeBase();
            logstream << "Set timebase to " << timebase_;
            corr_->addTimebase(timebase_);
            //next_ts_ = {0, timebase_};
            //local_timeshift_ = {0, timebase_};
        } else {
            throw Error("PTS corrector needs stream time base!");
        }
        if (!forward_start_shift) {
            next_ts_ = corr_->startTS(timebase_);
        }
        initMediaSpecific();
    }
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setFrameRate)> void setFrameRateIfPossible() {
        std::shared_ptr<IFrameRateSource> vfr = this->template findNodeUp<IFrameRateSource>();
        if (vfr) {
            mspec_.setFrameRate(vfr->frameRate());
        } else {
            throw Error("Unknown frame rate!");
        }
    }
    template<typename ...Args> void setFrameRateIfPossible(Args...) {
        // NOOP
    }
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setPreferredPixelFormat)> void setPreferredPixelFormatIfPossible(av::PixelFormat pix_fmt) {
        mspec_.setPreferredPixelFormat(pix_fmt);
    }
    template<typename ...Args> void setPreferredPixelFormatIfPossible(Args...) {
        // NOOP
    }
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setPreferredResolution)> void setPreferredResolutionIfPossible(int width, int height) {
        mspec_.setPreferredResolution(width, height);
    }
    template<typename ...Args> void setPreferredResolutionIfPossible(Args...) {
        // NOOP
    }
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setAudioParameters)> void setAudioParametersIfPossible() {
        std::shared_ptr<IAudioMetadataSource> amd = this->template findNodeUp<IAudioMetadataSource>();
        if (amd) {
            mspec_.setAudioParameters(amd->channelLayout(), amd->sampleRate(), amd->sampleFormat());
        } else {
            throw Error("Unknown audio parameters!");
        }
    }
    template<typename ...Args> void setAudioParametersIfPossible(Args...) {
        // NOOP
    }
    void initMediaSpecific() {
        setFrameRateIfPossible<>();
        setAudioParametersIfPossible<>();
    }

    // mspec_.setFrameRate exists only for video streams, so we abuse it here for SFINAE
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setFrameRate)> void setInitialPictureBuffer(av::VideoFrame frm) {
        last_frame_ = frm;
        if (last_no_card_pts_.isNoPts()) {
            last_no_card_pts_ = corr_->startTS();
        }
    }
    template<typename ...Args> void setInitialPictureBuffer(Args...) {
        throw Error("initial_picture_buffer specified for non-video sentinel");
    }

    // mspec_.setFrameRate exists only for video streams, so we abuse it here for SFINAE
    template<typename MSpec = decltype(mspec_), typename = decltype(&MSpec::setFrameRate)> void setFrameTimestamps(
            av::VideoFrame& frm, const av::Timestamp& ts_in, const av::Timestamp& ts_out, const av::Timestamp& ts_wallclock) {
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
    template<typename ...Args> void setFrameTimestamps(Args...) {
        // NOOP
    }

    static std::shared_ptr<PTSCorrectorNode> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::string cg = "default";
        if (params.count("correction_group")==1) {
            cg = params.at("correction_group");
        }
        double max_stalled_sec = 1.0;
        if (params.count("timeout")==1) {
            max_stalled_sec = params.at("timeout");
        }
        double max_freeze_sec = 5.0;
        if (params.count("freeze")==1) {
            max_freeze_sec = params.at("freeze");
        }
        bool forward_start_shift = false;
        if (params.count("forward_start_shift")) {
            forward_start_shift = params.at("forward_start_shift");
        }
        double max_streams_diff = 0.001;
        if (params.count("max_streams_diff")) {
            max_streams_diff = params.at("max_streams_diff");
        }
        auto corr = InstanceSharedObjects<PTSCorrectorCommon>::get(nci.instance, cg);
        if (params.count("reporting_url")) {
            corr->setReportingURL(params.at("reporting_url"));
        }
        if (params.count("lock_timeshift")) {
            if (params.at("lock_timeshift")) {
                corr->lockTimeshift();
            }
        }
        if (params.count("start_ts")) {
            corr->start_ts_ = av::Timestamp(AVTS(params["start_ts"].get<float>()*1000.0f+0.5f), {1, 1000});
        }
        auto r = NodeSISO<T, T>::template createCommon<PTSCorrectorNode>(edges, params, corr, params, max_stalled_sec, max_freeze_sec, forward_start_shift, max_streams_diff, nci.instance);
        if (params.count("initial_picture_buffer")) {
            std::string pict_buf_name = params["initial_picture_buffer"];
            std::shared_ptr<PictureBuffer> pictbuf = InstanceSharedObjects<PictureBuffer>::get(nci.instance, pict_buf_name);
            r->setInitialPictureBuffer(pictbuf->getFrame());
        }
        if (params.count("track_wallclock")) {
            r->track_wallclock_ = params["track_wallclock"];
        }
        if (params.count("max_wallclock_drift")) {
            r->max_wallclock_drift_ = params["max_wallclock_drift"];
        }
        if (params.count("wallclock_drift_grace_period")) {
            r->wallclock_drift_grace_period_ = params["wallclock_drift_grace_period"];
        }
        if (params.count("history_file")) {
            r->write_history_ = true;
            corr->openHistoryFile(params.at("history_file"));
        }
        if (params.count("history_file_text")) {
            r->write_history_ = true;
            corr->openHistoryFileText(params.at("history_file_text"));
        }
        return r;
    }
};

class VideoSentinel: public PTSCorrectorNode<av::VideoFrame> {};
class AudioSentinel: public PTSCorrectorNode<av::AudioSamples> {};

DECLNODE(sentinel_video, VideoSentinel);
DECLNODE(sentinel_audio, AudioSentinel);
