#include "stats.hpp"

#include <cstdint>
#include <utility>
#include <avcpp/audioresampler.h>
#include "libavutil/dict.h"
#include "util.hpp"
#include "graph_mgmt.hpp"
#include "rest_client.hpp"

#ifdef SYNCMETER
#include "syncmeter.hpp"
#endif

using nlohmann::json;

template<typename Y, bool autoclean = true> class TimedHistory {
public:
    struct Item {
        av::Timestamp time;
        Y value;
    };
protected:
    double max_age_sec_;
    std::list<Item> items_;
public:
    void cleanupWithRefTS(const av::Timestamp now) {
        if (now.isNoPts()) return;
        av::Timestamp min_ts = { now.timestamp() - static_cast<int64_t>(max_age_sec_/now.timebase().getDouble()), now.timebase() };
        items_.remove_if([min_ts](Item &item) {
            return (item.time < min_ts);
            // FIXME: this is inefficient. We should discard all items before the first not-too-old item.
        });
    }
    TimedHistory(const double max_age_sec = 30): max_age_sec_(max_age_sec) {
    }
    void setMaxAge(const double seconds) {
        max_age_sec_ = seconds;
    }
    void cleanup() {
        if (items_.empty()) return;
        // use last value as reference time
        cleanupWithRefTS(items_.back().time);
    }
    void clearAll() {
        items_.clear();
    }
    double itemsPerSecond() {
        size_t nitems = items_.size();
        if (nitems==0) {
            return 0; // or maybe NAN?
        }
        av::Timestamp deltat = addTS(items_.back().time, negateTS(items_.front().time));
        if (deltat.timestamp()==0) {
            return 0; // or maybe NAN?
        }
        return ((double)(nitems-1)) / deltat.seconds();
    }
    void push(const av::Timestamp time, const Y value) {
        items_.push_back({ time, value });
        if (autoclean) cleanup();
    }
    void pushWallclockNow(const Y value) {
        push(wallclock.ts(), value);
    }
    bool empty() {
        return items_.empty();
    }
    Item& earliest() {
        return items_.front();
    }
    Item& latest() {
        return items_.back();
    }
    template<typename Ret = Y> Ret valueDiff() {
        return latest().value - earliest().value;
    }
    av::Timestamp timeDiff() {
        return latest().time - earliest().time;
    }
};

/*
 * mss stats:{"AV_diff":1726560.435,"queued_packets":0,"stalled_seconds":0.0,"streams_audio":[{"codec":"aac","frame_num":16,"index":0,"kbitrate":0.0,"samplerate":0.0,"speed":0.0,"type":"A"}],"streams_video":[{"codec":"h264","field_order":"PROGRESSIVE","fps":0.0,"frame_num":0,"height":720,"index":1,"kbitrate":0.0,"speed":0.0,"type":"V","width":1280}],"card":false}
*/

class AbstractStreamStats {
public:
    virtual void fillStats(json &jstats) = 0;
    virtual av::Timestamp lastTS() = 0;
    virtual ~AbstractStreamStats() {
    }
};

class StatsSender;

class NodeAccessor {
protected:
    std::weak_ptr<NodeWrapper> nw_;
    std::shared_ptr<NodeManager> manager_;
    std::string name_;
public:
    NodeAccessor() {
    }
    NodeAccessor(std::shared_ptr<NodeManager> manager, const std::string node_name): manager_(manager), name_(node_name) {
    }
    std::shared_ptr<NodeWrapper> wrapper() {
        int tries = 2;
        std::shared_ptr<NodeWrapper> r;
        while (tries>0) {
            r = nw_.lock();
            if (r) {
                return r;
            } else {
                if (!manager_) {
                    break;
                }
                try {
                    nw_ = manager_->node(name_);
                } catch (std::exception &e) {
                    break;
                }
                tries--;
            }
        }
        return nullptr;
    }
    template<typename T> std::shared_ptr<T> node() {
        std::shared_ptr<NodeWrapper> nw = wrapper();
        if (nw == nullptr) {
            return nullptr;
        }
        return std::dynamic_pointer_cast<T>(nw->node());
    }
};

template<typename T> class StreamStats: public AbstractStreamStats {
protected:
    std::shared_ptr<NodeManager> manager_;
    std::weak_ptr<NodeWrapper> decoder_;
    std::string dec_node_name_;
    size_t total_bytes_ = 0;
    size_t total_frames_ = 0;
    int stream_index_ = -1;
    DiscontinuityDetector packet_dd_, frame_dd_;
    TimedHistory<av::Timestamp> ts_by_rtc_;
    TimedHistory<size_t> bytes_;
    std::recursive_mutex pre_dec_mutex_;
    std::recursive_mutex post_dec_mutex_;
    StatsSender *sender_;
    double max_age_;
    bool clear_on_send_ = false;
    bool processes_decoded_frames_;

    template<typename U> std::shared_ptr<U> decoderAs() {
        try {
            std::shared_ptr<NodeWrapper> nw = decoder_.lock();
            if (nw == nullptr) {
                // if decoding node was deleted, maybe it was recreated afterwards?
                decoder_ = manager_->node(dec_node_name_);
                if (!decoder_.expired()) {
                    // pointer reclaimed, retry by recursion
                    return decoderAs<U>();
                } else {
                    return nullptr;
                }
            }
            std::shared_ptr<U> dec = std::dynamic_pointer_cast<U>(nw->node());
            return dec;
        } catch (std::exception &e) {
            return nullptr;
        }
    }
    std::shared_ptr<IDecoder> decoder() {
        return decoderAs<IDecoder>();
    }
    std::string decoderCodecName() {
        std::shared_ptr<IDecoder> dec = decoder();
        if (dec) {
            return dec->codecName();
        } else {
            return "???";
        }
    }
public:
    virtual void resetHistory() {
        logstream << "Discontinuity in stream " << stream_index_ << ", purging statistics.";
        ts_by_rtc_.clearAll();
        bytes_.clearAll();
    }
    void resetHistoryWrapper() {
        std::lock(pre_dec_mutex_, post_dec_mutex_);
        {
            std::lock_guard<decltype(pre_dec_mutex_)> lock1(pre_dec_mutex_, std::adopt_lock);
            std::lock_guard<decltype(post_dec_mutex_)> lock2(post_dec_mutex_, std::adopt_lock);
            resetHistory();
        }
    }
    virtual void processDecodedFrame(const T &frm) = 0;
    virtual void fillMediaSpecificStatsPreDec(json &jstats) {
    };
    virtual void fillMediaSpecificStatsPostDec(json &jstats) = 0;

    StreamStats(std::shared_ptr<NodeManager> manager, json &jobj, StatsSender* sender, const double max_age);

    virtual av::Timestamp lastTS() override {
        if (ts_by_rtc_.empty()) return NOTS;
        return ts_by_rtc_.latest().value;
    }

    virtual void fillStats(json &jstats) override {
        {
            std::lock_guard<decltype(pre_dec_mutex_)> lock(pre_dec_mutex_);
            jstats["frame_num"] = total_frames_;
            if (!bytes_.empty()) {
                jstats["kbitrate"] = static_cast<double>(bytes_.valueDiff<>())*8.0 / (1024.0*bytes_.timeDiff().seconds());
                jstats["speed"] = ts_by_rtc_.valueDiff<>().seconds() / ts_by_rtc_.timeDiff().seconds();
                jstats["duration_analyzed"] = bytes_.timeDiff().seconds();
            }
            if (stream_index_ != -1) {
                jstats["index"] = stream_index_;
            }
            try {
                std::shared_ptr<IDecoder> dec = decoder();
                if (dec != nullptr) {
                    jstats["codec"] = dec->codecName();
                    jstats["type"] = dec->codecMediaTypeString();
                }
            } catch (std::exception &e) {
            }
            fillMediaSpecificStatsPreDec(jstats);
            if (clear_on_send_) {
                bytes_.clearAll();
                ts_by_rtc_.clearAll();
            }
        }
        {
            std::lock_guard<decltype(post_dec_mutex_)> lock(post_dec_mutex_);
            fillMediaSpecificStatsPostDec(jstats);
        }
    }
};

class VideoStreamStats: public StreamStats<av::VideoFrame> {
    int width_ = -1;
    int height_ = -1;
    av::PixelFormat pix_fmt_;
    av::Timestamp last_pts_ = NOTS;
    size_t missing_frames_ = 0;
    double frame_duration_ = -1;
public:
    using StreamStats<av::VideoFrame>::StreamStats;
    virtual void processDecodedFrame(const av::VideoFrame &frm) override {
        width_ = frm.width();
        height_ = frm.height();
        pix_fmt_ = frm.pixelFormat();
        if (frame_duration_ <= 0) {
            std::shared_ptr<IFrameRateSource> vframerate = decoderAs<IFrameRateSource>();
            if (vframerate != nullptr) {
                av::Rational fps = vframerate->frameRate();
                frame_duration_ = double(fps.getDenominator()) / double(fps.getNumerator());
            }
        }
        if (last_pts_.isValid() && (frame_duration_>0) && (frm.pts().timebase() == last_pts_.timebase())) {
            av::Timestamp delta = addTSSameTB(frm.pts(), negateTS(last_pts_));
            if (delta.timestamp()>0) {
                double diff_frames = delta.seconds() / frame_duration_;
                if (diff_frames > 1.001 || diff_frames < 0.999) {
                    logstream << "stats: video PTS jumped " << last_pts_.timestamp() << " -> " << frm.pts().timestamp() << ", delta = " << delta.seconds() << "s = " << diff_frames << "frames";
                }
                if (diff_frames > 1.001) {
                    missing_frames_ += long(diff_frames-1);
                }
            }
        }
        last_pts_ = frm.pts();
    }
    virtual void fillMediaSpecificStatsPreDec(json &jstats) override {
        std::shared_ptr<IFrameRateSource> vframerate = decoderAs<IFrameRateSource>();
        if (vframerate != nullptr) {
            av::Rational fps = vframerate->frameRate();
            jstats["fps_md"] = std::to_string(fps.getNumerator()) + "/" + std::to_string(fps.getDenominator());
            frame_duration_ = double(fps.getDenominator()) / double(fps.getNumerator());
        }

        if (!this->bytes_.empty()) {
            jstats["fps"] = bytes_.itemsPerSecond();
        }
        std::shared_ptr<IDecoder> dec = decoder();
        if (dec != nullptr) {
            jstats["field_order"] = dec->fieldOrderString();
        }
        if (!processes_decoded_frames_) {
            std::shared_ptr<IVideoFormatSource> vformat = decoderAs<IVideoFormatSource>();
            if (vformat != nullptr) {
                jstats["width"] = vformat->width();
                jstats["height"] = vformat->height();
                jstats["pix_fmt"] = vformat->pixelFormat().name();
            }
        }
    }
    virtual void fillMediaSpecificStatsPostDec(json &jstats) override {
        if (width_ != -1 && height_ != -1) {
            jstats["width"] = width_;
            jstats["height"] = height_;
            jstats["pix_fmt"] = pix_fmt_.name();
        }
        jstats["dropped_frames"] = missing_frames_;
        if (clear_on_send_) {
            missing_frames_ = 0;
        }
    }
};

#include "audio_parameters.hpp"

static double todb(const double v) {
    return 20.0*std::log10(v);
}

class SoundAnalyzer {
private:
    using sample_t = int32_t;
    static constexpr AVSampleFormat sample_format = AV_SAMPLE_FMT_S32P;
    static constexpr sample_t sample_max = INT32_MAX;
    struct FrameStats {
        static constexpr size_t histogram_bins = 17;
        static constexpr int histogram_db_range = 60;
        static constexpr double clipped_level = 0.9885309; // -0.1 dB
        static constexpr double epsilon_level = 0.001; // -60 dB
        double peak;
        double squares_sum;
        size_t histogram[histogram_bins];
        size_t samples_count;
        FrameStats operator+=(const FrameStats &other) {
            this->peak = std::max(this->peak, other.peak);
            this->squares_sum += other.squares_sum;
            for (size_t i=0; i<histogram_bins; i++) {
                this->histogram[i] += other.histogram[i];
            }
            this->samples_count += other.samples_count;
            return *this;
        }
        FrameStats operator+(const FrameStats &other) const {
            FrameStats r = *this;
            r += other;
            return r;
        }
        double getRMS() const {
            return std::sqrt(squares_sum / (double)samples_count);
        }
        size_t getClipped() const {
            return histogram[histogram_bins-1];
        }
        static size_t getBin(const double sample) {
            double value = std::abs(sample);
            if (value<epsilon_level) return 0;
            if (value>clipped_level) return histogram_bins-1;
            int ret = std::ceil((histogram_bins-2) + (histogram_bins-2)*20.0*std::log10(value)/histogram_db_range);
            if (ret >= histogram_bins) ret = histogram_bins-1;
            if (ret < 0) ret = 0;
            return ret;
        }
        FrameStats(): peak(0), squares_sum(0), samples_count(0) {
            for (size_t i=0; i<histogram_bins; i++) histogram[i] = 0;
        }
        FrameStats(const sample_t *buffer, const size_t buffsize): FrameStats() {
            for (size_t i=0; i<buffsize; i++) {
                double sample = (double)buffer[i]/sample_max;

                // PEAK
                double absv = std::abs(sample);
                if (absv > this->peak) this->peak = absv;

                // RMS
                this->squares_sum += (double)sample*(double)sample;

                // Histogram
                this->histogram[getBin(sample)] += 1;

            }
            this->samples_count += buffsize;
        }
        std::tuple<size_t, size_t> histogramPeak() const {
            size_t maxindex = 0;
            size_t maxval = histogram[0];
            for (size_t i=1; i<histogram_bins; i++) {
                if (histogram[i]>maxval) {
                    maxindex = i;
                    maxval = histogram[i];
                }
            }
            return std::tuple<size_t, size_t>{maxindex, maxval};
        }
        template<typename ProblemsListType> void detectProblems(ProblemsListType &problems, std::vector<FrameStats> &channels) const {
            auto peak_info = histogramPeak();
            if (getClipped()>0) problems.push_back("CLIPPED");
            if (std::get<0>(peak_info) == histogram_bins-1 || std::get<0>(peak_info) == histogram_bins-2) problems.push_back("SEVERELY_CLIPPED");
            if (std::get<0>(peak_info) == 0) problems.push_back("QUIET");
            bool channels_inequal = false;
            for (const auto &channel: channels) {
                if ( std::abs(todb(channel.getRMS()) - todb(getRMS())) > 3.0) {
                    channels_inequal = true;
                    break;
                }
            }
            if (channels_inequal) problems.push_back("IMBALANCE");
        }
        void print(std::ostream &ostream) {
            ostream <<
                "Peak " << todb(peak) << "dB, " <<
                "RMS " << todb(getRMS()) << "dB, " <<
                "Clipped " << getClipped() << ", "
                "Histogram: ";
            for (size_t i=0; i<histogram_bins; i++) {
                ostream << histogram[i] << "  ";
            }
            ostream << std::endl;
            //    "Histogram peak @ bin " << std::get<0>(histogramPeak()) << std::endl;
        }
    };

    AudioParameters audio_params_;
    std::unique_ptr<av::AudioResampler> resampler_;
    size_t channels_num = 0;
    double stats_seconds_max = 60;
    double last_sr_ = 0;
    float stats_seconds = 0;
    std::list<std::shared_ptr<std::vector<FrameStats> > > stats;
    std::list<std::shared_ptr<std::vector<FrameStats> > > rtstats;
    static float samplesDuration(const av::AudioSamples &samples) {
        return (float)samples.samplesCount()/(float)samples.sampleRate();
    }
    void processConvertedSamples(const av::AudioSamples &samples) {
        std::shared_ptr<std::vector<FrameStats> > chstats = std::make_shared<std::vector<FrameStats> >(channels_num);
        for (size_t ch=0; ch<channels_num; ch++) {
            const sample_t *chbuff = reinterpret_cast<const sample_t*>(samples.data(ch));
            (*chstats)[ch] = FrameStats(chbuff, samples.samplesCount());
            //chstats[ch].print(logstream);
        }
        stats.push_back(chstats);
        rtstats.push_back(chstats);
        stats_seconds += samplesDuration(samples);
        try {
            while (stats_seconds > stats_seconds_max) {
                stats_seconds -= (float)(*stats.front())[0].samples_count/(float)audio_params_.sample_rate;
                stats.pop_front();
            }
        } catch (std::exception &e) {
            logstream << "Error cleaning sound analyzer: " << e.what() << std::endl;
        }
    }
public:
    SoundAnalyzer() {
    }
    void setMaxAge(const double sec) {
        stats_seconds_max = sec;
    }
    void processSamples(const av::AudioSamples &in_samples) {
        if (in_samples.isComplete() && in_samples.samplesCount()>0) {
            if ( (resampler_==nullptr) || (audio_params_ != AudioParameters(in_samples)) ) {
                audio_params_ = AudioParameters(in_samples);
                // preserve channel layout & sample rate
                // convert only sample format
                resampler_ = make_unique<av::AudioResampler>(audio_params_.channel_layout, audio_params_.sample_rate, sample_format,
                                                             audio_params_.channel_layout, audio_params_.sample_rate, audio_params_.sample_format);
                channels_num = in_samples.channelsCount();
                resetHistory();
                logstream << "Sound analyzer started: " << audio_params_;
            }
            resampler_->push(in_samples);

            while(true) {
                av::AudioSamples samples(sample_format, in_samples.samplesCount(), audio_params_.channel_layout, audio_params_.sample_rate);
                bool got = resampler_->pop(samples, true);
                if (!got) break;
                processConvertedSamples(samples);
            }
        }
    }
    void resetHistory() {
        for (auto &v: stats) {
            v->clear();
        }
        for (auto &v: rtstats) {
            v->clear();
        }
        stats_seconds = 0;
        last_sr_ = 0;
    }
    void getStats(nlohmann::json &jobj, const double sr_to_compare = 0) {
        // long-term statistics:
        if (!stats.empty()) {
            std::vector<FrameStats> channel_sums(channels_num);
            for (const auto &channels: stats) {
                for (size_t ch=0; ch<channels_num; ch++) {
                    channel_sums[ch] += (*channels)[ch];
                }
            }
            jobj["duration_analyzed_sound"] = stats_seconds;
            FrameStats sum;
            for (const auto &channel: channel_sums) {
                sum += channel;
            }
            jobj["rms"] = todb(sum.getRMS());
            jobj["peak"] = todb(sum.peak);
            jobj["clipped_samples"] = sum.getClipped();

            nlohmann::json problems;
            sum.detectProblems(problems, channel_sums);
            if ((sr_to_compare>0) && (last_sr_>0)) {
                if (std::fabs((sr_to_compare-last_sr_)/last_sr_) > 0.02) {
                    problems.push_back("SAMPLE_RATE_DRIFT");
                }
            }
            if (!problems.empty()) {
                jobj["problems"] = problems;
            }
        }
        //sum.print(logstream);

        // real-time statistics:
        if (!rtstats.empty()) {
            std::vector<FrameStats> channel_sums(channels_num);
            for (const auto &channels: rtstats) {
                for (size_t ch=0; ch<channels_num; ch++) {
                    channel_sums[ch] += (*channels)[ch];
                }
            }
            nlohmann::json jchannels;
            for (const auto &channel: channel_sums) {
                nlohmann::json jch;
                jch["rms"] = todb(channel.getRMS());
                jch["peak"] = todb(channel.peak);
                jch["clipped_samples"] = channel.getClipped();
                jchannels.push_back(jch);
            }
            jobj["channels_rt"] = jchannels;
            rtstats.clear();
        }
    }
};

constexpr AVSampleFormat SoundAnalyzer::sample_format;


class AudioStreamStats: public StreamStats<av::AudioSamples> {
protected:
    int sample_rate_md_ = -1;
    size_t total_samples_ = 0;
    std::string channel_layout_;
    TimedHistory<size_t> samples_;
    SoundAnalyzer analyzer_;
    bool has_r128_stats_ = false;
    static constexpr size_t MAX_AUDIO_CHANNELS = 32;
    float momentary_, short_term_, integrated_, loudness_range_, lra_low_, lra_high_;
    float true_peaks_[MAX_AUDIO_CHANNELS] = {0};
    size_t channels_count_ = 0;
public:
    AudioStreamStats(std::shared_ptr<NodeManager> manager, json &jobj, StatsSender* sender, const double max_age):
        StreamStats<av::AudioSamples>(manager, jobj, sender, max_age) {
        samples_.setMaxAge(max_age_);
        analyzer_.setMaxAge(max_age_);
    }
    void extractR128Stats(const av::AudioSamples &frm) {
        #define bail { has_r128_stats_ = false; return; }
        #define META_PREFIX "lavfi.r128."
        static const std::string TRUE_PEAK_META_PREFIX = META_PREFIX "true_peaks_per_frame_ch";

        if (!frm.raw()) bail;
        const AVDictionary *md = frm.raw()->metadata;
        if (!md) bail;

        #define readval(target_var, key) { \
            AVDictionaryEntry* entry = av_dict_get(md, META_PREFIX key, nullptr, 0); \
            if (!entry) bail; \
            target_var = std::stof(entry->value); \
        }
        // FIXME in case of Momentary loudness 0.4s we're losing precision when we send the data in 1s intervals
        readval(momentary_, "M");
        readval(short_term_, "S");
        readval(integrated_, "I");
        readval(loudness_range_, "LRA");
        readval(lra_low_, "LRA.low");
        readval(lra_high_, "LRA.high");

        for (size_t ch=0 ;; ch++) {
            std::string key = TRUE_PEAK_META_PREFIX + std::to_string(ch);
            AVDictionaryEntry* entry = av_dict_get(md, key.c_str(), nullptr, 0);

            if ((!entry) || (ch >= MAX_AUDIO_CHANNELS)) {
                channels_count_ = ch;
                break;
            }

            float tp = std::stof(entry->value);
            if (tp > true_peaks_[ch]) {
                true_peaks_[ch] = tp;
            }
        }
        

        has_r128_stats_ = true;
        #undef META_PREFIX
        #undef readval
        #undef bail
    }
    virtual void processDecodedFrame(const av::AudioSamples &frm) override {
        sample_rate_md_ = frm.sampleRate();
        channel_layout_ = frm.channelsLayoutString();
        total_samples_ += frm.samplesCount();
        samples_.push(frm.pts(), total_samples_);
        extractR128Stats(frm);
        analyzer_.processSamples(frm);
    }
    virtual void fillMediaSpecificStatsPostDec(json &jstats) override {
        double sr_by_ts = 0;
        if (!samples_.empty()) {
            sr_by_ts = static_cast<double>(samples_.valueDiff<>()) / samples_.timeDiff().seconds();
            jstats["samplerate"] = sr_by_ts;
        }
        if (sample_rate_md_ != -1) {
            jstats["samplerate_md"] = sample_rate_md_;
        }
        if (!channel_layout_.empty()) {
            jstats["channel_layout"] = channel_layout_;
        }
        if (has_r128_stats_) {
            Parameters jobj;
            jobj["M"] = momentary_;
            jobj["S"] = short_term_;
            jobj["I"] = integrated_;
            jobj["LRA"] = loudness_range_;
            jobj["LRA_low"] = lra_low_;
            jobj["LRA_high"] = lra_high_;
            Parameters tpobj;
            for (size_t ch=0; ch<channels_count_; ch++) {
                tpobj.push_back(20.0f*log10(true_peaks_[ch]));
            }
            jobj["TP"] = tpobj;
            jstats["r128"] = jobj;

            // clean up the stats:
            for (size_t ch=0; ch<MAX_AUDIO_CHANNELS; ch++) {
                true_peaks_[ch] = 0;
            }
            channels_count_ = 0;
        }
        analyzer_.getStats(jstats, sr_by_ts);
        if (clear_on_send_) {
            samples_.clearAll();
            analyzer_.resetHistory();
        }
    }
    virtual void resetHistory() override {
        StreamStats<av::AudioSamples>::resetHistory();
        samples_.clearAll();
        analyzer_.resetHistory();
    }
};

class StatsSender: public std::enable_shared_from_this<StatsSender> {
protected:
    RESTEndpoint rest_;
    std::string name_;
    std::shared_ptr<NodeManager> manager_;
    AVTS interval_ms_ = 1000;
    double history_max_age_ = 30;
    av::Timestamp last_frame_rtc_ = NOTS;
    NodeAccessor sentinel_;
    std::list<std::shared_ptr<AbstractStreamStats>> stats_video_, stats_audio_;
    #ifdef SYNCMETER
    std::list<SyncMeter::Meter> sync_meters_;
    #endif
    static void fillStatsList(json &jglobal, const std::string key, std::list<std::shared_ptr<AbstractStreamStats>> &stats) {
        json jlist;
        for (auto &sst: stats) {
            json jobj;
            sst->fillStats(jobj);
            jlist.push_back(jobj);
        }
        jglobal["streams_" + key] = jlist;
    }
    void send() {
        json jglobal;
        if (last_frame_rtc_.isValid()) {
            jglobal["stalled_seconds"] = (wallclock.ts() - last_frame_rtc_).seconds();
        }
        if ( (!stats_video_.empty()) && (!stats_audio_.empty()) ) {
            av::Timestamp last_ts_audio = stats_audio_.front()->lastTS();
            av::Timestamp last_ts_video = stats_video_.front()->lastTS();
            if (last_ts_audio.isValid() && last_ts_video.isValid()) {
                jglobal["AV_diff"] = addTS(last_ts_audio, negateTS(last_ts_video)).seconds();
            }
        }
        jglobal["name"] = name_;
        auto sen = sentinel_.node<ISentinel>();
        if (sen) {
            std::pair<bool, uint64_t> card_status = sen->getCardStatus();
            jglobal["card"] = card_status.first;
            jglobal["last_card_switch_ms"] = card_status.second;
        }
        fillStatsList(jglobal, "video", stats_video_);
        fillStatsList(jglobal, "audio", stats_audio_);
        std::ostringstream ss;
        ss << jglobal;
        rest_.send("", ss.str());
    }
    void parseStream(json &jobj, const std::string subkey) {
        if (!jobj.is_object()) {
            throw Error("Parsing statistics definition failed: Object required");
        }
        std::shared_ptr<AbstractStreamStats> sst;
        if (subkey=="video") {
            sst = std::make_shared<VideoStreamStats>(manager_, jobj, this, history_max_age_);
            stats_video_.push_back(sst);
        } else if (subkey=="audio") {
            sst = std::make_shared<AudioStreamStats>(manager_, jobj, this, history_max_age_);
            stats_audio_.push_back(sst);
        } else {
            throw Error("Unsupported media type: " + subkey);
        }
    }
    void parseStreams(json &jstreams, const std::string subkey) {
        if (jstreams.count(subkey)==0) return;
        json jv = jstreams[subkey];
        if (jv.is_array()) {
            for (json &js: jv) {
                parseStream(js, subkey);
            }
        } else {
            parseStream(jv, subkey);
        }
    }
public:
    StatsSender(json params, std::shared_ptr<NodeManager> manager): manager_(manager) {
        if (params.count("url")) {
            rest_.setBaseURL(params["url"].get<std::string>());
        }
        if (params.count("interval")) {
            interval_ms_ = static_cast<AVTS>(params["interval"].get<double>() * 1000.0);
        }
        if (params.count("max_age")) {
            history_max_age_ = params["max_age"];
        }
        if (params.count("name")) {
            name_ = params["name"].get<std::string>();
        }
        if (params.count("sentinel")) {
            std::string sentinel_name = params["sentinel"].get<std::string>();
            sentinel_ = NodeAccessor(manager_, sentinel_name);
        }
        #ifdef SYNCMETER
        if (params.count("syncmeters")) {
            json jmeters = params["syncmeters"];
            for (json &jm: jmeters) {
                bool audio = true;
                bool video = false;
                std::shared_ptr<Edge<av::AudioSamples>> audio_edge;
                std::shared_ptr<Edge<av::VideoFrame>> video_edge;
                std::string prefix = "syncmeter: ";
                for (std::string edgename: jm) {
                    if (audio) {
                        audio_edge = manager_->edges()->template find<av::AudioSamples>(edgename);
                        audio = false;
                        video = true;
                        prefix += edgename;
                    } else if (video) {
                        video_edge = manager_->edges()->template find<av::VideoFrame>(edgename);
                        video = false;
                        prefix += " - " + edgename + ": ";
                    } else {
                        throw Error("Invalid syncmeters specification: each meter must have 2 edges");
                    }
                }
                if (!audio_edge || !video_edge) {
                    throw Error("Not enough edges in syncmeter specification");
                }
                sync_meters_.emplace_back(audio_edge, video_edge, prefix);
            }
        }
        #endif
        json jstreams = params.at("streams");
        parseStreams(jstreams, "video");
        parseStreams(jstreams, "audio");
    }
    void mainloop() {
        auto gtod_ms = []() -> AVTS {
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            return tv.tv_sec * 1000 + tv.tv_usec / 1000;
        };
        AVTS next_send = gtod_ms() + interval_ms_;
        AVTS remainder = next_send % interval_ms_;
        next_send -= remainder;
        next_send += interval_ms_ / 10;
        while(true) {
            wallclock.sleepms(next_send - gtod_ms());
            next_send += interval_ms_;
            try {
                send();
            } catch (std::exception &e) {
                logstream << "Error in stats sender: " << e.what();
            }
        }
    }
    void frameNow() {
        last_frame_rtc_ = wallclock.ts();
    }
};

StatsSenderThread::StatsSenderThread(json params, std::shared_ptr<NodeManager> manager) {
    auto sender = std::make_shared<StatsSender>(params, manager);
    thr_ = start_thread("stats sender", [sender]() {
        sender->mainloop();
    });
    thr_.detach();
}

template<typename T> StreamStats<T>::StreamStats(std::shared_ptr<NodeManager> manager, json &jobj, StatsSender* sender, const double max_age):
    manager_(manager),
    sender_(sender),
    max_age_(max_age>0 ? max_age : 864000),
    clear_on_send_(max_age<=0) {
    ts_by_rtc_.setMaxAge(max_age_);
    bytes_.setMaxAge(max_age_);

    std::shared_ptr<Edge<av::Packet>> pre_dec_edge = manager_->edges()->template find<av::Packet>(jobj.at("q_pre_dec"));

    pre_dec_edge->addWiretapCallback([this](const av::Packet &pkt) {
        try {
            if (pkt.isComplete() && pkt.dts().isValid()) {
                if (packet_dd_.check(pkt.dts())) {
                    resetHistoryWrapper();
                }
                {
                    std::lock_guard<decltype(pre_dec_mutex_)> lock(pre_dec_mutex_);
                    sender_->frameNow();
                    ts_by_rtc_.pushWallclockNow(pkt.dts());
                    total_bytes_ += pkt.size();
                    stream_index_ = pkt.streamIndex();
                    bytes_.push(pkt.dts(), total_bytes_);
                    total_frames_++;
                }
            }
        } catch (std::exception &e) {
            logstream << "Stats collecting error (pre-decoder): " << e.what();
        }
    });

    if (jobj.count("q_post_dec")) {
        dec_node_name_ = jobj.at("decoder");

        std::shared_ptr<Edge<T>> post_dec_edge = manager_->edges()->template find<T>(jobj.at("q_post_dec"));
        post_dec_edge->addWiretapCallback([this](const T &frm) {
            try {
                if (frm.isComplete() && frm.pts().isValid()) {
                    if (frame_dd_.check(frm.pts())) {
                        resetHistoryWrapper();
                    }
                    {
                        std::lock_guard<decltype(post_dec_mutex_)> lock(post_dec_mutex_);
                        processDecodedFrame(frm);
                    }
                }
            } catch (std::exception &e) {
                logstream << "Stats collecting error (post-decoder): " << e.what();
            }
        });
        processes_decoded_frames_ = true;
    } else {
        dec_node_name_ = jobj.at("relay");
        processes_decoded_frames_ = false;
    }
}
