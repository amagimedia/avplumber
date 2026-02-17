#include "node_common.hpp"
#include "../audio_parameters.hpp"
#include <avcpp/audioresampler.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstdint>

#if defined(__SIZEOF_INT128__)
using int128_t = __int128;
#else
// Fallback: use int64_t; segment_index_ * duration_num can overflow for very long streams
using int128_t = int64_t;
#endif

namespace {

static constexpr AVSampleFormat ANALYSIS_SAMPLE_FMT = AV_SAMPLE_FMT_FLTP;

// 0.5 dB resolution, -127..0 dB; byte 0 = -127 dB (or below), 254 = 0 dB, 255 = clip
static uint8_t linearToEnvelopeByte(float linear) {
    if (linear <= 1e-10) return 0; // silence or <= -127 dB
    if (linear > 1.0) return 255; // clip
    float db = 20.0 * std::log10(linear);
    int b = static_cast<int>(std::floor((db + 127.0) * 2.0));
    return static_cast<uint8_t>(std::max(0, std::min(255, b)));
}

struct ChannelAccum {
    float pos_peak = 0;
    float neg_peak = 0;
    float sum_squares = 0;
    size_t count = 0;
    void add(float normalized_sample) {
        if (normalized_sample > pos_peak) pos_peak = normalized_sample;
        if (normalized_sample < neg_peak) neg_peak = normalized_sample;
        sum_squares += normalized_sample * normalized_sample;
        count++;
    }
    void reset() {
        pos_peak = 0;
        neg_peak = 0;
        sum_squares = 0;
        count = 0;
    }
    float rms() const {
        return count > 0 ? std::sqrt(sum_squares / (float)count) : 0.0;
    }
};

struct LevelState {
    av::Rational duration_sec_;
    av::Rational duration_ts_;     // segment duration in (1, sample_rate) units; set when sample_rate known
    std::string duration_str_;
    std::string filename_;
    std::ofstream file_;
    int64_t segment_index_ = 0;    // current segment index; boundaries computed from start_pts + index * duration_ts_
    std::vector<ChannelAccum> accum_;
    bool started_ = false;
};

} // namespace

class WriteAudioEnvelope: public NodeSingleInput<av::AudioSamples> {
    std::string path_;
    std::vector<LevelState> levels_;
    std::unique_ptr<av::AudioResampler> resampler_;
    AudioParameters audio_params_;
    size_t channels_num_ = 0;
    bool start_pts_set_ = false;
    int64_t start_pts_ts_ = 0; // first PTS in (1, sample_rate) units
    bool index_written_ = false;

    av::Rational audioTimebase() const { return av::Rational(1, audio_params_.sample_rate); }

    void writeEnvelopeSample(LevelState& level, std::vector<uint8_t>& out) {
        out.clear();
        out.reserve(3 * level.accum_.size());
        for (const auto& a : level.accum_) {
            out.push_back(linearToEnvelopeByte(a.pos_peak));
            out.push_back(linearToEnvelopeByte(-a.neg_peak)); // negative peak as positive dB
            out.push_back(linearToEnvelopeByte(a.rms()));
        }
        level.file_.write(reinterpret_cast<const char*>(out.data()), out.size());
        level.file_.flush();
    }

    void flushSegment(LevelState& level, std::vector<uint8_t>& out) {
        writeEnvelopeSample(level, out);
        for (auto& a : level.accum_) a.reset();
        level.segment_index_++;
    }

    // Segment boundaries from rational: no accumulation, no drift. Use 128-bit intermediate to avoid overflow for large timestamps/segment_index.
    static int64_t clampToInt64(int128_t x) {
#if defined(__SIZEOF_INT128__)
        if (x > INT64_MAX) return INT64_MAX;
        if (x < INT64_MIN) return INT64_MIN;
#endif
        return static_cast<int64_t>(x);
    }
    int64_t segmentStartTs(const LevelState& level) const {
        int64_t num = level.duration_ts_.getNumerator();
        int64_t den = level.duration_ts_.getDenominator();
        int128_t offset = (int128_t)level.segment_index_ * (int128_t)num;
        offset /= den;
        return clampToInt64((int128_t)start_pts_ts_ + offset);
    }
    int64_t segmentEndTs(const LevelState& level) const {
        int64_t num = level.duration_ts_.getNumerator();
        int64_t den = level.duration_ts_.getDenominator();
        int128_t offset = ((int128_t)level.segment_index_ + 1) * (int128_t)num;
        offset /= den;
        return clampToInt64((int128_t)start_pts_ts_ + offset);
    }

    void ensureIndexWritten() {
        if (index_written_ || channels_num_ == 0) return;
        std::string index_path = path_;
        if (!index_path.empty() && index_path.back() != '/') index_path += '/';
        index_path += "index.json";
        nlohmann::json j;
        j["version"] = 1;
        j["sample_rate"] = audio_params_.sample_rate;
        j["channels"] = channels_num_;
        j["levels"] = nlohmann::json::object();
        for (size_t i = 0; i < levels_.size(); i++)
            j["levels"][levels_[i].duration_str_] = {{"file", levels_[i].filename_}};
        j["metrics"] = nlohmann::json::object();
        j["metrics"]["positive_peak"] = {{"offset_bytes", 0}, {"stride_bytes", 3}};
        j["metrics"]["negative_peak"] = {{"offset_bytes", 1}, {"stride_bytes", 3}};
        j["metrics"]["rms"] = {{"offset_bytes", 2}, {"stride_bytes", 3}};
        std::ofstream f(index_path);
        if (f) {
            f << j.dump(2);
            index_written_ = true;
        }
    }

    void processConvertedSamples(const av::AudioSamples& samples, int64_t buf_start_ts, int64_t buf_end_ts) {
        const size_t nch = samples.channelsCount();
        const size_t nsamples = samples.samplesCount();
        const float* planes[32];
        for (size_t ch = 0; ch < nch; ch++)
            planes[ch] = reinterpret_cast<const float*>(samples.data(ch));

        std::vector<uint8_t> out;

        for (auto& level : levels_) {
            if (!level.file_.is_open() || level.duration_ts_.getNumerator() <= 0 || level.duration_ts_.getDenominator() <= 0) continue;
            int64_t rem_start = buf_start_ts;
            int64_t rem_end = buf_end_ts;
            while (rem_start < rem_end) {
                int64_t seg_start_ts = segmentStartTs(level);
                int64_t seg_end_ts = segmentEndTs(level);
                if (rem_start >= seg_end_ts) {
                    flushSegment(level, out);
                    continue;
                }
                int64_t chunk_start_ts = std::max(rem_start, seg_start_ts);
                int64_t chunk_end_ts = std::min(rem_end, seg_end_ts);
                int64_t start_sample_64 = chunk_start_ts - buf_start_ts;
                int64_t end_sample_64 = chunk_end_ts - buf_start_ts;
                start_sample_64 = std::max(int64_t(0), std::min(start_sample_64, (int64_t)nsamples));
                end_sample_64 = std::max(int64_t(0), std::min(end_sample_64, (int64_t)nsamples));
                if (start_sample_64 >= end_sample_64) {
                    rem_start = chunk_end_ts;
                    continue;
                }
                size_t start_sample = static_cast<size_t>(start_sample_64);
                size_t end_sample = static_cast<size_t>(end_sample_64);
                for (size_t ch = 0; ch < nch; ch++) {
                    for (size_t i = start_sample; i < end_sample; i++)
                        level.accum_[ch].add(planes[ch][i]);
                }
                if (chunk_end_ts >= seg_end_ts)
                    flushSegment(level, out);
                rem_start = chunk_end_ts;
            }
        }
    }

    void openLevelFiles() {
        std::string dir = path_;
        if (!dir.empty() && dir.back() != '/') dir += '/';
        for (size_t i = 0; i < levels_.size(); i++) {
            std::string full = dir + levels_[i].filename_;
            levels_[i].file_.open(full, std::ios::binary | std::ios::app);
            if (!levels_[i].file_) {
                logstream << "write_audio_envelope: failed to open " << full;
            }
        }
    }

public:
    using NodeSingleInput<av::AudioSamples>::NodeSingleInput;

    virtual void process() override {
        av::AudioSamples frame = this->source_->get();
        if (!frame.isComplete() || frame.samplesCount() == 0) return;
        if (!frame.pts().isValid()) return;

        if (resampler_ == nullptr || audio_params_ != AudioParameters(frame)) {
            for (auto& level : levels_) {
                if (level.file_.is_open()) level.file_.close();
            }
            audio_params_ = AudioParameters(frame);
            resampler_ = std::make_unique<av::AudioResampler>(
                audio_params_.channel_layout, audio_params_.sample_rate, av::SampleFormat(ANALYSIS_SAMPLE_FMT),
                audio_params_.channel_layout, audio_params_.sample_rate, audio_params_.sample_format);
            channels_num_ = frame.channelsCount();
            start_pts_set_ = false;
            index_written_ = false;
            for (auto& level : levels_) {
                level.accum_.resize(channels_num_);
                level.started_ = false;
            }
            openLevelFiles();
            logstream << "write_audio_envelope: started " << audio_params_;
        }

        av::Rational tb = audioTimebase();
        if (!start_pts_set_) {
            start_pts_ts_ = rescaleTS(frame.pts(), tb).timestamp();
            start_pts_set_ = true;
            const int64_t sr = audio_params_.sample_rate;
            for (auto& level : levels_) {
                level.segment_index_ = 0;
                int64_t dur_num = (int64_t)level.duration_sec_.getNumerator() * sr;
                int64_t dur_den = level.duration_sec_.getDenominator();
                int rnum = 0, rden = 0;
                if (av_reduce(&rnum, &rden, dur_num, dur_den, INT_MAX))
                    level.duration_ts_ = av::Rational(rnum, rden);
                else
                    level.duration_ts_ = av::Rational(static_cast<int>(dur_num), static_cast<int>(dur_den));
                for (auto& a : level.accum_) a.reset();
            }
        }

        resampler_->push(frame);
        int64_t converted_start_ts = rescaleTS(frame.pts(), tb).timestamp();
        while (true) {
            av::AudioSamples converted(av::SampleFormat(ANALYSIS_SAMPLE_FMT), frame.samplesCount(),
                audio_params_.channel_layout, audio_params_.sample_rate);
            bool got = resampler_->pop(converted, true);
            if (!got) break;
            int128_t end_ts = (int128_t)converted_start_ts + (int128_t)converted.samplesCount();
            int64_t converted_end_ts = clampToInt64(end_ts);
            processConvertedSamples(converted, converted_start_ts, converted_end_ts);
            converted_start_ts = converted_end_ts;
        }

        ensureIndexWritten();
    }

    static std::shared_ptr<WriteAudioEnvelope> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        if (params.count("path") != 1)
            throw Error("write_audio_envelope: path required");
        if (params.count("granularities") != 1 || !params["granularities"].is_array())
            throw Error("write_audio_envelope: granularities must be a list of rational strings");
        std::shared_ptr<Edge<av::AudioSamples>> edge = edges.find<av::AudioSamples>(params["src"]);
        auto r = std::make_shared<WriteAudioEnvelope>(make_unique<EdgeSource<av::AudioSamples>>(edge));
        r->path_ = params["path"].get<std::string>();
        size_t idx = 0;
        for (const auto& g : params["granularities"]) {
            std::string str = g.get<std::string>();
            av::Rational dur = parseRatio(str);
            if (dur.getNumerator() <= 0 || dur.getDenominator() <= 0)
                throw Error("write_audio_envelope: invalid granularity " + str);
            LevelState level;
            level.duration_sec_ = dur;
            level.duration_str_ = str;
            std::ostringstream fn;
            fn << "level_" << idx << ".bin";
            level.filename_ = fn.str();
            level.accum_ = {};
            r->levels_.push_back(std::move(level));
            idx++;
        }
        return r;
    }
};

DECLNODE(write_audio_envelope, WriteAudioEnvelope);
