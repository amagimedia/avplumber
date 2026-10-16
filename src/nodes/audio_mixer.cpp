#include "node_common.hpp"
#include "audio_mixer_core.hpp"
#include "../audio_parameters.hpp"
#include "../avutils.hpp"
#include <avcpp/audioresampler.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kProgramSentinel = "program";
constexpr float kGainEpsilon = 1e-6f;

int channelCount(uint64_t layout) {
#if API_NEW_CHANNEL_LAYOUT
    AVChannelLayout chl{};
    av_channel_layout_from_mask(&chl, layout);
    int n = chl.nb_channels;
    av_channel_layout_uninit(&chl);
    return n > 0 ? n : 0;
#else
    return av_get_channel_layout_nb_channels(layout);
#endif
}

int64_t sampleToPts(int64_t sample, int sample_rate) {
    return (sample * 1000000000LL) / int64_t(sample_rate);
}

struct GainEvent {
    int64_t pts_ns = 0;
    std::vector<float> gains;
    float program_gain = 0.f;
};

}  // namespace

class AudioMixer: public NodeMultiInput<av::AudioSamples>,
                  public NodeWithOutputs<av::AudioSamples>,
                  public IInputsObjects,
                  public IAudioMetadataSource,
                  public ITimeBaseSource,
                  public IInitAfterCreate {
    // FIFO per source: wait for every live input to have a full block, then mix.
    // AFV hard-sync uses each input's media PTS head (realtime set_pts), not the
    // FIFO write cursor.
    struct SourceState {
        std::string id;
        std::vector<std::deque<float>> lanes;  // [ch]
        AudioParameters last_in;
        std::unique_ptr<av::AudioResampler> resampler;
        bool head_pts_valid = false;
        int64_t head_pts_ns = 0;
    };

    struct Bus {
        std::string id;
        std::shared_ptr<Edge<av::AudioSamples>> edge;
        std::unique_ptr<Sink<av::AudioSamples>> sink;
        std::vector<float> gains;
        float program_gain = 0.f;
        std::deque<GainEvent> pending;
    };

    int sample_rate_ = 48000;
    uint64_t channel_layout_ = 0;
    int channels_ = 2;
    size_t frame_samples_ = 960;
    std::vector<SourceState> sources_;
    std::unordered_map<std::string, size_t> source_index_;
    std::vector<Bus> buses_;
    std::unordered_map<std::string, size_t> bus_index_;
    std::vector<std::vector<float>> program_frame_;
    int64_t mix_sample_ = 0;
    std::mutex mutex_;

    size_t bufferedSamples(const SourceState& src) const {
        if (src.lanes.empty()) return 0;
        return src.lanes[0].size();
    }

    bool inputHasFullBlock(size_t i) const {
        return bufferedSamples(sources_[i]) >= frame_samples_;
    }

    bool allInputsHaveFullBlock() const {
        if (sources_.empty()) return false;
        for (size_t i = 0; i < sources_.size(); ++i) {
            if (!inputHasFullBlock(i)) return false;
        }
        return true;
    }

    // Media time of sample 0 of the block about to mix: max head PTS among inputs.
    bool blockStartPtsNs(int64_t& out) const {
        bool any = false;
        int64_t chosen = 0;
        for (const auto& src : sources_) {
            if (!src.head_pts_valid) continue;
            if (!any || src.head_pts_ns > chosen) {
                chosen = src.head_pts_ns;
                any = true;
            }
        }
        if (!any) return false;
        out = chosen;
        return true;
    }

    void pushPlanar(SourceState& src, const std::vector<std::vector<float>>& planes, int64_t pts_ns) {
        if (planes.empty() || planes[0].empty()) return;
        const size_t n = planes[0].size();
        const size_t was = bufferedSamples(src);
        if (src.lanes.size() != size_t(channels_))
            src.lanes.assign(size_t(channels_), {});
        for (int ch = 0; ch < channels_; ++ch) {
            auto& q = src.lanes[size_t(ch)];
            if (ch < int(planes.size()) && planes[size_t(ch)].size() == n)
                q.insert(q.end(), planes[size_t(ch)].begin(), planes[size_t(ch)].end());
            else if (ch < int(planes.size()) && !planes[size_t(ch)].empty())
                q.insert(q.end(), planes[size_t(ch)].begin(), planes[size_t(ch)].end());
            else
                q.insert(q.end(), n, 0.f);
        }
        if (!src.head_pts_valid || was == 0) {
            src.head_pts_ns = pts_ns;
            src.head_pts_valid = true;
        }
    }

    void ingest(SourceState& src, const av::AudioSamples& in) {
        if (!in.isComplete() || in.samplesCount() <= 0) return;
        if (in.sampleFormat().get() != AV_SAMPLE_FMT_FLTP) {
            logstream << "audio_mixer: ignoring non-fltp input on " << src.id;
            return;
        }
        const int64_t pts_ns = in.pts().isValid()
            ? in.pts().timestamp({1, 1000000000}) : int64_t(0);
        AudioParameters cur(in);
        const bool need_swr = cur.sample_rate != sample_rate_ || cur.channel_layout != int64_t(channel_layout_)
            || cur.sample_format.get() != AV_SAMPLE_FMT_FLTP;
        std::vector<std::vector<float>> planes;
        if (!need_swr) {
            planes.assign(size_t(channels_), {});
            const int in_ch = in.channelsCount();
            for (int ch = 0; ch < channels_; ++ch) {
                planes[size_t(ch)].resize(size_t(in.samplesCount()));
                const float* sp = ch < in_ch ? reinterpret_cast<const float*>(in.data(ch)) : nullptr;
                if (sp)
                    std::copy(sp, sp + in.samplesCount(), planes[size_t(ch)].begin());
                else if (in_ch == 1) {
                    const float* mono = reinterpret_cast<const float*>(in.data(0));
                    std::copy(mono, mono + in.samplesCount(), planes[size_t(ch)].begin());
                } else {
                    std::fill(planes[size_t(ch)].begin(), planes[size_t(ch)].end(), 0.f);
                }
            }
            pushPlanar(src, planes, pts_ns);
            return;
        }
        if (!src.resampler || src.last_in != cur) {
            av::Dictionary opts;
            src.resampler = make_unique<av::AudioResampler>(
                channel_layout_, sample_rate_, av::SampleFormat(AV_SAMPLE_FMT_FLTP),
                cur.channel_layout, cur.sample_rate, cur.sample_format, opts);
            src.last_in = cur;
        }
        src.resampler->push(in);
        planes.assign(size_t(channels_), {});
        size_t total = 0;
        while (true) {
            av::AudioSamples conv(av::SampleFormat(AV_SAMPLE_FMT_FLTP), 4096, channel_layout_, sample_rate_);
            if (!src.resampler->pop(conv, false) || conv.samplesCount() <= 0) break;
            for (int ch = 0; ch < channels_; ++ch) {
                const float* sp = reinterpret_cast<const float*>(conv.data(ch));
                planes[size_t(ch)].insert(planes[size_t(ch)].end(), sp, sp + conv.samplesCount());
            }
            total += size_t(conv.samplesCount());
        }
        if (total) pushPlanar(src, planes, pts_ns);
    }

    void applyEvents(Bus& bus, int64_t pts_ns) {
        while (!bus.pending.empty() && (bus.pending.front().pts_ns == 0 || bus.pending.front().pts_ns <= pts_ns)) {
            const GainEvent& e = bus.pending.front();
            bus.gains = e.gains;
            bus.program_gain = e.program_gain;
            bus.pending.pop_front();
        }
    }

    void insertEvent(Bus& bus, GainEvent ev) {
        if (ev.pts_ns == 0) {
            bus.gains = ev.gains;
            bus.program_gain = ev.program_gain;
            return;
        }
        auto it = bus.pending.begin();
        while (it != bus.pending.end() && it->pts_ns <= ev.pts_ns) ++it;
        bus.pending.insert(it, std::move(ev));
    }

    std::vector<int> mapIds(const Parameters& arr, bool& has_program) const {
        std::vector<int> idx;
        has_program = false;
        if (!arr.is_array()) return idx;
        for (const auto& item : arr) {
            const std::string id = item.get<std::string>();
            if (id == kProgramSentinel) {
                has_program = true;
                continue;
            }
            auto it = source_index_.find(id);
            if (it != source_index_.end())
                idx.push_back(int(it->second));
        }
        return idx;
    }

    void consumeBlockFrom(SourceState& src) {
        for (auto& q : src.lanes) {
            if (q.size() < frame_samples_)
                q.clear();
            else
                q.erase(q.begin(), q.begin() + std::ptrdiff_t(frame_samples_));
        }
        if (src.head_pts_valid)
            src.head_pts_ns += sampleToPts(int64_t(frame_samples_), sample_rate_);
    }

    std::vector<av::AudioSamples> buildFrameLocked() {
        std::vector<av::AudioSamples> frames;
        int64_t block_start_ns = 0;
        const bool have_media_pts = blockStartPtsNs(block_start_ns);
        const int64_t frame_start = mix_sample_;
        program_frame_.assign(size_t(channels_), std::vector<float>(frame_samples_, 0.f));

        // Snapshot one block per source before consume.
        std::vector<std::vector<std::vector<float>>> blocks(sources_.size());
        for (size_t si = 0; si < sources_.size(); ++si) {
            blocks[si].assign(size_t(channels_), std::vector<float>(frame_samples_, 0.f));
            SourceState& src = sources_[si];
            const size_t have = std::min(bufferedSamples(src), frame_samples_);
            for (int ch = 0; ch < channels_; ++ch) {
                auto& q = src.lanes[size_t(ch)];
                std::copy(q.begin(), q.begin() + std::ptrdiff_t(have), blocks[si][size_t(ch)].begin());
            }
        }

        auto mix_one = [&](Bus& bus, bool is_program) -> av::AudioSamples {
            av::AudioSamples out(av::SampleFormat(AV_SAMPLE_FMT_FLTP), int(frame_samples_),
                                 channel_layout_, sample_rate_);
            out.setTimeBase({1, sample_rate_});
            out.setPts({frame_start, {1, sample_rate_}});
            out.setComplete(true);
            for (int ch = 0; ch < channels_; ++ch) {
                float* dp = reinterpret_cast<float*>(out.data(ch));
                std::fill(dp, dp + frame_samples_, 0.f);
            }

            const auto ptsAt = [&](size_t sample) {
                return have_media_pts
                    ? block_start_ns + sampleToPts(int64_t(sample), sample_rate_)
                    : sampleToPts(frame_start + int64_t(sample), sample_rate_);
            };
            size_t begin = 0;
            while (begin < frame_samples_) {
                applyEvents(bus, ptsAt(begin));
                size_t end = frame_samples_;
                if (!bus.pending.empty() && bus.pending.front().pts_ns != 0)
                    end = avp::audio_mixer::eventSampleOffset(
                        bus.pending.front().pts_ns, block_start_ns, frame_start,
                        sample_rate_, frame_samples_, have_media_pts);
                // Rounding at a timestamp boundary must not leave the mixer on
                // the same segment forever.
                if (end <= begin) end = begin + 1;

                for (size_t si = 0; si < sources_.size(); ++si) {
                    const float g = bus.gains[si];
                    if (g < kGainEpsilon) continue;
                    for (int ch = 0; ch < channels_; ++ch) {
                        float* dp = reinterpret_cast<float*>(out.data(ch));
                        const float* sp = blocks[si][size_t(ch)].data();
                        avp::audio_mixer::addSamples(dp, sp, begin, end, g);
                    }
                }
                if (!is_program && bus.program_gain >= kGainEpsilon) {
                    for (int ch = 0; ch < channels_; ++ch) {
                        float* dp = reinterpret_cast<float*>(out.data(ch));
                        const float* sp = program_frame_[size_t(ch)].data();
                        avp::audio_mixer::addSamples(dp, sp, begin, end, bus.program_gain);
                    }
                }
                if (is_program) {
                    for (int ch = 0; ch < channels_; ++ch) {
                        const float* sp = reinterpret_cast<const float*>(out.data(ch));
                        std::copy(sp + begin, sp + end, program_frame_[size_t(ch)].begin() + begin);
                    }
                }
                begin = end;
            }
            return out;
        };

        size_t program_bi = SIZE_MAX;
        auto pit = bus_index_.find("program");
        if (pit != bus_index_.end()) program_bi = pit->second;
        frames.resize(buses_.size());
        if (program_bi != SIZE_MAX)
            frames[program_bi] = mix_one(buses_[program_bi], true);
        for (size_t bi = 0; bi < buses_.size(); ++bi) {
            if (bi == program_bi) continue;
            frames[bi] = mix_one(buses_[bi], false);
        }
        for (auto& src : sources_)
            consumeBlockFrom(src);
        mix_sample_ = frame_start + int64_t(frame_samples_);
        return frames;
    }

public:
    virtual void forEachOutput(std::function<void(Sink<av::AudioSamples>*)> cb) override {
        for (auto& b : buses_) cb(b.sink.get());
    }

    virtual av::Rational timeBase() override {
        return {1, sample_rate_};
    }
    virtual int sampleRate() override { return sample_rate_; }
    virtual av::SampleFormat sampleFormat() override { return av::SampleFormat(AV_SAMPLE_FMT_FLTP); }
    virtual uint64_t channelLayout() override { return channel_layout_; }

    virtual void setObject(const std::string key, const Parameters& params) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (key != "transition_ended" && key != "transition_progress") return;
        if (!params.count("output")) return;
        const std::string output = params["output"].get<std::string>();
        auto bit = bus_index_.find(output);
        if (bit == bus_index_.end()) return;
        Bus& bus = buses_[bit->second];
        const int64_t pts_ns = params.value("pts", int64_t(0));
        const std::string kind = params.value("kind", std::string("cut"));
        GainEvent ev;
        ev.pts_ns = pts_ns;
        ev.gains.assign(sources_.size(), 0.f);
        ev.program_gain = 0.f;
        if (key == "transition_ended") {
            bool has_program = false;
            auto idx = mapIds(params.value("sources", Parameters::array()), has_program);
            for (int i : idx) ev.gains[size_t(i)] = 1.f;
            if (has_program) ev.program_gain = 1.f;
            (void)kind;
        } else {
            bool from_prog = false, to_prog = false;
            auto from_idx = mapIds(params.value("from", Parameters::array()), from_prog);
            auto to_idx = mapIds(params.value("to", Parameters::array()), to_prog);
            const double t = params.value("t", 0.0);
            avp::audio_mixer::gainsForProgress(
                kind, t, from_idx, to_idx, sources_.size(), ev.gains, ev.program_gain,
                from_prog, to_prog);
        }
        insertEvent(bus, std::move(ev));
    }

    virtual void process() override {
        for (size_t i = 0; i < source_edges_.size(); ++i) {
            while (true) {
                av::AudioSamples* p = source_edges_[i]->peek();
                if (!p) break;
                if (isEofMarker(*p)) {
                    source_edges_[i]->pop();
                    continue;
                }
                av::AudioSamples in = *p;
                source_edges_[i]->pop();
                std::lock_guard<std::mutex> lock(mutex_);
                ingest(sources_[i], in);
            }
        }
        while (true) {
            std::vector<av::AudioSamples> frames;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!allInputsHaveFullBlock()) break;
                frames = buildFrameLocked();
            }
            for (size_t bi = 0; bi < frames.size(); ++bi) {
                if (frames[bi].isComplete())
                    buses_[bi].sink->put(frames[bi]);
            }
        }
        waitForInput();
    }

    virtual void init(EdgeManager&, const Parameters&) override {
        for (auto& b : buses_) {
            auto* es = dynamic_cast<EdgeSink<av::AudioSamples>*>(b.sink.get());
            if (es) es->edge()->setProducer(this->shared_from_this());
        }
    }

    static std::shared_ptr<AudioMixer> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto node = std::make_shared<AudioMixer>();

        node->sample_rate_ = params.value("dst_sample_rate", 48000);
        node->frame_samples_ = size_t(std::max(1, params.value("frame_samples", 960)));
        const std::string layout_s = params.value("dst_channel_layout", std::string("stereo"));
        node->channel_layout_ = stringToChannelLayout(layout_s);
        if (!node->channel_layout_)
            throw Error("audio_mixer: invalid dst_channel_layout");
        node->channels_ = channelCount(node->channel_layout_);
        if (node->channels_ <= 0)
            throw Error("audio_mixer: could not resolve channel count");

        if (!params.count("source_ids") || !params["source_ids"].is_array())
            throw Error("audio_mixer: source_ids array required");
        if (!params.count("src"))
            throw Error("audio_mixer: src required");

        node->createSourcesFromParameters(edges, params);
        const auto ids = params["source_ids"];
        if (ids.size() != node->source_edges_.size())
            throw Error("audio_mixer: source_ids length must match src");
        node->sources_.resize(node->source_edges_.size());
        for (size_t i = 0; i < node->sources_.size(); ++i) {
            node->sources_[i].id = ids[i].get<std::string>();
            node->sources_[i].lanes.assign(size_t(node->channels_), {});
            node->source_index_[node->sources_[i].id] = i;
        }

        if (!params.count("outputs") || !params["outputs"].is_object() || params["outputs"].empty())
            throw Error("audio_mixer: outputs object required (e.g. {\"program\": \"edge\"})");
        for (auto it = params["outputs"].begin(); it != params["outputs"].end(); ++it) {
            Bus bus;
            bus.id = it.key();
            const std::string edge_name = it.value().get<std::string>();
            bus.edge = edges.find<av::AudioSamples>(edge_name);
            bus.sink = make_unique<EdgeSink<av::AudioSamples>>(bus.edge);
            bus.gains.assign(node->sources_.size(), 0.f);
            node->bus_index_[bus.id] = node->buses_.size();
            node->buses_.push_back(std::move(bus));
        }
        return node;
    }
};

DECLNODE(audio_mixer, AudioMixer);
