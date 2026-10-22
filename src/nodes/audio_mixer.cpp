#include "node_common.hpp"
#include "audio_mixer_core.hpp"
#include "../audio_parameters.hpp"
#include "../avutils.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"
#include "../mixer/primitives/TickGrid.hpp"
#include <avcpp/audioresampler.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace {

constexpr const char* kProgramSentinel = "program";
constexpr float kGainEpsilon = 1e-6f;
// A mix thread held up longer than this resumes at the current block instead of catching up.
constexpr int64_t kMaxBehindNs = 1000000000;

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

struct GainEvent {
    int64_t sample = 0;   // the first sample it applies to; 0: the next one mixed
    std::vector<float> gains;
    float program_gain = 0.f;
};

}  // namespace

class AudioMixer: public NodeMultiInput<av::AudioSamples>,
                  public NodeWithOutputs<av::AudioSamples>,
                  public IInputsObjects,
                  public IReturnsObjects,
                  public IAudioMetadataSource,
                  public ITimeBaseSource,
                  public IInitAfterCreate {
    // Clocked by the program's video ticks, not by its inputs. Blocks of frame_samples leave at
    // the deadline of the tick holding their last sample, time(tick) + latency, with that tick's
    // picture; each takes whatever its sources delivered by then and is silence for a source that
    // is late or gone (SampleQueue). Samples are placed by their realtime set_pts media time.
    struct SourceState {
        std::string id;
        AudioParameters last_in;
        std::unique_ptr<av::AudioResampler> resampler;
        std::unique_ptr<avp::audio_mixer::SampleQueue> queue;   // from the first block on
        std::vector<std::vector<float>> block;                   // [ch], the block being mixed
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
    int64_t fps_num_ = 25, fps_den_ = 1;
    std::optional<avp::mixer::TickGrid> grid_;
    int64_t latency_ns_ = 0;
    int64_t next_block_ = -1;   // index of the next block to mix, in frame_samples; -1: not started
    std::vector<SourceState> sources_;
    std::unordered_map<std::string, size_t> source_index_;
    std::vector<Bus> buses_;
    std::unordered_map<std::string, size_t> bus_index_;
    std::vector<std::vector<float>> program_frame_;
    uint64_t blocks_ = 0, skipped_blocks_ = 0, output_drops_ = 0;
    std::mutex mutex_;

    avp::audio_mixer::SampleQueue newQueue(int64_t first) const {
        const int64_t ms = sample_rate_ / 1000;
        // Timestamp rounding within 1 ms continues a frame; gaps fade over 3 ms; at most 1 s held.
        return avp::audio_mixer::SampleQueue(size_t(channels_), first, ms, 3 * ms, sample_rate_);
    }

    int64_t blockDeadline(int64_t block) const {
        const int64_t last = (block + 1) * int64_t(frame_samples_) - 1;
        return grid_->time(avp::audio_mixer::sampleTick(last, sample_rate_, fps_num_, fps_den_)) + latency_ns_;
    }

    /// The first block not due at `now`: blocks ending in a tick whose deadline has passed are.
    int64_t firstBlockAfter(int64_t now) const {
        const int64_t due_tick = grid_->atOrBefore(now - latency_ns_);
        return avp::audio_mixer::tickSample(due_tick + 1, sample_rate_, fps_num_, fps_den_) / int64_t(frame_samples_);
    }

    void ingest(SourceState& src, const av::AudioSamples& in) {
        if (!in.isComplete() || in.samplesCount() <= 0) return;
        if (in.sampleFormat().get() != AV_SAMPLE_FMT_FLTP) {
            logstream << "audio_mixer: ignoring non-fltp input on " << src.id;
            return;
        }
        // A frame without a timestamp continues the previous one.
        const int64_t first = in.pts().isValid()
            ? avp::audio_mixer::sampleAt(in.pts().timestamp({1, 1000000000}), sample_rate_) : src.queue->end();
        AudioParameters cur(in);
        const bool need_swr = cur.sample_rate != sample_rate_ || cur.channel_layout != int64_t(channel_layout_)
            || cur.sample_format.get() != AV_SAMPLE_FMT_FLTP;
        std::vector<std::vector<float>> planes(static_cast<size_t>(channels_));
        if (!need_swr) {
            for (int ch = 0; ch < channels_; ++ch) {
                const float* sp = reinterpret_cast<const float*>(in.data(ch));
                planes[size_t(ch)].assign(sp, sp + in.samplesCount());
            }
            src.queue->write(first, planes);
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
        while (true) {
            av::AudioSamples conv(av::SampleFormat(AV_SAMPLE_FMT_FLTP), 4096, channel_layout_, sample_rate_);
            if (!src.resampler->pop(conv, false) || conv.samplesCount() <= 0) break;
            for (int ch = 0; ch < channels_; ++ch) {
                const float* sp = reinterpret_cast<const float*>(conv.data(ch));
                planes[size_t(ch)].insert(planes[size_t(ch)].end(), sp, sp + conv.samplesCount());
            }
        }
        src.queue->write(first, planes);
    }

    void applyEvents(Bus& bus, int64_t sample) {
        while (!bus.pending.empty() && bus.pending.front().sample <= sample) {
            const GainEvent& e = bus.pending.front();
            bus.gains = e.gains;
            bus.program_gain = e.program_gain;
            bus.pending.pop_front();
        }
    }

    void insertEvent(Bus& bus, GainEvent ev) {
        if (ev.sample == 0) {
            bus.gains = ev.gains;
            bus.program_gain = ev.program_gain;
            return;
        }
        auto it = bus.pending.begin();
        while (it != bus.pending.end() && it->sample <= ev.sample) ++it;
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

    av::AudioSamples mixBus(Bus& bus, bool is_program, int64_t first) {
        av::AudioSamples out(av::SampleFormat(AV_SAMPLE_FMT_FLTP), int(frame_samples_),
                             channel_layout_, sample_rate_);
        out.setTimeBase({1, sample_rate_});
        out.setPts({first, {1, sample_rate_}});
        out.setComplete(true);
        for (int ch = 0; ch < channels_; ++ch) {
            float* dp = reinterpret_cast<float*>(out.data(ch));
            std::fill(dp, dp + frame_samples_, 0.f);
        }
        size_t begin = 0;
        while (begin < frame_samples_) {
            // Every event at or before this sample is applied, so the next one is after it.
            applyEvents(bus, first + int64_t(begin));
            const size_t end = bus.pending.empty() ? frame_samples_
                : size_t(std::min<int64_t>(bus.pending.front().sample - first, int64_t(frame_samples_)));
            for (size_t si = 0; si < sources_.size(); ++si) {
                const float g = bus.gains[si];
                if (g < kGainEpsilon) continue;
                for (int ch = 0; ch < channels_; ++ch) {
                    float* dp = reinterpret_cast<float*>(out.data(ch));
                    avp::audio_mixer::addSamples(dp, sources_[si].block[size_t(ch)].data(), begin, end, g);
                }
            }
            if (!is_program && bus.program_gain >= kGainEpsilon) {
                for (int ch = 0; ch < channels_; ++ch) {
                    float* dp = reinterpret_cast<float*>(out.data(ch));
                    avp::audio_mixer::addSamples(dp, program_frame_[size_t(ch)].data(), begin, end, bus.program_gain);
                }
            }
            begin = end;
        }
        if (is_program) {
            for (int ch = 0; ch < channels_; ++ch) {
                const float* sp = reinterpret_cast<const float*>(out.data(ch));
                std::copy(sp, sp + frame_samples_, program_frame_[size_t(ch)].begin());
            }
        }
        return out;
    }

    void mixBlock(int64_t block) {
        const int64_t first = block * int64_t(frame_samples_);
        for (auto& src : sources_) src.queue->read(frame_samples_, src.block);
        program_frame_.assign(size_t(channels_), std::vector<float>(frame_samples_, 0.f));
        size_t program_bi = SIZE_MAX;
        auto pit = bus_index_.find(kProgramSentinel);
        if (pit != bus_index_.end()) program_bi = pit->second;
        std::vector<av::AudioSamples> frames(buses_.size());
        if (program_bi != SIZE_MAX)
            frames[program_bi] = mixBus(buses_[program_bi], true, first);
        for (size_t bi = 0; bi < buses_.size(); ++bi)
            if (bi != program_bi) frames[bi] = mixBus(buses_[bi], false, first);
        // Never waits for a consumer: a full output edge loses this block.
        for (size_t bi = 0; bi < frames.size(); ++bi)
            if (!buses_[bi].sink->put(frames[bi], true)) ++output_drops_;
        ++blocks_;
    }

    void mixDue(int64_t now) {
        if (next_block_ < 0) {
            next_block_ = firstBlockAfter(now);
            for (auto& src : sources_)
                src.queue = make_unique<avp::audio_mixer::SampleQueue>(newQueue(next_block_ * int64_t(frame_samples_)));
            return;
        }
        if (now - blockDeadline(next_block_) > kMaxBehindNs) {
            const int64_t resume = firstBlockAfter(now);
            skipped_blocks_ += uint64_t(resume - next_block_);
            next_block_ = resume;
            for (auto& src : sources_) src.queue->skipTo(resume * int64_t(frame_samples_));
        }
        while (now >= blockDeadline(next_block_)) mixBlock(next_block_++);
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
        ev.sample = pts_ns ? avp::audio_mixer::sampleAt(pts_ns, sample_rate_) : 0;
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

    virtual Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("audio_mixer: unknown object " + key);
        std::lock_guard<std::mutex> lock(mutex_);
        Parameters sources = Parameters::array();
        for (const auto& src : sources_) {
            Parameters s = {{"id", src.id}};
            if (src.queue) {
                const auto& c = src.queue->counts();
                s["dropped"] = c.dropped;
                s["missing"] = c.missing;
                s["jumps"] = c.jumps;
            }
            sources.push_back(s);
        }
        return {{"blocks", blocks_}, {"skipped_blocks", skipped_blocks_},
                {"output_drops", output_drops_}, {"sources", sources}};
    }

    virtual void process() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            mixDue(avp::mixer::monotonicNs());
            for (size_t i = 0; i < source_edges_.size(); ++i) {
                while (av::AudioSamples* p = source_edges_[i]->peek()) {
                    if (!isEofMarker(*p)) ingest(sources_[i], *p);
                    source_edges_[i]->pop();
                }
            }
            mixDue(avp::mixer::monotonicNs());
        }
        // Sleeps until the next block is due or a source delivers.
        waitForInput(avp::mixer::waitMilliseconds(blockDeadline(next_block_), avp::mixer::monotonicNs()));
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

        // The program's video rate and playout latency: a block leaves with its tick's picture.
        if (!params.count("fps") || !params.count("latency_ms"))
            throw Error("audio_mixer: fps and latency_ms (the program compositor's) required");
        const av::Rational fps = parseRatio(params["fps"].get<std::string>());
        node->fps_num_ = fps.getNumerator();
        node->fps_den_ = fps.getDenominator();
        node->grid_.emplace(fps);
        node->latency_ns_ = int64_t(std::llround(params["latency_ms"].get<double>() * 1e6));
        // A block is complete once the tick holding its last sample has passed.
        if (node->latency_ns_ <= node->grid_->time(1))
            throw Error("audio_mixer: latency_ms must be longer than one video frame");

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

DECLNODE(audio_mixer, AudioMixer)
