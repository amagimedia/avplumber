#include "node_common.hpp"
#include "../InterruptibleReader.hpp"
extern "C" {
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
}

struct Packet {
    int64_t magic_number;
    uint64_t channels;
    uint64_t samples_per_channel;
    uint64_t sample_rate;
    uint64_t bytes_per_sample; // sample is always signed int, 2 = 16bit, 4 = 32bit
    struct timespec timestamp; // 1/1e9 timebase
};

class IPCAudioSource: public NodeSingleOutput<av::AudioSamples>, public ReportsFinishByFlag, public IStoppable {
protected:
    InterruptibleReader pipe_reader_;
    static constexpr av::SampleFormat::Alignment align_ = av::SampleFormat::Alignment::AlignDefault;
    size_t samples_to_skip_ = 0;
    size_t sre_analyze_samples_ = 0;
    size_t sre_samples_count_ = 0;
    av::Timestamp sre_first_pts_ = NOTS;
    int64_t sample_rate_override_ = -1;
public:
    using NodeSingleOutput::NodeSingleOutput;
    virtual void process() {
        Packet packet;
        if (!pipe_reader_.read(&packet, sizeof(Packet))) {
            logstream << "failed to read packet from pipe (maybe node was stopped)";
            wallclock.sleepms(500);
            return;
        }
        if (packet.magic_number != 0x12345678abcdeffe) {
            logstream << "invalid magic number, ignoring packet";
            return;
        }
        av::SampleFormat sample_format(AV_SAMPLE_FMT_NONE);
        if (packet.bytes_per_sample==4) {
            sample_format = AV_SAMPLE_FMT_S32;
        } else if (packet.bytes_per_sample==2) {
            sample_format = AV_SAMPLE_FMT_S16;
        } else {
            logstream << "unsupported bytes per sample " << packet.bytes_per_sample;
            return;
        }
        int64_t channel_layout = av_get_default_channel_layout(packet.channels);
        av::AudioSamples outfrm(sample_format, packet.samples_per_channel, channel_layout,
            sample_rate_override_>0 ? sample_rate_override_ : packet.sample_rate, align_);

        size_t size = sample_format.requiredBufferSize(packet.channels, packet.samples_per_channel, align_);
        uint8_t* ptr = outfrm.data(0);
        //logstream << "reading " << packet.channels << " channels of " << packet.samples_per_channel << " samples of " << packet.bytes_per_sample << " = " << size << " bytes";
        if (!pipe_reader_.read(ptr, size)) {
            logstream << "failed to read audio payload from pipe (maybe node was stopped)";
            return;
        }

        if (samples_to_skip_ > packet.samples_per_channel) {
            samples_to_skip_ -= packet.samples_per_channel;
            return;
        } else if (samples_to_skip_ > 0) {
            logstream << "skipped starting samples";
            samples_to_skip_ = 0;
            return;
        }

        outfrm.setTimeBase({1, 1000000});
        outfrm.raw()->pts = packet.timestamp.tv_sec * 1000000 + packet.timestamp.tv_nsec / 1000;

        // DIRTY HACK for imprecise sample rate input
        // TODO: should be handled in resample_audio.cpp
        if (sre_analyze_samples_ > 0) {
            if (sre_first_pts_.isNoPts()) {
                sre_first_pts_ = outfrm.pts();
            }
            if (sre_samples_count_ >= sre_analyze_samples_) {
                sre_analyze_samples_ = 0;
                av::Timestamp ts_diff = addTS(outfrm.pts(), negateTS(sre_first_pts_));
                sample_rate_override_ = int64_t(double(sre_samples_count_) / double(ts_diff.seconds()) + 0.5);
                logstream << "estimated sample rate = " << sample_rate_override_;
                if (sample_rate_override_ > 47760) {
                    sample_rate_override_ = 48000;
                }
                logstream << "set sample rate override to " << sample_rate_override_;
            }
            sre_samples_count_ += packet.samples_per_channel;
            return;
        }

        outfrm.setComplete(true);
        this->sink_->put(outfrm);
    }
    virtual void stop() {
        pipe_reader_.interrupt();
        this->finished_ = true;
    }
    IPCAudioSource(std::unique_ptr<SinkType> &&sink, const std::string pipe_path):
        NodeSingleOutput(std::move(sink)), pipe_reader_(pipe_path) {
    }
    ~IPCAudioSource() {
    }
    static std::shared_ptr<IPCAudioSource> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::AudioSamples>> edge = edges.find<av::AudioSamples>(params["dst"]);
        auto r = std::make_shared<IPCAudioSource>(make_unique<EdgeSink<av::AudioSamples>>(edge), params["pipe"]);
        if (params.count("sre_skip_samples")) {
            r->samples_to_skip_ = params["sre_skip_samples"];
        }
        if (params.count("sre_analyze_samples")) {
            r->sre_analyze_samples_ = params["sre_analyze_samples"];
        }
        return r;
    }
};

DECLNODE(ipc_audio_source, IPCAudioSource);
