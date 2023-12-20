#include "../node_common.hpp"
#include <obs-module.h>
#include <util/platform.h>

// various parts of this code adapted from OBS source code: deps/media-playback/media-playback/media.c
// Copyright (c) 2017 Hugh Bailey <obs.jim@gmail.com>

static inline enum audio_format convert_sample_format(int f)
{
	switch (f) {
	case AV_SAMPLE_FMT_U8:
		return AUDIO_FORMAT_U8BIT;
	case AV_SAMPLE_FMT_S16:
		return AUDIO_FORMAT_16BIT;
	case AV_SAMPLE_FMT_S32:
		return AUDIO_FORMAT_32BIT;
	case AV_SAMPLE_FMT_FLT:
		return AUDIO_FORMAT_FLOAT;
	case AV_SAMPLE_FMT_U8P:
		return AUDIO_FORMAT_U8BIT_PLANAR;
	case AV_SAMPLE_FMT_S16P:
		return AUDIO_FORMAT_16BIT_PLANAR;
	case AV_SAMPLE_FMT_S32P:
		return AUDIO_FORMAT_32BIT_PLANAR;
	case AV_SAMPLE_FMT_FLTP:
		return AUDIO_FORMAT_FLOAT_PLANAR;
	default:;
	}

	return AUDIO_FORMAT_UNKNOWN;
}

static inline enum speaker_layout convert_speaker_layout(uint8_t channels)
{
	switch (channels) {
	case 0:
		return SPEAKERS_UNKNOWN;
	case 1:
		return SPEAKERS_MONO;
	case 2:
		return SPEAKERS_STEREO;
	case 3:
		return SPEAKERS_2POINT1;
	case 4:
		return SPEAKERS_4POINT0;
	case 5:
		return SPEAKERS_4POINT1;
	case 6:
		return SPEAKERS_5POINT1;
	case 8:
		return SPEAKERS_7POINT1;
	default:
		return SPEAKERS_UNKNOWN;
	}
}

class ObsAudioSink: public NodeSingleInput<av::AudioSamples> {
protected:
    InstanceData& app_instance_;
    struct obs_source_audio obs_frame_ = {0};
public:
    using NodeSingleInput::NodeSingleInput;
    virtual void process() {
        av::AudioSamples frm = this->source_->get();
        if (!frm) return;

        obs_frame_.format = convert_sample_format(frm.sampleFormat().get());

        if (obs_frame_.format==AUDIO_FORMAT_UNKNOWN) {
            logstream << "Audio sample format not supported: " << frm.sampleFormat();
            return;
        }

        obs_frame_.samples_per_sec = frm.sampleRate();
        obs_frame_.speakers = convert_speaker_layout(frm.channelsCount());
        obs_frame_.frames = frm.samplesCount();

        for (size_t i = 0; i < MAX_AV_PLANES; i++) {
            obs_frame_.data[i] = frm.raw()->data[i];
        }

        obs_frame_.timestamp = rescaleTS(frm.pts(), av::Rational(1, 1000000000)).timestamp();
		app_instance_.doWithObsSource([this](obs_source_t *s) {
			obs_source_output_audio(s, &obs_frame_);
		});
    }
	ObsAudioSink(std::unique_ptr<SourceType> &&source, InstanceData& app_instance): NodeSingleInput(std::move(source)), app_instance_(app_instance) {
    }
    static std::shared_ptr<ObsAudioSink> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::AudioSamples>> edge = edges.find<av::AudioSamples>(params["src"]);
        auto r = std::make_shared<ObsAudioSink>(make_unique<EdgeSource<av::AudioSamples>>(edge), nci.instance);
        return r;
    }
};

DECLNODE(obs_audio_sink, ObsAudioSink);
