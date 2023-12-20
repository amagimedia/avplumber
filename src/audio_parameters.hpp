#pragma once
#include <avcpp/sampleformat.h>
#include <avcpp/rational.h>
#include <avcpp/frame.h>

struct AudioParameters {
    int64_t channel_layout = -1;
    int sample_rate = -1;
    av::SampleFormat sample_format{AV_SAMPLE_FMT_NONE};
    bool operator==(const AudioParameters &other) {
        return (channel_layout==other.channel_layout) && (sample_rate==other.sample_rate) && (sample_format==other.sample_format);
    }
    bool operator!=(const AudioParameters &other) {
        return !( (*this)==other );
    }
    AudioParameters() {
    }
    av::Rational timebase() const {
        return { 1, sample_rate };
    }
    AudioParameters(const av::AudioSamples &samples):
        channel_layout(samples.channelsLayout()),
        sample_rate(samples.sampleRate()),
        sample_format(samples.sampleFormat()) {
    }
    friend std::ostream& operator<< (std::ostream &stream, const AudioParameters &params) {
        if (params.isValid()) {
            char chlayout[64];
            av_get_channel_layout_string(chlayout, 63, -1, params.channel_layout);
            stream << chlayout << ',' << params.sample_format << ',' << params.sample_rate << "Hz";
        } else {
            stream << "NO_AUDIO";
        }
        return stream;
    }
    bool isValid() const {
        return (channel_layout != -1) && (sample_rate != -1);
    }
};
