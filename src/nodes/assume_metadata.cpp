#include "node_common.hpp"

class AssumeAudioFormat: public TransparentNode<av::AudioSamples>, public IAudioMetadataSource, public ITimeBaseSource {
private:
    int sample_rate_;
    av::SampleFormat sample_format_;
    uint64_t channel_layout_;
public:
    virtual int sampleRate() {
        return sample_rate_;
    }
    virtual av::SampleFormat sampleFormat() {
        return sample_format_;
    }
    virtual uint64_t channelLayout() {
        return channel_layout_;
    }
    virtual av::Rational timeBase() {
        return {1, sample_rate_};
    }
    AssumeAudioFormat(std::unique_ptr<Source<av::AudioSamples>> &&source, std::unique_ptr<Sink<av::AudioSamples>> &&sink, const int sample_rate, const av::SampleFormat sample_format, const uint64_t channel_layout):
        TransparentNode<av::AudioSamples>(std::move(source), std::move(sink)),
        sample_rate_(sample_rate), sample_format_(sample_format), channel_layout_(channel_layout) {
    }
    static std::shared_ptr<AssumeAudioFormat> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        int sr = 48000;
        av::SampleFormat fmt(AV_SAMPLE_FMT_S32P);
        uint64_t chl = AV_CH_LAYOUT_STEREO;
        if (params.count("sample_rate")) sr = params["sample_rate"];
        if (params.count("sample_format")) fmt = av::SampleFormat(params["sample_format"].get<std::string>());
        if (params.count("channel_layout")) chl = av_get_channel_layout(params["channel_layout"].get<std::string>().c_str());
        return NodeSISO<av::AudioSamples, av::AudioSamples>::template createCommon<AssumeAudioFormat>(edges, params, sr, fmt, chl);
    }
    virtual ~AssumeAudioFormat() {
    }
};

class AssumeVideoFormat: public TransparentNode<av::VideoFrame>, public IVideoFormatSource {
private:
    int width_, height_;
    av::PixelFormat pix_fmt_;
    av::PixelFormat real_pix_fmt_;
public:
    virtual int width() {
        return width_;
    }
    virtual int height() {
        return height_;
    }
    virtual av::PixelFormat pixelFormat() {
        return pix_fmt_;
    }
    virtual av::PixelFormat realPixelFormat() {
        return real_pix_fmt_;
    }
    AssumeVideoFormat(std::unique_ptr<Source<av::VideoFrame>> &&source, std::unique_ptr<Sink<av::VideoFrame>> &&sink,
        const int width, const int height, const av::PixelFormat pixfmt, const av::PixelFormat real_pixfmt):
        TransparentNode<av::VideoFrame>(std::move(source), std::move(sink)),
        width_(width), height_(height), pix_fmt_(pixfmt), real_pix_fmt_(real_pixfmt) {
    }
    static std::shared_ptr<AssumeVideoFormat> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        int width = 1920;
        int height = 1080;
        av::PixelFormat pixel_format = av::PixelFormat(AV_PIX_FMT_YUV420P);
        av::PixelFormat real_pixel_format = av::PixelFormat(AV_PIX_FMT_YUV420P);
        if (params.count("width")) width = params["width"];
        if (params.count("height")) height = params["height"];
        if (params.count("pixel_format")) pixel_format = av::PixelFormat(params["pixel_format"].get<std::string>());
        if (params.count("real_pixel_format")) real_pixel_format = av::PixelFormat(params["real_pixel_format"].get<std::string>());
        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<AssumeVideoFormat>(edges, params, width, height, pixel_format, real_pixel_format);
    }
};

DECLNODE(assume_audio_format, AssumeAudioFormat);
DECLNODE(assume_video_format, AssumeVideoFormat);

DECLNODE_ALIAS(fake_audio_metadata, AssumeAudioFormat);
DECLNODE_ALIAS(fake_video_format, AssumeVideoFormat);
