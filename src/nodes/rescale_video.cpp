#include "node_common.hpp"
#include <avcpp/videorescaler.h>
#include "../util.hpp"
#include "../video_parameters.hpp"

class DynamicVideoScaler: public NodeSISO<av::VideoFrame, av::VideoFrame>, public IVideoFormatSource {
protected:
    VideoParameters src_params_, dst_params_;
    std::unique_ptr<av::VideoRescaler> rescaler_;
    //av::Rational timebase_ = {0, 1};
    //av::Rational frame_rate_ = {0, 1};
    int32_t sws_flags_;
    bool sourceChanged(const av::VideoFrame &frame) {
        return src_params_ != VideoParameters(frame);
    }
    void createRescaler() {
        rescaler_ = make_unique<av::VideoRescaler>(dst_params_.width, dst_params_.height, dst_params_.pixel_format, src_params_.width, src_params_.height, src_params_.pixel_format);
    }
public:
    virtual void process() {
        av::VideoFrame in_frame = this->source_->get();
        if (in_frame) {
            //logstream << "scale in: PTS = " << in_frame.pts() << std::endl;
            if (sourceChanged(in_frame)) {
                src_params_ = VideoParameters(in_frame);
                createRescaler();
            }
            //av::VideoFrame out_frame;
            //rescaler_->rescale(out_frame, in_frame);
            av::VideoFrame out_frame = rescaler_->rescale(in_frame, av::throws());
            if (out_frame) {
                //logstream << "scale out: PTS = " << out_frame.pts() << std::endl;
                this->sink_->put(out_frame);
            }
        }/* else {
            flush();
        }*/
    }
    /*virtual void flush() {
        // NOOP, frame rescaler doesn't need flushing
    }*/
    DynamicVideoScaler(std::unique_ptr<Source<av::VideoFrame>> &&source, std::unique_ptr<Sink<av::VideoFrame>> &&sink, const VideoParameters &dst_params, int32_t flags): NodeSISO<av::VideoFrame, av::VideoFrame>(std::move(source), std::move(sink)), dst_params_(dst_params), sws_flags_(flags) {
        // DynamicVideoScaler will generally process everything thrown on it
        // but by setting preferred pix_fmt and resolution we can reduce CPU usage (and console spam from libav warnings ;) )
        std::shared_ptr<IPreferredFormatReceiver> fmt_recv = this->findNodeUp<IPreferredFormatReceiver>();
        if (fmt_recv) {
            fmt_recv->setPreferredPixelFormat(dst_params_.pixel_format);
            fmt_recv->setPreferredResolution(dst_params_.width, dst_params_.height);
        }
    }
    static std::shared_ptr<DynamicVideoScaler> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        VideoParameters dst_params;
        dst_params.width = params.at("dst_width").get<int>();
        dst_params.height = params.at("dst_height").get<int>();
        int32_t flags_i = 0;
        if (params.count("flags")==1) {
            const Parameters &flags_list = params["flags"];
            if (!flags_list.is_array()) {
                throw Error("flags parameter must be a list!");
            }
            for (auto &item: flags_list) {
                if (!item.is_string()) {
                    throw Error("flag parameter must be string!");
                }
                std::string flagstr = item.get<std::string>();
                #define declflag(p) if ( flagstr == #p ) { flags_i |= p; } else
                // list generated from https://github.com/h4tr3d/avcpp/blob/master/src/videorescaler.h
                // using: sed -e 's/^.\+\(SWS_.\+\),$/declflag(\1)/'
                declflag(SWS_FAST_BILINEAR)
                declflag(SWS_BILINEAR)
                declflag(SWS_BICUBIC)
                declflag(SWS_X)
                declflag(SWS_POINT)
                declflag(SWS_AREA)
                declflag(SWS_BICUBLIN)
                declflag(SWS_GAUSS)
                declflag(SWS_SINC)
                declflag(SWS_LANCZOS)
                declflag(SWS_SPLINE)
                declflag(SWS_PRINT_INFO)
                declflag(SWS_ACCURATE_RND)
                declflag(SWS_FULL_CHR_H_INT)
                declflag(SWS_FULL_CHR_H_INP)
                declflag(SWS_BITEXACT)
                declflag(SWS_ERROR_DIFFUSION)
                throw Error("invalid flag");
                #undef declflag
            }
        }
        dst_params.pixel_format = av::PixelFormat(params.at("dst_pixel_format").get<std::string>());
        // FIXME: specifying invalid pixel format causes segfault!
        return NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<DynamicVideoScaler>(edges, params, dst_params, flags_i);
    }
    virtual int width() {
        return dst_params_.width;
    }
    virtual int height() {
        return dst_params_.height;
    }
    virtual av::PixelFormat pixelFormat() {
        return dst_params_.pixel_format;
    }
    /*virtual av::Rational frameRate() {
        if (frame_rate_.getNumerator()==0) {
            throw Error("Error: tried to get frame_rate_ which is unknown!");
        }
        return frame_rate_;
    }*/
    /*virtual av::Rational timeBase() {
        if (timebase_.getNumerator()==0) {
            throw Error("Error: tried to get timebase_ which is unknown!");
        }
        return timebase_;
    }*/
};

DECLNODE(rescale_video, DynamicVideoScaler);
