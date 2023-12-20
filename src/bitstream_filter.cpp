#include <avcpp/packet.h>
#include <libavcodec/avcodec.h>
#include <string>

class BitStreamFilter {
private:
    const AVBitStreamFilter *bsf;
    AVBSFContext *ctx;
    std::string m_name;
    AVCodecParameters *codecpar;
    bool extradata_updated = false;
public:
    BitStreamFilter(const char* name) {
        m_name = name;
        bsf = av_bsf_get_by_name(name);
        if (bsf==nullptr) {
            throw Error(std::string("Couldn't find BSF ") + name);
        }
        int ret = av_bsf_alloc(bsf, &ctx);
        if (ret < 0) {
            throw Error("Couldn't create BSF context: " + std::to_string(ret));
        }
    }
    BitStreamFilter(const std::string name): BitStreamFilter(name.c_str()) {
    }
    void init(av::Stream &in_stream, av::Stream &out_stream, const bool reset_out) {
        ctx->time_base_in = in_stream.timeBase().getValue();
        int ret = avcodec_parameters_copy(ctx->par_in, in_stream.raw()->codecpar);
        
        //ctx->par_in->codec_tag = stream_out.raw()->codecpar->codec_tag;
        if (ret < 0) {
            throw Error("Couldn't copy input parameters to BSF");
        }
        ret = av_bsf_init(ctx);
        if (ret < 0) {
            throw Error("Couldn't initialize BSF context: " + std::to_string(ret));
        }
        codecpar = out_stream.raw()->codecpar;
        //copyOutputParametersTo(codecpar);
        avcodec_parameters_copy(codecpar, ctx->par_out);
        out_stream.setTimeBase(ctx->time_base_out);
        if (reset_out) codecpar->codec_tag = 0;
    }
    AVBSFContext* raw() {
        return ctx;
    }
    std::string name() const {
        return m_name;
    }
    av::Rational outputTimeBase() const {
        return ctx->time_base_out;
    }
    /*void copyOutputParametersTo(AVCodecParameters *dest_codecpar) {
        avcodec_parameters_copy(dest_codecpar, ctx->par_out);
    }*/
    void send(av::Packet packet) {
        packet.setTimeBase(ctx->time_base_in);
        int ret = av_bsf_send_packet(ctx, packet.raw());
        if (ret < 0) {
            throw Error("Couldn't send packet to BSF: " + std::to_string(ret));
        }
    }
    void flush() {
        av_bsf_send_packet(ctx, nullptr);
    }
    bool receive(av::Packet &destination) {
        //AVPacket pkt;
        av::Packet tmpdest;
        int ret = av_bsf_receive_packet(ctx, tmpdest.raw());
        if ( ret >= 0 ) {
            tmpdest.setComplete(true);
            destination = tmpdest.clone(); // HACK: workaround for aac_adtstoasc which seems to work incorrectly with refcounted frames
            destination.setTimeBase(ctx->time_base_out);
            if (!extradata_updated) { // HACK to make aac_adtstoasc work because it updates parameters after first frame, see ffmpeg sources
                avcodec_parameters_copy(codecpar, ctx->par_out);
                extradata_updated = true;
            }
        }
        if ( (ret == AVERROR(EAGAIN)) || (ret == AVERROR_EOF) ) {
            return false;
        } else if (ret < 0) {
            throw Error("Couldn't receive packet from BSF: " + std::to_string(ret));
        } else {
            return true;
        }
    }
};
