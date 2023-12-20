#pragma once
#include <avcpp/format.h>
#include <avcpp/formatcontext.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/videorescaler.h>

namespace SyncMeter {

    class IStreamProcessor {
    public:
        virtual bool processPacket(av::Packet &pkt) = 0; // returns whether something interesting happened
        virtual av::Timestamp lastTriggered() = 0;
        virtual std::string streamTextId() = 0;
        virtual ~IStreamProcessor() {
        };
    };

    template<typename Decoder> class StreamProcessor: public IStreamProcessor {
    protected:
        Decoder dec_;
        std::string text_id_;
        av::Timestamp last_triggered_ = {AV_NOPTS_VALUE, {1,1}};
    public:
        StreamProcessor(av::Stream &stream): dec_(stream) {
            char mt = '?';
            if (stream.isVideo()) mt = 'V';
            if (stream.isAudio()) mt = 'A';
            text_id_ = std::to_string(stream.index()) + ':' + mt;
            
            dec_.open(av::Codec());
        }
        StreamProcessor(char media_type) {
            text_id_ = media_type;
        }
        virtual av::Timestamp lastTriggered() {
            return last_triggered_;
        }
        virtual std::string streamTextId() {
            return text_id_;
        }
    };

    class Gray8PixelAccessor {
    protected:
        av::VideoFrame &frm_;
        int linesize_;
        uint8_t* data_;
    public:
        Gray8PixelAccessor(av::VideoFrame &frm): frm_(frm), linesize_(frm.raw()->linesize[0]), data_(frm.raw()->data[0]) {
            if (linesize_<=0) {
                throw std::runtime_error("Frame has invalid linesize");
            }
            if (data_==nullptr) {
                throw std::runtime_error("Frame has null pointer to data");
            }
        }
        uint8_t& pixel(size_t x, size_t y) {
            return data_[linesize_*y + x];
        }
        uint8_t* linePointer(size_t y) {
            return data_ + (linesize_*y);
        }
        size_t lineSize() {
            return linesize_;
        }
    };

    class VideoProcessor: public StreamProcessor<av::VideoDecoderContext> {
    protected:
        av::VideoRescaler rescaler_;
        av::VideoFrame prev_frame_;
        const float x1_ = 0.8; //0.5;
        const float x2_ = 0.886;
        const float y1_ = 0.35;
        const float y2_ = 0.64;
    public:
        VideoProcessor(av::Stream &stream): StreamProcessor(stream) {
        }
        VideoProcessor(): StreamProcessor('V') {
        }
        virtual bool processPacket(av::Packet &pkt) {
            av::VideoFrame frame = dec_.decode(pkt);
            if (!frame) return false;
            
            return processFrame(frame);
        }
        bool processFrame(const av::VideoFrame &frame) {
            bool result = false;
            // convert to grayscale:
            av::VideoFrame gray(av::PixelFormat(AV_PIX_FMT_GRAY8), frame.width(), frame.height());
            rescaler_.rescale(gray, frame, av::throws());
            
            // compare frames:
            if (prev_frame_ && gray && prev_frame_.width() == gray.width() && prev_frame_.height() == gray.height()) {
                Gray8PixelAccessor curr(gray);
                Gray8PixelAccessor prev(prev_frame_);
                int iy1 = int(y1_*(float)gray.height());
                int iy2 = int(y2_*(float)gray.height());
                int ix1 = int(x1_*(float)gray.width());
                int ix2 = int(x2_*(float)gray.width());
                uint64_t diffsum = 0;
                for (int y = iy1; y < iy2; y++) {
                    // get pointers to lines:
                    uint8_t* lc = curr.linePointer(y);
                    uint8_t* lp = prev.linePointer(y);
                    // compare pixels:
                    for (int x = ix1; x < ix2; x++) {
                        diffsum += abs(int(lc[x]) - int(lp[x]));
                    }
                }
                float diff_per_pixel = float(diffsum) / ( (iy2-iy1)*(ix2-ix1) );
                if (diff_per_pixel > 255.0*0.04) { // 4% of pixels changed black->white or white->black
                    result = true;
                    this->last_triggered_ = frame.pts();
                }
            }
            prev_frame_ = gray;
            return result;
        }
    };

    class AudioProcessor: public StreamProcessor<av::AudioDecoderContext> {
    protected:
        std::unique_ptr<av::AudioResampler> conv_;
        int prev_sample_ = 0;
        av::Rational timebase_;
        const av::SampleFormat dst_sample_format_ = AV_SAMPLE_FMT_S16;
        const int64_t dst_ch_layout_ = AV_CH_LAYOUT_MONO;
    public:
        AudioProcessor(av::Stream &stream): StreamProcessor(stream) {
        }
        AudioProcessor(): StreamProcessor('A') {
        }
        virtual bool processPacket(av::Packet &pkt) {
            av::AudioSamples asamp = dec_.decode(pkt);
            if (!asamp) return false;
            return processSamples(asamp);
        }
        bool processSamples(const av::AudioSamples &asamp) {
            if (asamp.samplesCount()==0) return false;
            bool result = false;
            
            // check whether converter is used for the first time, or parameters changed:
            if ( conv_==nullptr || conv_->srcChannelLayout() != asamp.channelsLayout() || conv_->srcSampleFormat() != asamp.sampleFormat() || conv_->srcSampleRate() != asamp.sampleRate() ) {
                // initialize converter which will output mono audio as int16s
                conv_ = make_unique<av::AudioResampler>(dst_ch_layout_, asamp.sampleRate(), dst_sample_format_, asamp.channelsLayout(), asamp.sampleRate(), asamp.sampleFormat());
                timebase_ = av::Rational(1, asamp.sampleRate());
                logstream << "Initialized audio converter. " << asamp.sampleRate() << "Hz, " << asamp.sampleFormat() << ", " << asamp.channelsLayoutString();
            }
            
            conv_->push(asamp);
            
            av::AudioSamples amono(dst_sample_format_, asamp.samplesCount(), dst_ch_layout_, asamp.sampleRate());
            conv_->pop(amono, false);
            assert(!amono.isNull());
            assert(asamp.samplesCount() == amono.samplesCount()); // libswresample shouldn't buffer any data when sample rate conversion is disabled. Make sure it really doesn't.
            amono.setPts(asamp.pts()); // avcpp sets PTS using previous frame PTS & length. It may cause desynchronization, so set it back to original value.
            
            // now the fun part, process audio samples:
            int16_t* samples = (int16_t*)amono.data(0);
            for (int i=0; i<amono.samplesCount(); i++) {
                int sample = abs(int(samples[i])); // since -32768 overflows int16 when abs()ed, we must use at least int32
                if ( (sample > prev_sample_) && (sample > 32768/4) ) { // at least -12dBFS & must be peak
                    av::Timestamp ts = { asamp.pts().timestamp(timebase_) + i, timebase_ };
                    if (this->last_triggered_.isNoPts() || (ts - this->last_triggered_).seconds() > 0.2) { // at least 200ms pause between triggers
                        last_triggered_ = ts;
                        result = true;
                    }
                }
                prev_sample_ = sample;
            }
            
            return result;
        }
    };
    
    class Meter {
    protected:
        std::shared_ptr<Edge<av::AudioSamples>> audio_edge_;
        std::shared_ptr<Edge<av::VideoFrame>> video_edge_;
        AudioProcessor audio_proc_;
        VideoProcessor video_proc_;
        std::mutex busy_;
        int trigctr_;
        std::string prefix_;
    public:
        Meter(std::shared_ptr<Edge<av::AudioSamples>> audio_edge, std::shared_ptr<Edge<av::VideoFrame>> video_edge, const std::string prefix): audio_edge_(audio_edge), video_edge_(video_edge), prefix_(prefix) {
            audio_edge_->addWiretapCallback([this](const av::AudioSamples samples) {
                if (audio_proc_.processSamples(samples)) {
                    triggered();
                }
            });
            video_edge_->addWiretapCallback([this](const av::VideoFrame frame) {
                if (video_proc_.processFrame(frame)) {
                    triggered();
                }
            });
        }
        void triggered() {
            std::unique_lock<decltype(busy_)> lock(busy_);
            trigctr_++;
            if (trigctr_ >= 2) {
                if (audio_proc_.lastTriggered().isNoPts()) return;
                if (video_proc_.lastTriggered().isNoPts()) return;
                double diff = (audio_proc_.lastTriggered() - video_proc_.lastTriggered()).seconds();
                if (abs(diff) > 0.5) return; // misaligned trigger positions, do not reset triggered count right now
                logstream << prefix_ << "A-V = " << diff;
                trigctr_ = 0;
            }
        }
    };
};
