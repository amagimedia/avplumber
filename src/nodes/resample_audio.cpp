#include "node_common.hpp"
#include <avcpp/audioresampler.h>
#include <string>
#include "../util.hpp"

#include "../audio_parameters.hpp"

class DynamicAudioResampler: public NodeSISO<av::AudioSamples, av::AudioSamples>, public IFlushable, public ReportsFinishByFlag, public INeedsOutputFrameSize, public ITimeBaseSource {
protected:
    size_t enc_frame_size_ = 0;
    std::unique_ptr<av::AudioResampler> resampler_;
    AudioParameters src_params_, dst_params_;
    bool forward_channels_ = false;
    //TSEqualizer eq_;
    //int out_ts_adjust_ = 0;
    int comp_samp_;
    //av::Timestamp in_ts_ = NOTS;
    av::Timestamp next_out_ts_ = NOTS;
    av::Rational timebase_ = {1, 28224000}; // LCM of 44100, 48000, 192000
    av::Timestamp inside_resampler_ = {0, timebase_};
    DiscontinuityDetector discodet_;
    std::vector<double> drifts_;
    static constexpr size_t drifts_size_ = 50;
    double max_drift_ = 0.001;
    size_t drifted_frames_;
    size_t drift_index_;
    bool prev_drift_negative_ = false;
    bool now_compensating_ = false;
    av::AudioSamples to_out_;
    //bool outputted_ = false;
    //av::Timestamp out_ts_shift_ = { 0, {1,1} };
    bool sourceChanged(const av::AudioSamples &samples) {
        return src_params_ != AudioParameters(samples);
    }
    void createResampler() {
        av::Dictionary opts;
        //av_dict_set_int(opts.rawPtr(), "async", comp_samp_, 0);
        opts["async"].set(std::to_string(comp_samp_));
        opts["dither_method"].set("triangular");
        //opts["min_comp"].set("0.001");
        //opts["min_hard_comp"].set("1.0");
        //opts["comp_duration"].set("1.0");
        //opts["max_soft_comp"].set("10000.0");
        logstream << "Creating resampler with async=" << opts["async"].value();
        if (inside_resampler_.timestamp() != 0) {
            logstream << "Discarding " << inside_resampler_ << " = " << inside_resampler_.seconds() << "s inside resampler";
        }
        inside_resampler_ = {0, timebase_};
        
        if (forward_channels_) {
            dst_params_.channel_layout = src_params_.channel_layout;
        }
        resampler_ = make_unique<av::AudioResampler>(dst_params_.channel_layout, dst_params_.sample_rate, dst_params_.sample_format, src_params_.channel_layout, src_params_.sample_rate, src_params_.sample_format, opts);
    }
    void out(av::AudioSamples &out_samples) {
        if (out_samples.samplesCount()>0) {
            assert(out_samples.sampleRate() == dst_params_.sample_rate);
            inside_resampler_ = addTS(inside_resampler_, { -out_samples.samplesCount(), {1, out_samples.sampleRate()} });
            if (!next_out_ts_.isValid()) {
                throw Error("Trying to output samples without knowing next PTS!");
            }
            if (inside_resampler_.seconds() < -0.008) {
                logstream << "Very strange: Negative number of samples inside resampler: " << inside_resampler_ << " = " << inside_resampler_.seconds() << "s";
                //inside_resampler_ = {0, timebase_};
            }
            out_samples.setTimeBase({1, dst_params_.sample_rate});
            out_samples.setPts(next_out_ts_);
            next_out_ts_ = addTSSameTB(next_out_ts_, av::Timestamp(out_samples.samplesCount(), out_samples.timeBase()) );
            //logstream << "out " << out_samples.samplesCount() << " * " << out_samples.timeBase();
            this->sink_->put(out_samples);
        }
    }
    void flushInternal() {
        if (!resampler_) return;
        drainResampler(true);
        resampler_ = nullptr;
    }
    void drainResampler(bool output_incomplete = false) {
        while(true) {
            // it seems that we need to drain all samples
            // otherwise libswresample will occassionally drop some without any warning in log. cute.
            size_t req_samples = enc_frame_size_ - to_out_.samplesCount();
            av::AudioSamples out_samples(dst_params_.sample_format, req_samples, dst_params_.channel_layout, dst_params_.sample_rate);
            bool has_frame = resampler_->pop(out_samples, true);
            if (has_frame) {
                if (to_out_.samplesCount()>0) {
                    if (to_out_.channelsCount() != out_samples.channelsCount()) {
                        soft_assert(forward_channels_, "BUG: different number of channels in subsequent outputs but forward_channels_==false");
                        // audioConcat will not work, output it in parts ignoring the output_incomplete flag
                        out(to_out_);
                    } else {
                        out_samples = audioConcat(to_out_, out_samples);
                    }
                    to_out_ = av::AudioSamples(nullptr);
                }
                
                if (out_samples.samplesCount()==enc_frame_size_ || output_incomplete) {
                    out(out_samples);
                } else if (out_samples.samplesCount()<enc_frame_size_) {
                    to_out_ = out_samples;
                    break;
                } else {
                    throw Error("Too many samples from resampler: " + std::to_string(out_samples.samplesCount()) + " > " + std::to_string(enc_frame_size_));
                }
            } else {
                break;
            }
        }
    }
public:
    virtual void setOutputFrameSize(const size_t size) {
        if (size>0) {
            enc_frame_size_ = size;
        } else {
            enc_frame_size_ = 1024;
            logstream << "Warning: Audio resampler got invalid frame size " << size << ", setting to " << enc_frame_size_ << std::endl;
        }
    }
    virtual void start() {
        if (enc_frame_size_==0) {
            enc_frame_size_ = 1024;
            logstream << "Notice: Audio resampler must know output frame size! Assuming " << enc_frame_size_;
        }
    }
    virtual void process() {
        av::AudioSamples in_samples = this->source_->get();
        if (in_samples) {
            //logstream << "in " << in_samples.samplesCount() << " * " << in_samples.timeBase();
            bool really_drift = false;
            bool discontinuity = discodet_.check(in_samples.pts());
            if (discontinuity) {
                logstream << "Detected discontinuity: " << next_out_ts_ << " -> " << in_samples.pts();
                next_out_ts_ = NOTS;
            }
            if (sourceChanged(in_samples) || discontinuity) {
                flushInternal();
                AudioParameters prev_params = src_params_;
                src_params_ = AudioParameters(in_samples);

                // Drain the internal audio buffer to prevent -ve number of samples inside re-sampler
                to_out_ = av::AudioSamples(nullptr);
                createResampler();
                drifts_ = std::vector<double>(drifts_size_, 0);
                drift_index_ = 0;
                drifted_frames_ = 0;
                now_compensating_ = false;
                //eq_.reset();
                //out_ts_adjust_ = 0;
                //out_ts_shift_ = { 0, {1,1} };
                av::Timestamp outts = rescaleTS(in_samples.pts(), dst_params_.timebase());
                if (!next_out_ts_.isValid()) {
                    next_out_ts_ = outts;
                } else {
                    double drift = addTS(outts, negateTS(next_out_ts_)).seconds();
                    drifted_frames_ += 50;
                    really_drift = true; // assume that drift occurs when source changes
                    logstream << "Recreated resampler. Input: " << prev_params << " -> " << src_params_ << " Output: " << dst_params_ << ", Drift: " << drift << " s";
                }
            }
            
            // add current drift to averaging table:
            av::Timestamp swr_delay_r = { swr_get_delay(resampler_->raw(), dst_params_.sample_rate), {1, dst_params_.sample_rate} };
            av::Timestamp cur_drift_r = addTS(in_samples.pts(), negateTS(next_out_ts_), negateTS(inside_resampler_));
            double cur_drift = cur_drift_r.seconds();
            drifts_[drift_index_++] = cur_drift;
            drift_index_ %= drifts_size_;
            
            // compute average:
            double avg_drift = std::accumulate(drifts_.begin(), drifts_.end(), 0.0) / drifts_size_;
            
            bool drift_negative = cur_drift < 0;
            
            // make sure signs are equal (long-term and momentary):
            if ( ((cur_drift>0 && avg_drift>0) || (cur_drift<0 && avg_drift<0)) && (std::fabs(avg_drift) > max_drift_) && (std::fabs(cur_drift) > max_drift_) && (drift_negative==prev_drift_negative_) ) {
                really_drift = true;
            }
            if (std::fabs(cur_drift) < max_drift_) {
                really_drift = false;
            }
            if (now_compensating_ && (std::fabs(cur_drift) > (0.7*max_drift_))) {
                really_drift = true;
            }
            
            prev_drift_negative_ = drift_negative;
            
            if (really_drift) {
                if ((!now_compensating_) && (drifted_frames_ < 40)) {
                    drifted_frames_++;
                    really_drift = false;
                }
            } else {
                drifted_frames_ = 0;
            }
            if (really_drift) {
                logstream << "Resampler drift: average " << avg_drift << " s, momentary " << cur_drift << " s = " << cur_drift_r << ", swr_delay " << swr_delay_r << ", inside " << inside_resampler_;
            } else {
                //logstream << "...negligible drift... " << drift << " s";
                now_compensating_ = false;
            }
            
            // note for libswresample usage: swr_inject_silence and swr_drop_output are poorly documented
            // from my reading of source code:
            // apparently swr_inject_silence uses input sample rate,
            // and swr_drop_output uses output sample rate
            
            if ((comp_samp_==0) && really_drift && (cur_drift > 0)) {
                // input PTS too big / output PTS too small
                // inject silence to bump output PTS
                int samp_count = cur_drift * src_params_.sample_rate;
                if (samp_count > enc_frame_size_) {
                    samp_count = enc_frame_size_;
                    now_compensating_ = true;
                }
                logstream << "output PTSes too small, injecting silence, " << samp_count << " samples";
                swr_inject_silence(resampler_->raw(), samp_count);
                inside_resampler_ = addTS(inside_resampler_, {samp_count, {1, src_params_.sample_rate} });
            }
            if (comp_samp_!=0) {
                av::Rational tb = {1, src_params_.sample_rate * dst_params_.sample_rate};
                av::Timestamp nextpts(swr_next_pts(resampler_->raw(), in_samples.pts().timestamp(tb)), tb);
                logstream << "swr_next_pts returned " << nextpts.seconds() << "s, next_out_ts_ = " << next_out_ts_.seconds() << "s";
            }
            
            //in_ts_ = in_samples.pts();
            //eq_.in(in_samples);
            in_samples.setPts(av::Timestamp(AV_NOPTS_VALUE, {1, in_samples.sampleRate()}));
            resampler_->push(in_samples);
            inside_resampler_ = addTS(inside_resampler_, { in_samples.samplesCount(), {1, in_samples.sampleRate()} });
            
            
            if ((comp_samp_==0) && really_drift && (cur_drift < 0)) {
                // input PTS too small / output PTS too big
                int samp_count = -cur_drift * dst_params_.sample_rate;
                if (samp_count > enc_frame_size_) {
                    samp_count = enc_frame_size_;
                    now_compensating_ = true;
                }
                logstream << "output PTSes too large, dropping output, " << samp_count << " samples";
                swr_drop_output(resampler_->raw(), samp_count);
                inside_resampler_ = addTS(inside_resampler_, {-samp_count, {1, dst_params_.sample_rate} });
            }
            //logstream << "Resampler delay in->out: " << resampler_->delay() << " samp";
            drainResampler(false);
        } else {
            logstream << "no input samples, flushing";
            flush();
        }
    }
    static av::AudioSamples audioConcat(av::AudioSamples s1, av::AudioSamples s2) {
        auto align = av::SampleFormat::AlignDefault;
        
        if (s1.samplesCount()==0) {
            return s2;
        }
        if (s2.samplesCount()==0) {
            return s1;
        }
        
        if (!(s1.sampleFormat()==s2.sampleFormat() && s1.channelsLayout()==s2.channelsLayout() && s1.sampleRate()==s2.sampleRate())) {
            throw Error(std::string("audioConcat: different audio parameters") +
                " format: " + s1.sampleFormat().name() + " " + s2.sampleFormat().name() +
                ", channel layout: " + s1.channelsLayoutString() + " " + s2.channelsLayoutString() +
                ", sample rate: " + std::to_string(s1.sampleRate()) + " " + std::to_string(s2.sampleRate()));
        }
        
        av::AudioSamples r(s1.sampleFormat(), s1.samplesCount()+s2.samplesCount(), s1.channelsLayout(), s1.sampleRate(), align);
        auto copyPlane = [&](size_t channels_per_plane, size_t i) {
            uint8_t* ptr = r.data(i);
            size_t s1size = channels_per_plane*s1.samplesCount()*r.sampleFormat().bytesPerSample();
            memcpy(ptr, s1.data(i), s1size);
            memcpy(ptr+s1size, s2.data(i), channels_per_plane*s2.samplesCount()*r.sampleFormat().bytesPerSample());
        };
        if (r.sampleFormat().isPlanar()) {
            for (size_t i=0; i<r.channelsCount(); i++) {
                copyPlane(1, i);
            }
        } else {
            copyPlane(r.channelsCount(), 0);
        }
        r.setTimeBase({1, r.sampleRate()});
        r.setComplete(true);
        return r;
    }
    virtual void flush() {
        flushInternal();
        this->finished_ = true;
    }
    DynamicAudioResampler(std::unique_ptr<Source<av::AudioSamples>> &&source, std::unique_ptr<Sink<av::AudioSamples>> &&sink, const AudioParameters &dst_params, size_t comp_samp): NodeSISO<av::AudioSamples, av::AudioSamples>(std::move(source), std::move(sink)), dst_params_(dst_params), forward_channels_(dst_params.channel_layout==0), comp_samp_(comp_samp) {
    }
    template<typename Child> static std::shared_ptr<Child> createCommon(EdgeManager &edges, const Parameters &params, bool have_channels) {
        AudioParameters dst_params;
        if (have_channels) {
            if (params.count("dst_channel_layout")==1) {
                std::string layout_s = params["dst_channel_layout"].get<std::string>();
                dst_params.channel_layout = av_get_channel_layout(layout_s.c_str());
            } else if (params.count("dst_channels")==1) {
                int64_t cnt = params["dst_channels"].get<int>();
                dst_params.channel_layout = av_get_channel_layout_nb_channels(cnt);
            } else {
                throw Error("No dst_channel_layout or dst_channels specified!");
            }
            if (dst_params.channel_layout==0) {
                throw Error("Invalid channels specification!");
            }
        } else {
            dst_params.channel_layout = 0;
        }
        dst_params.sample_rate = params.at("dst_sample_rate").get<int>();
        // TODO: automatic sample_format detection
        // based on what encoder accepts
        // (it requires refactoring of INeedsOutputFrameSize -> INeedAudioEncoderParameters???)
        dst_params.sample_format = av::SampleFormat(params.at("dst_sample_format").get<std::string>());
        
        if (dst_params.sample_format.get() == AV_SAMPLE_FMT_NONE) {
            throw Error("Invalid dst_sample_format");
        }
        double compensation = 0;
        if (params.count("compensation")==1) {
            compensation = params["compensation"];
        }
        int comp_samp;
        if (compensation >= 0) {
            comp_samp = static_cast<int>(compensation * (double)dst_params.sample_rate);
            if (comp_samp==1) comp_samp = 2;
        } else {
            comp_samp = 1;
        }
        
        auto r = NodeSISO<av::AudioSamples, av::AudioSamples>::template createCommon<Child>(edges, params, dst_params, comp_samp);
        if (params.count("max_drift")==1) {
            r->max_drift_ = params["max_drift"];
        }
        return r;
    }

    virtual av::Rational timeBase() {
        return av::Rational(1, dst_params_.sample_rate);
    }
};

class DynamicAudioResamplerProcessChannels: public DynamicAudioResampler, public IAudioMetadataSource {
public:
    virtual int sampleRate() {
        return dst_params_.sample_rate;
    }
    virtual av::SampleFormat sampleFormat() {
        return dst_params_.sample_format;
    }
    virtual uint64_t channelLayout() {
        return dst_params_.channel_layout;
    }
    static std::shared_ptr<DynamicAudioResamplerProcessChannels> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return DynamicAudioResampler::createCommon<DynamicAudioResamplerProcessChannels>(edges, params, true);
    }
    using DynamicAudioResampler::DynamicAudioResampler;
};

class DynamicAudioResamplerForwardChannels: public DynamicAudioResampler {
public:
    static std::shared_ptr<DynamicAudioResamplerForwardChannels> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        return DynamicAudioResampler::createCommon<DynamicAudioResamplerForwardChannels>(edges, params, false);
    }
    using DynamicAudioResampler::DynamicAudioResampler;
};

DECLNODE(resample_audio, DynamicAudioResamplerProcessChannels);
DECLNODE(resample_audio_forward_channels, DynamicAudioResamplerForwardChannels);
