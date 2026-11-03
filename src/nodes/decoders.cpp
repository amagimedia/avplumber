#include "node_common.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>
#include "../avutils.hpp"
#include "../timeline.hpp"
#include <avcpp/codeccontext.h>
#include "../cuda_array_format.hpp"
#include "../decoder_surfaces.hpp"
#include "../hwaccel.hpp"
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
}

// A decoder never ends over what its stream gives it: a packet it cannot decode, a pool it
// cannot get or a surface it cannot find costs that one source pictures, which its consumers
// repeat, while the node counts what happened in its `status` object and goes on with the next
// packet. The one thing that ends it is being told to.
template<typename Child, typename DecoderContext, typename OutputFrame> class Decoder:
    public NodeSISO<av::Packet, OutputFrame>, public ReportsFinishByFlag, public IFlushable, public IDecoder, public ITimeBaseSource,
    public IReturnsObjects {
protected:
    using Clock = std::chrono::steady_clock;
    av::Codec codec_;
    DecoderContext dec_;
    std::recursive_mutex mutex_;
    std::shared_ptr<IStreamsInput> input_hold_;
    std::weak_ptr<IPlaybackControl> playback_hold_;
    size_t dec_errors_ = 0;   // consecutive packets that failed to decode
    av::Timestamp last_pts_ = NOTS;
    av::PixelFormat pixel_format_ = AV_PIX_FMT_NONE;
    bool pixel_format_optional_ = false;
    std::shared_ptr<HWAccelDevice> hwaccel_;
    av::Timestamp discard_until_ = NOTS;
    std::mutex discard_until_mutex_;
    std::list<OutputFrame> flush_frames_;
    bool flush_magic_ = false;
    int waiting_for_frame_ = 0;
    av::Timestamp flush_magic_frame_ = NOTS;   // target frame; its re-fed decode must not surface twice
    bool finish_after_flush_ = false;
    bool hold_at_eof_ = false;
    bool decoder_at_eof_ = false;
    // The packet in front of the queue restarted the stream (takeStreamRestart): sent again after
    // a surface shortage, it must not drain and reset the decoder a second time. Cleared when the
    // packet leaves the queue.
    bool restarted_for_front_ = false;
    std::atomic<uint64_t> stream_restarts_{0};
    av::Timestamp flush_timeout_ts_ = NOTS;
    av::Timestamp flush_timeout_ = {500, {1, 1000}}; // 500ms
    // CUarray output with unsafe_output: a fixed pool whose surfaces are the output frames.
    std::unique_ptr<avp::DecoderSurfaceGate> surface_gate_;
    std::optional<Clock::time_point> gate_paused_since_;
    bool gate_overrun_ = false;
    uint64_t gate_pauses_ = 0;
    avp::SurfaceExhaustion exhaustion_;
    bool exhausted_reported_ = false;   // this episode is in the ledger's count
    std::shared_ptr<avp::DecoderPoolLedger> pool_ledger_;
    int64_t pool_budget_bytes_ = 0;
    int64_t pool_bytes_ = 0;   // the pool reserved last, for parking
    int pool_surfaces_ = 0;
    avp::DecoderParking::Geometry pool_geometry_;
    std::string pool_refused_;   // why get_format refused the pool; the source shows nothing until one fits
    int configured_extra_ = -1;  // extra_hw_frames as asked for, before a stream made it smaller
    // Parking: see DecoderParking. The newest decoded frame is kept so that closing can park the
    // decoder; it is the frame the consumers hold anyway, and the gate does not count this copy.
    // resume_frame_ is the parked frame this decoder took over, until it has a frame of its own.
    std::shared_ptr<avp::DecoderParking> parking_;
    int park_limit_ = 0;
    OutputFrame last_frame_;
    AVFrame *resume_frame_ = nullptr;
    bool counted_live_ = false;   // in the lot's count of decoders with a session
    // Consecutive failures from here on make the source degraded: the consumers show its last
    // picture, and the show's totals count it until a decode succeeds.
    static constexpr size_t kDegradedAfterErrors = 30;
    bool degraded_ = false;
    int pace_ms_ = 0;   // the rest before the next packet, after a failure
    uint64_t frames_out_ = 0;
    // What the control thread reads: a copy of the node thread's state, under its own lock.
    struct Health {
        bool degraded = false;
        std::string reason;
        uint64_t decode_errors = 0, consecutive_errors = 0, frames = 0;
        Clock::time_point last_frame{};
        avp::SurfaceExhaustion::Counters surfaces;
        bool exhausted_now = false, hiding = false;
        uint64_t gate_pauses = 0;
        int surfaces_held = 0;
        std::string pool;
        bool took_over = false;   // decodes on a parked decoder, see DecoderParking
    };
    std::mutex health_mutex_;
    Health health_;
    uint64_t decode_errors_total_ = 0;
    Clock::time_point last_frame_at_{};
    Clock::time_point error_logged_at_{};
    uint64_t errors_unlogged_ = 0;
    //AVBufferRef *out_frames_ref_ = nullptr;
    /* input_hold_ is a workaround to prevent StreamInput from being destroyed
     * when the shared_ptr is set to null in NodeWrapper
     * There are other ways to handle this:
     * X- use waitSinksEmpty() as it is and redesign the EdgeSource to have separate
     *    peek() and pop() functions (just like ReaderWriterQueue has)
     *    - IT DOESN'T WORK!
     *  - redesign NodeWrapper to not destroy wrapped Node until the group is destroyed
     */
    static av::Codec codecFromName(const std::string codec_name) {
        if (codec_name.length()>0) {
            av::Codec r = av::findDecodingCodec(codec_name);
            if (r.isNull()) {
                throw Error("unknown codec: " + codec_name);
            }
            return r;
        } else {
            return {};
        }
    }
    static bool isAvError(const std::exception &e, int code) {
        auto se = dynamic_cast<const std::system_error*>(&e);
        return se && se->code().category() == av::ffmpeg_category() && se->code().value() == code;
    }
    bool isDecoderAgain(const std::exception &e) const { return isAvError(e, AVERROR(EAGAIN)); }
    // What a failed decode was. Shortage: the hwaccel's "No decoder surfaces left" came back as
    // ENOMEM (HEVC), the picture still to be had. LostPicture: an error while the consumers hold
    // the pool's spare surfaces; H.264 swallows the hwaccel's ENOMEM, goes on without the
    // surface and fails later on the reference it lost, so the error code says nothing and the
    // picture is gone. Error: anything else.
    enum class Failure { Error, Shortage, LostPicture };
    Failure classify(const std::exception &e) const {
        if (!surface_gate_) return Failure::Error;
        if (isAvError(e, AVERROR(ENOMEM))) return Failure::Shortage;
        if (surface_gate_->held() >= surface_gate_->limit()) return Failure::LostPicture;
        return Failure::Error;
    }

    void publishHealth() {
        std::lock_guard<std::mutex> lock(health_mutex_);
        health_.degraded = degraded_;
        health_.reason = degraded_ ? (pool_refused_.empty() ? "decode errors" : "pool refused: " + pool_refused_) : "";
        health_.decode_errors = decode_errors_total_;
        health_.consecutive_errors = dec_errors_;
        health_.frames = frames_out_;
        health_.last_frame = last_frame_at_;
        health_.surfaces = exhaustion_.counters();
        health_.exhausted_now = exhaustion_.stuck();
        health_.hiding = exhaustion_.hiding();
        health_.gate_pauses = gate_pauses_;
        health_.surfaces_held = surface_gate_ ? surface_gate_->held() : 0;
    }
    void setDegraded(bool degraded) {
        if (degraded == degraded_) return;
        degraded_ = degraded;
        if (pool_ledger_) pool_ledger_->degraded(degraded);
        logstream << (degraded ? "decoder degraded: " + (pool_refused_.empty() ? std::to_string(dec_errors_) + " packets failed in a row"
                                                                              : pool_refused_)
                               : std::string("decoder recovered"));
    }
    // Milliseconds of one picture, for pacing packets the decoder cannot use: without a picture
    // to hand on nothing downstream paces the reader, and a decoder that fails every packet of a
    // file would run through it at the speed of its errors.
    int picturePeriodMs() const {
        const av::Rational rate = dec_.stream().frameRate();
        if (rate.getNumerator() <= 0 || rate.getDenominator() <= 0) return 20;
        return std::clamp(int(1000 * int64_t(rate.getDenominator()) / rate.getNumerator()), 1, 100);
    }
    void decodeFailed(const std::exception &e) {
        ++dec_errors_;
        ++decode_errors_total_;
        pace_ms_ = dec_errors_ > 1 ? picturePeriodMs() : 0;
        const auto now = Clock::now();
        if (now - error_logged_at_ >= std::chrono::seconds(1)) {
            logstream << "Decode error: " << e.what()
                      << (errors_unlogged_ ? " (" + std::to_string(errors_unlogged_) + " more since the last line, not logged)" : "");
            error_logged_at_ = now;
            errors_unlogged_ = 0;
        } else {
            ++errors_unlogged_;
        }
        if (dec_errors_ >= kDegradedAfterErrors) setDegraded(true);
        publishHealth();
    }
    void decodeSucceeded() {
        dec_errors_ = 0;
        pace_ms_ = 0;
    }
    // A picture handed on is what ends a degradation or a surface shortage; a decode call that
    // merely raised nothing (a slice the decoder skipped, a picture withheld) is not.
    void pictureOut() {
        ++frames_out_;
        last_frame_at_ = Clock::now();
        if (pool_refused_.empty()) setDegraded(false);
        if (exhaustion_.stuck()) {
            const auto stuck_ms = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame_at_ - exhaustion_.episodeSince()).count();
            const uint64_t repeated = exhaustion_.shown(last_frame_at_, std::chrono::milliseconds(picturePeriodMs()));
            if (pool_ledger_) pool_ledger_->repeated(repeated);
            logstream << "decoder shows pictures again " << stuck_ms << " ms after it ran out of surfaces (about "
                      << repeated << " pictures repeated, " << exhaustion_.counters().retries << " packets sent meanwhile)";
            avp::timeline::mark("decoder surfaces back");
        }
        if (exhausted_reported_) {
            exhausted_reported_ = false;
            if (pool_ledger_) pool_ledger_->exhausted(false);
            publishHealth();   // the alert ends now, not at the next periodic publish
        } else if (frames_out_ % 60 == 1) {
            publishHealth();
        }
    }
    // A packet's decode failed. True: the packet is used up. False: it stays queued and is sent
    // again once there is room for its picture.
    bool failed(const std::exception &e) {
        switch (classify(e)) {
        case Failure::Shortage: return !surfaceShortage(e);
        case Failure::LostPicture: pictureLost(e); return true;
        default: decodeFailed(e); return true;
        }
    }
    void reportExhausted(const std::exception &e, const char *what) {
        if (exhausted_reported_) return;
        exhausted_reported_ = true;
        if (pool_ledger_) pool_ledger_->exhausted(true);
        logstream << "Warning: no decoder surface for a picture (" << e.what() << "): consumers hold "
                  << surface_gate_->held() << " of the pool's " << pool_surfaces_ << "; " << what
                  << " (episode " << exhaustion_.counters().exhaustions << ")";
        avp::timeline::mark("decoder out of surfaces");
    }
    void pictureLost(const std::exception &e) {
        exhaustion_.lost(Clock::now());
        reportExhausted(e, "the picture is lost, nothing shown until the next keyframe");
        ++decode_errors_total_;
        pace_ms_ = picturePeriodMs();
        publishHealth();
    }
    bool handleFrame(OutputFrame &frm, const av::Packet &pkt) {
        if (!frm) {
            return false;
        }
        if (!pkt.isKeyPacket() && (last_pts_.isValid() && (last_pts_ > frm.pts()))) {
            logstream << "Warning: Got out of order frame from decoder: " << last_pts_ << " -> " << frm.pts();
        }
        last_pts_ = frm.pts();
        bool put = true;
        bool completed_flush_magic = false;
        if (flush_magic_ && waiting_for_frame_ > 0) {
            if (abs(addTS(pkt.pts(), negateTS(frm.pts())).seconds()) < 0.008) {
                logstream << "flush magic done, got the frame that we need, " << waiting_for_frame_ << " iterations";
                // The packet was fed once per iteration; the extra decodes of the
                // target frame surface later and must not be output again.
                flush_magic_frame_ = waiting_for_frame_ > 1 ? frm.pts() : NOTS;
                completed_flush_magic = true;
                waiting_for_frame_ = 0;
            } else {
                waiting_for_frame_++;
                put = false;
            }
            if (waiting_for_frame_ > 5) {
                logstream << "decoder did not give us correct frame within " << waiting_for_frame_ << " frames, breaking the loop";
                waiting_for_frame_ = 0;
                put = true;
            }
        }
        if (put && !completed_flush_magic && flush_magic_frame_.isValid()) {
            if (frm.pts() == flush_magic_frame_) {
                logstream << "dropping re-fed duplicate of flush magic frame " << frm.pts();
                put = false;
            } else if (frm.pts() > flush_magic_frame_) {
                flush_magic_frame_ = NOTS;
            }
        }
        // Only frames that survived flush magic may consume the discard target: a
        // stale frame left in the decoder queue after a seek must not clear it.
        if (put) {
            auto lock = std::lock_guard<decltype(discard_until_mutex_)>(discard_until_mutex_);
            if (discard_until_.isValid()) {
                if (frm.pts() >= discard_until_) {
                    logstream << "pts " << frm.pts() << " reached discard_until " << discard_until_;
                    discard_until_ = NOTS;
                } else {
                    put = false;
                }
            }
        }
        // After a lost picture, what the decoder predicts from it is not shown until a keyframe;
        // paced like pictures, so the reader is not run through the stream meanwhile.
        if (put) {
            const bool was_hiding = exhaustion_.hiding();
            if (!exhaustion_.showPicture(frm.raw()->flags & AV_FRAME_FLAG_KEY)) {
                put = false;
                pace_ms_ = picturePeriodMs();
            } else if (was_hiding) {
                logstream << "decoder shows pictures again from a keyframe, " << exhaustion_.counters().pictures_hidden
                          << " hidden in all";
            }
        }
        if (parking_) {
            last_frame_ = frm;   // before track(): this copy is not a consumer's
            if (resume_frame_) forgetResumeFrame();
        }
        if (put) {
            setFrameTimestamps(frm);
            if (surface_gate_ && !surface_gate_->track(frm.raw()))
                logstream << "Warning: decoded frame has no free buffer slot; its surface is not counted";
            pictureOut();
            this->sink_->put(frm);
        }
        return true;
    }
    // A decode needs a free surface. While consumers hold limit() decoded frames the packet stays
    // queued, and they repeat the frames they hold. A consumer that releases a frame only when a
    // newer one arrives, a window larger than the pool, would wait forever: after a second,
    // decoding goes on and the pool may run out, as without the gate. Once it has, every packet
    // waits for the consumers to hold fewer than limit() again (at most retryAfter(), in case
    // the shortage is the decoder's own) until a picture is shown: the gate's rule, with no
    // giving up, since giving up is what ran the pool out.
    bool admitPacket() {
        using namespace std::chrono;
        if (exhaustion_.stuck()) {
            surface_gate_->wait(exhaustion_.retryAfter());
            exhaustion_.retrying();
            return !bool(this->finished_);
        }
        if (gate_overrun_) {
            if (surface_gate_->held() >= surface_gate_->limit()) return true;
            gate_overrun_ = false;
        }
        if (surface_gate_->wait(milliseconds(10))) {
            if (gate_paused_since_) avp::timeline::mark("decoder resumed");
            if (gate_paused_since_ && (gate_pauses_ == 1 || gate_pauses_ % 100 == 0))
                logstream << "decoder resumed after "
                          << duration_cast<milliseconds>(steady_clock::now() - *gate_paused_since_).count()
                          << " ms (" << gate_pauses_ << " pauses so far)";
            gate_paused_since_.reset();
            return true;
        }
        const auto now = steady_clock::now();
        if (!gate_paused_since_) {
            gate_paused_since_ = now;
            avp::timeline::mark("decoder paused: consumers hold " + std::to_string(surface_gate_->held()) + " frames");
            if (++gate_pauses_ == 1 || gate_pauses_ % 100 == 0)
                logstream << "decoder paused: consumers hold " << surface_gate_->held()
                          << " decoded frames, the pool's spare surfaces (extra_hw_frames + 1); pause " << gate_pauses_;
            publishHealth();
        } else if (now - *gate_paused_since_ > seconds(1)) {
            logstream << "Warning: consumers have held " << surface_gate_->held()
                      << " decoded frames for over a second; decoding on, the surface pool may run out: "
                         "raise extra_hw_frames above what they hold";
            avp::timeline::mark("decoder gate gave up");
            gate_overrun_ = true;
            gate_paused_since_.reset();
            return true;
        }
        return false;
    }
    // The picture `pkt` carries found no surface. Retry: the packet stays queued and is sent
    // again after a release. Drop: it is given up on, and what follows is hidden until a keyframe.
    bool surfaceShortage(const std::exception &e) {
        const auto now = Clock::now();
        const auto waiting_since = exhaustion_.stuck() ? exhaustion_.since() : now;
        const auto next = exhaustion_.failed(now);
        reportExhausted(e, "the packet waits for room");
        if (next == avp::SurfaceExhaustion::Next::Drop) {
            logstream << "Warning: still no decoder surface after "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - waiting_since).count()
                      << " ms; dropping the packet, pictures hidden until the next keyframe";
            pace_ms_ = picturePeriodMs();
        }
        publishHealth();
        return next == avp::SurfaceExhaustion::Next::Retry;
    }
    // What avcodec_get_hw_frames_parameters leaves out: nvdec_init_hwframes adds these to its count.
    void forgetResumeFrame() {
        dec_.raw()->hwaccel_context = nullptr;
        av_frame_free(&resume_frame_);
    }
    // Hands the decoder this node has to the parking lot, the session closed or about to be.
    void parkSession() {
        const AVFrame *frame = last_frame_ ? last_frame_.raw() : resume_frame_;
        if (!frame) return;
        if (const int freed = parking_->park(frame, pool_geometry_, pool_bytes_, pool_ledger_, codecName(), park_limit_))
            logstream << "parked this decoder, freeing " << freed << " older parked decoder(s) over the limit of " << park_limit_;
        else
            logstream << "parked this decoder (" << pool_surfaces_ << " surfaces)";
        last_frame_ = OutputFrame();
        if (resume_frame_) forgetResumeFrame();
    }

    static constexpr int kNvdecWorkingSurfaces = 3;
    static constexpr int kNvdecMaxSurfaces = 32;   // nvdec.c MAX_NUM_REGISTERED_DECODE_SURFACES

    // Before FFmpeg allocates the pool: its size for this stream, whatever the codec (references,
    // reorder delay, working surfaces and extra_hw_frames), against NVDEC's limit and the instance's
    // budget. A stream past the limit gives up headroom first: it decodes with less room for a late
    // output, which beats not decoding. A parked decoder that fits takes this stream over; otherwise
    // parked decoders are freed, oldest first, when the pool would not fit beside them.
    bool reservePool(AVCodecContext *cc) {
        if (configured_extra_ < 0) configured_extra_ = cc->extra_hw_frames;
        cc->extra_hw_frames = configured_extra_;   // an earlier SPS may have needed less of it
        AVBufferRef *frames = nullptr;
        const int r = avcodec_get_hw_frames_parameters(cc, cc->hw_device_ctx, AV_PIX_FMT_CUDA, &frames);
        if (r < 0) {
            logstream << "Warning: decoder pool size unknown (" << av::error2string(r) << "); not counted";
            return true;
        }
        const auto *ctx = reinterpret_cast<const AVHWFramesContext *>(frames->data);
        const int surface = av_image_get_buffer_size(ctx->sw_format, ctx->width, ctx->height, 1);
        int surfaces = ctx->initial_pool_size + kNvdecWorkingSurfaces;
        avp::DecoderParking::Geometry geometry;
        geometry.codec = cc->codec_id;
        geometry.width = ctx->width;
        geometry.height = ctx->height;
        geometry.sw_format = ctx->sw_format;
        av_buffer_unref(&frames);
        const int over = surfaces - kNvdecMaxSurfaces;
        if (over > 0 && over > cc->extra_hw_frames) {
            pool_refused_ = "stream needs " + std::to_string(surfaces - cc->extra_hw_frames) + " surfaces before any headroom, NVDEC holds " +
                            std::to_string(kNvdecMaxSurfaces);
            logstream << "Warning: " << pool_refused_ << "; this source shows nothing until a stream fits";
            setDegraded(true);
            publishHealth();
            return false;
        }
        if (over > 0) {
            logstream << "Warning: stream needs " << surfaces << " surfaces, NVDEC holds " << kNvdecMaxSurfaces
                      << ": extra_hw_frames " << cc->extra_hw_frames << " -> " << cc->extra_hw_frames - over
                      << ", so a late output can starve this source sooner";
            cc->extra_hw_frames -= over;
            surfaces = kNvdecMaxSurfaces;
        }
        surface_gate_->setLimit(cc->extra_hw_frames + 1);
        geometry.surfaces = surfaces;
        int64_t bytes = int64_t(surfaces) * std::max(surface, 0);
        if (parking_) {
            // A new stream's parameters closed this context's session before get_format: the
            // decoder goes through the lot too, and comes back if the stream still fits it.
            pool_ledger_->release(this);
            parkSession();
            resume_frame_ = parking_->take(geometry, bytes);
            cc->hwaccel_context = resume_frame_;
            {
                std::lock_guard<std::mutex> lock(health_mutex_);
                health_.took_over = resume_frame_ != nullptr;
            }
            if (resume_frame_)
                logstream << "taking over a parked decoder";
            else if (pool_budget_bytes_)
                if (const int freed = parking_->makeRoom(*pool_ledger_, bytes, pool_budget_bytes_))
                    logstream << "freed " << freed << " parked decoder(s) for this pool";
        }
        pool_geometry_ = geometry;
        bool accepted = false;
        const int64_t total = pool_ledger_->reserve(this, bytes, pool_budget_bytes_, accepted);
        const std::string pools = "decoder pool " + std::to_string(surfaces) + " surfaces, " +
            std::to_string(bytes >> 20) + " MiB; all decoder pools " + std::to_string(total >> 20) + " MiB" +
            (pool_budget_bytes_ ? " of a " + std::to_string(pool_budget_bytes_ >> 20) + " MiB budget" : "");
        {
            std::lock_guard<std::mutex> lock(health_mutex_);
            health_.pool = pools;
        }
        if (!accepted) {
            pool_refused_ = pools;
            logstream << "Warning: over the decoder pool budget: " << pools << "; this source shows nothing until a pool fits";
            setDegraded(true);
            publishHealth();
        } else {
            pool_refused_.clear();
            pool_bytes_ = bytes;
            if (parking_ && !counted_live_) {
                parking_->opened();
                counted_live_ = true;
            }
            pool_surfaces_ = surfaces;
            logstream << pools;
        }
        return accepted;
    }
    bool drainFrame(const av::Packet &blocked_pkt) {
#if API_AVCODEC_NEW_INIT_PACKET
        OutputFrame frm = dec_.decode(av::Packet{nullptr});
        return handleFrame(frm, blocked_pkt);
#else
        (void)blocked_pkt;
        return false;
#endif
    }
public:
    template<typename ...Ts> Decoder(std::unique_ptr<Source<av::Packet>> &&source, std::unique_ptr<Sink<OutputFrame>> &&sink, av::Stream &stream, const std::string codec_name, av::Dictionary options, std::string pixel_format, std::shared_ptr<HWAccelDevice> hwaccel):
        NodeSISO<av::Packet, OutputFrame>(std::move(source), std::move(sink)),
        codec_(codecFromName(codec_name)),
        dec_(stream, codec_),
        hwaccel_(hwaccel)
    {
        this->auto_eof_ = false;
        if (!pixel_format.empty()) {
            pixel_format_optional_ = pixel_format[0]=='?';
            pixel_format_ = av::PixelFormat(pixel_format_optional_ ? pixel_format.substr(1) : pixel_format);
        }

        input_hold_ = this->template findNodeUp<IStreamsInput>();
        dec_.setRefCountedFrames(true);

        bool good_tb = dec_.timeBase().getNumerator() && dec_.timeBase().getDenominator();
        if (!good_tb) {
            logstream << "Decoder detected invalid timebase";
            #if LIBAVCODEC_VERSION_MAJOR < 59 // FFmpeg 5.0
            FF_DISABLE_DEPRECATION_WARNINGS
            if (stream.raw()->codec != nullptr) {
                AVRational &tb = stream.raw()->codec->time_base;
                logstream << "stream->codec not null, tb " << tb.num << "/" << tb.den;
                if (tb.num && tb.den) {
                    dec_.setTimeBase(av::Rational(tb));
                    logstream << "Set timebase from stream->codec";
                    good_tb = true;
                }
            }
            FF_ENABLE_DEPRECATION_WARNINGS
            #else
            logstream << "FIXME: Unable to fix timebase, ffmpeg too new";
            #endif
        }
        if (!good_tb) {
            if (stream.timeBase().getNumerator() && stream.timeBase().getDenominator()) {
                dec_.setTimeBase(stream.timeBase());
                logstream << "Set timebase from stream: " << stream.timeBase();
                good_tb = true;
            }
        }

        if (hwaccel) {
            dec_.raw()->hw_device_ctx = hwaccel->refDeviceContext();
        }
        if (pixel_format_ != AV_PIX_FMT_NONE) {
            dec_.raw()->opaque = this;
            dec_.raw()->get_format = [](AVCodecContext *cc, const enum AVPixelFormat *pix_fmts) -> AVPixelFormat {
                Decoder &self = *(Decoder*)(cc->opaque);
                const enum AVPixelFormat *p;
                for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                    if (*p == self.pixel_format_) {
                        if (self.surface_gate_ && !self.reservePool(cc))
                            return AV_PIX_FMT_NONE;
                        #if 0 // not really needed
                        if (self.hwaccel_) {
                            int r = avcodec_get_hw_frames_parameters(cc, self.hwaccel_->refDeviceContext(), *p, &cc->hw_frames_ctx);
                            if (r == AVERROR(ENOENT)) {
                                logstream << "avcodec_get_hw_frames_parameters not supported by this decoder, returning pixel format anyway";
                            } else {
                                if (r != 0) {
                                    logstream << "Failed to setup hardware decoder: avcodec_get_hw_frames_parameters failed: " << av::error2string(r);
                                    return AV_PIX_FMT_NONE;
                                }
                                r = av_hwframe_ctx_init(cc->hw_frames_ctx);
                                if (r != 0) {
                                    logstream << "Failed to setup hardware decoder: av_hwframe_ctx_init failed: " << av::error2string(r);
                                    return AV_PIX_FMT_NONE;
                                }
                            }
                        }
                        #endif
                        return *p;
                    }
                }
                logstream << "Decoder doesn't support specified pixel_format " << self.pixel_format_;
                if (self.pixel_format_optional_) {
                    logstream << "Using best decoder's pixel format: " << av::PixelFormat(pix_fmts[0]);
                    return pix_fmts[0];
                }
                return AV_PIX_FMT_NONE;
            };
        }

        if (avp::cuda::isArrayFormat(pixel_format_)) {
            const char *flags = options.get("hwaccel_flags");
            if (flags && std::string(flags).find("unsafe_output") != std::string::npos) {
                // Without a budget FFmpeg keeps its 16 extra surfaces; extra_hw_frames_only requires one.
                const char *extra = options.get("extra_hw_frames");
                surface_gate_ = std::make_unique<avp::DecoderSurfaceGate>((extra ? std::max(0, std::atoi(extra)) : 16) + 1);
            }
        }
        dec_.open(options, codec_);
        logstream << "Opened decoder " << dec_.codec().name() << std::endl;
    };
    virtual std::string codecName() const {
        if (!dec_.codec().isNull()) {
            return dec_.codec().name();
        } else {
            return "???";
        }
    }
    virtual std::string codecMediaTypeString() const {
        if (!dec_.codec().isNull()) {
            AVMediaType mt = dec_.codec().type();
            return mediaTypeToString(mt);
        }
        return "?";
    }
    virtual std::string fieldOrderString() const {
        return fieldOrderToString(dec_.raw()->field_order);
    }
    virtual av::Rational timeBase() final {
        //return av::Rational(this->dec_.stream().raw()->codec->time_base);
        return av::Rational(this->dec_.raw()->time_base);
    }
    /// `status`: how this source is doing. `degraded` with a `reason` while packets fail in a row
    /// or no pool fits the budget; `surfaces` (CUarray decoders) counts exhaustion episodes,
    /// retries, dropped packets, hidden pictures, recoveries and the pictures repeated in ended
    /// episodes, with `exhausted_now`.
    Parameters getObject(const std::string key) override {
        if (key != "status") throw Error("decoder: unknown object " + key);
        std::lock_guard<std::mutex> lock(health_mutex_);
        const auto &h = health_;
        Parameters result = {{"degraded", h.degraded}, {"reason", h.reason}, {"decode_errors", h.decode_errors},
                             {"consecutive_errors", h.consecutive_errors}, {"frames", h.frames},
                             {"stream_restarts", stream_restarts_.load()}};
        result["last_frame_age_ms"] = h.frames
            ? std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - h.last_frame).count() : -1;
        if (surface_gate_) {
            const auto &s = h.surfaces;
            result["surfaces"] = {{"held", h.surfaces_held}, {"limit", surface_gate_->limit()}, {"pool", pool_surfaces_},
                                  {"pauses", h.gate_pauses}, {"exhaustions", s.exhaustions}, {"exhausted_now", h.exhausted_now},
                                  {"retries", s.retries}, {"packets_dropped", s.packets_dropped}, {"pictures_lost", s.pictures_lost},
                                  {"pictures_hidden", s.pictures_hidden}, {"hiding", h.hiding}, {"recoveries", s.recoveries},
                                  {"repeats", s.repeats}, {"took_over", h.took_over}};
            result["pool"] = h.pool;
        }
        return result;
    }
    virtual void flush() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        bool decoded_something = false;
        while (true) {
            OutputFrame frm;
            try {
                frm = dec_.decode(av::Packet());
                if (frm) {
                    setFrameTimestamps(frm);
                    flush_frames_.push_back(frm);
                    decoded_something = true;
                }
            } catch (std::exception &e) {
                logstream << "Warning: Exception " << e.what() << " when flushing decoder." << std::endl;
                break; // flush error is not considered error
            }
            if (!frm)
                break;
        }
        if (decoded_something) {
            flush_timeout_ts_ = addTS(wallclock.absolute_ts(), flush_timeout_);
        }
        //dec_.close();
        //input_hold_ = nullptr;
        // try to put frames to sink
        while (!flush_frames_.empty()) {
            OutputFrame frame = flush_frames_.front();
            if (!this->sink_->put(frame, true)) {
                break;
            }
            flush_frames_.pop_front();
        }
        if (flush_frames_.empty()) {
            this->finished_ = true;
            flush_timeout_ts_ = NOTS;
        } else {
            if (flush_timeout_ts_.isValid() && (wallclock.absolute_ts() > flush_timeout_ts_)) {
                logstream << "Flush timeout " << flush_timeout_ts_ << " exceeded, finishing decoder";
                flush_frames_.clear();
                flush_timeout_ts_ = NOTS;
            }
            this->finish_after_flush_ = true;
        }
    }
    void setFrameTimestamps(OutputFrame& frm) {
        auto pb = playback_hold_.lock();
        if (!pb) {
            playback_hold_ = this->template findNodeUp<IPlaybackControl>();
            pb = playback_hold_.lock();
        }
        if (pb) {
            pb->setFrameMetadataTimestamps(frm);
        }
    }
    virtual void process() {
        // check if there is something in flush queue
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            if (!flush_frames_.empty()) {
                OutputFrame frame = *flush_frames_.begin();
                flush_frames_.pop_front();
                this->sink_->put(frame);
                if (flush_frames_.empty() && finish_after_flush_) {
                    this->finished_ = true;
                }
                return;
            }
        }
        if (pace_ms_ > 0) {
            // Rest between packets the decoder cannot use, outside the lock: see picturePeriodMs().
            std::this_thread::sleep_for(std::chrono::milliseconds(std::exchange(pace_ms_, 0)));
            if (this->finished_) return;
        }
        // Copy from the edge queue, but keep the packet queued until the
        // decoder accepts it. FFmpeg send_packet(EAGAIN) means retry same input.
        av::Packet pkt;
        // Blocking poll: park the decoder thread until a packet arrives instead of
        // busy-spinning. Measured ~41% CPU vs ~100% (full core) with non-blocking
        // ,0 on a starved live source. wait_peek() is event-woken by the producer's
        // enqueue and by finishConsumer() (stop / flushAndSeek), so shutdown, flush
        // and seek all wake it cleanly. Safe for the OBS switcher/playlist path: each
        // clip is a fresh graph (new edges reset finish_consumer_), so the post-seek
        // busy-poll degradation only affects in-place seeks (ssgw replay), where it
        // is merely no worse than the non-blocking form.
        if (!this->source_->tryPeek(pkt)) {
            //flush();
            return;
        }
        const bool data_packet = (!pkt.isNull()) && pkt.isComplete() && !isEofMarker(pkt);
        // The gate waits without the lock: a status read or a seek must not wait behind it.
        if (data_packet && surface_gate_ && !admitPacket())
            return;   // the packet stays queued
        // lock us, to prevent race condition with flushing!
        {
            std::lock_guard<std::recursive_mutex> lock(mutex_);
            if (data_packet) {
                if (decoder_at_eof_) {
                    avcodec_flush_buffers(dec_.raw());
                    decoder_at_eof_ = false;
                }
                if (takeStreamRestart(pkt) && !restarted_for_front_) {
                    restarted_for_front_ = true;
                    restartStream(pkt);
                }
                int iter = 0;
                bool packet_consumed = false;
                do {
                    iter++;
                    // not a flush packet
                    try {
                        OutputFrame frm = dec_.decode(pkt);
                        packet_consumed = true;
                        decodeSucceeded();
                        handleFrame(frm, pkt);
                    } catch (std::exception &e) {
                        if (isDecoderAgain(e)) {
                            bool drained = false;
                            try {
                                drained = drainFrame(pkt);
                            } catch (std::exception &pending) {
                                // The picture FFmpeg had pending failed, as a decode of our
                                // packet would have; the packet was not taken and stays queued.
                                failed(pending);
                                if (packet_consumed) break;
                                return;
                            }
                            if (drained) {
                                dec_errors_ = 0;
                                if (packet_consumed) {
                                    break;
                                }
                                return;
                            }
                            logstream << "Decode backpressure without drainable frame: " << e.what();
                            if (packet_consumed) {
                                break;
                            }
                            return;
                        }
                        if (!failed(e)) return;   // the packet stays queued, sent again when there is room
                        packet_consumed = true;
                    }
                    //if (!frm) this->finished_ = true;
                } while (flush_magic_ && waiting_for_frame_ > 0);
                if (packet_consumed) {
                    this->source_->pop();
                    restarted_for_front_ = false;
                }
            } else {
                this->source_->pop();
                // this is flush packet
                // so flush decoder
                flush();
                // pass eof marker to next node after all codec-drained frames.
                // Must do this AFTER flush() because flush() may have set finished_=true
                // when all codec frames fit in the sink; resetting here ensures the EOF
                // marker is actually sent before the decoder thread exits.
                if (isEofMarker(pkt) && hold_at_eof_) {
                    // Drain the last frames but keep an interactive recording
                    // seekable. Reset the codec before accepting the next packet.
                    finished_ = false;
                    finish_after_flush_ = false;
                    decoder_at_eof_ = true;
                } else if (isEofMarker(pkt)) {
                    finished_ = false;
                    finish_after_flush_ = true;
                    this->flush_frames_.push_back(OutputFrame());
                }
            }
        }
    }
    // The input started its stream again (a file looped): the pictures the decoder still holds
    // end the last pass and go out, then the decoder forgets its references. FFmpeg then shows
    // nothing it cannot decode without them: H.264 waits for an IDR or recovery point, HEVC for
    // a random access point and skips the leading pictures that predict from before it.
    void restartStream(const av::Packet &pkt) {
        for (;;) {
            OutputFrame frm;
            try {
                frm = dec_.decode(av::Packet());
            } catch (const std::exception &) {
                break;   // drained, or the last picture failed: nothing more to hand on
            }
            if (!frm) break;
            handleFrame(frm, pkt);
        }
        avcodec_flush_buffers(dec_.raw());
        ++stream_restarts_;
    }
    virtual void discardUntil(av::Timestamp pts) {
        logstream << "will wait for frame and discard until " << pts;
        std::lock_guard<std::recursive_mutex> decoder_lock(mutex_);
        if (hold_at_eof_) {
            flush_frames_.clear();
        }
        auto lock = std::lock_guard<decltype(discard_until_mutex_)>(discard_until_mutex_);
        waiting_for_frame_ = 1;
        discard_until_ = pts;
    }
    virtual ~Decoder() {
        // Park before closing: the frame keeps the NVDEC session and the pool, so closing
        // destroys neither here, on a thread the rest of the show shares the device with.
        if (parking_) {
            if (counted_live_) parking_->closed();
            parkSession();
        }
        last_frame_ = OutputFrame();
        try {
            dec_.close();
        } catch (std::exception &e) {
        }
        if (pool_ledger_) {
            pool_ledger_->release(this);
            if (degraded_) pool_ledger_->degraded(false);
            if (exhausted_reported_) pool_ledger_->exhausted(false);
            // A source that closes short of surfaces repeated its last picture until now.
            pool_ledger_->repeated(exhaustion_.episodePictures(Clock::now(), std::chrono::milliseconds(picturePeriodMs())));
        }
    }
    static std::shared_ptr<Child> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::Packet>> src_edge = edges.find<av::Packet>(params["src"]);
        std::shared_ptr<Edge<OutputFrame>> dst_edge = edges.find<OutputFrame>(params["dst"]);
        std::shared_ptr<IStreamsInput> input = src_edge->findNodeUp<IStreamsInput>();
        std::shared_ptr<InputStreamMetadata> md = nullptr;
        if (input != nullptr) md = src_edge->findMetadataUp<InputStreamMetadata>();
        if (md == nullptr) {
            throw Error("Couldn't initialize decoder: no input stream.");
        }
        std::string codec_name;
        AVCodecParameters &cpar = *md->source_stream.raw()->codecpar;
        std::string in_codec = avcodec_get_name(cpar.codec_id);
        if (params.count("codec")==1) {
            codec_name = params["codec"];
        } else if (params.count("codec_map")==1) {
            const Parameters &map = params["codec_map"];
            if (map.count(in_codec)==1) {
                codec_name = map[in_codec];
                logstream << "Detected codec " << in_codec << ", using implementation: " << codec_name;
            } else {
                logstream << "Detected codec " << in_codec << " not in codec_map, using libavcodec default";
            }
        }
        std::string pixel_format;
        if (params.count("pixel_format")) {
            pixel_format = params["pixel_format"];
        }
        std::shared_ptr<HWAccelDevice> hwaccel;
        if (params.count("hwaccel")) {
            bool use_hw = true;
            if (params.count("hwaccel_only_for_codecs")) {
                auto allowed_codecs = jsonToStringList(params["hwaccel_only_for_codecs"]);
                use_hw = std::find(allowed_codecs.begin(), allowed_codecs.end(), in_codec) != allowed_codecs.end();
            }
            if (use_hw) {
                hwaccel = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
            }
        }
        av::Dictionary options;
        if (params.count("options")) {
            options = parametersToDict(params["options"]);
        }
        std::shared_ptr<Child> r = std::make_shared<Child>(src_edge->makeSource(), dst_edge->makeSink(), md->source_stream, codec_name, options, pixel_format, hwaccel);
        if (params.count("flush_magic")) {
            r->flush_magic_ = params["flush_magic"];
        }
        if (params.count("waiting_for_frame")) {
            r->waiting_for_frame_ = params["waiting_for_frame"];
        }
        if (params.count("hold_at_eof")) {
            r->hold_at_eof_ = params["hold_at_eof"];
        }
        // Every decoder reports its health to the instance's ledger; fixed pools are also counted there.
        r->pool_ledger_ = InstanceSharedObjects<avp::DecoderPoolLedger>::get(nci.instance, "decoder_pools");
        if (r->surface_gate_) {
            r->pool_budget_bytes_ = params.value("pool_budget_mib", int64_t(0)) << 20;
            if (params.value("park", false)) {
                r->park_limit_ = params.value("park_limit", 8);
                if (r->park_limit_ < 1) throw Error("decoder: park_limit must be positive");
                r->parking_ = InstanceSharedObjects<avp::DecoderParking>::get(nci.instance, "decoder_parking");
            }
        } else if (params.contains("park")) {
            throw Error("decoder: park applies to CUarray decoders with unsafe_output only");
        }
        return r;
    }
};


class VideoDecoder: public Decoder<VideoDecoder, av::VideoDecoderContext, av::VideoFrame>, public IVideoFormatSource, public IFrameRateSource {
public:
    using Decoder::Decoder;
    virtual int width() {
        return this->dec_.width();
    }
    virtual int height() {
        return this->dec_.height();
    }
    virtual av::PixelFormat pixelFormat() {
        av::PixelFormat pf = this->dec_.pixelFormat();
        if (pf==AV_PIX_FMT_NONE) {
            pf = realPixelFormat();
            logstream << "decoder pixel format none, using stream pixel format " << pf;
        }
        return pf;
    }
    virtual av::PixelFormat realPixelFormat() {
        AVCodecContext* cc = this->dec_.raw();
        if (cc->sw_pix_fmt != AV_PIX_FMT_NONE) {
            return cc->sw_pix_fmt;
        } else {
            logstream << "realPixelFormat(): sw_pix_fmt none, falling back to stream pixel format";
        }
        #if 0
        if (cc && cc->hw_frames_ctx && cc->hw_frames_ctx->data) {
            AVHWFramesContext *frmctx = (AVHWFramesContext *)(cc->hw_frames_ctx->data);
            logstream << "have hw frames context in decoder, sw_format " << av::PixelFormat(frmctx->sw_format);
            if (frmctx->sw_format != AV_PIX_FMT_NONE) {
                return frmctx->sw_format;
            } else {
                logstream << "falling back to stream pixel format";
            }
        }
        #endif
        return av::PixelFormat(static_cast<AVPixelFormat>(this->dec_.stream().raw()->codecpar->format));
    }
    virtual av::Rational frameRate() {
        return this->dec_.stream().frameRate();
    }
};
class AudioDecoder: public Decoder<AudioDecoder, av::AudioDecoderContext, av::AudioSamples>, public IAudioMetadataSource {
public:
    using Decoder::Decoder;
    virtual int sampleRate() {
        return this->dec_.sampleRate();
    }
    virtual av::SampleFormat sampleFormat() {
        return this->dec_.sampleFormat();
    }
    virtual uint64_t channelLayout() {
        return this->dec_.channelLayout();
    }
};

DECLNODE(dec_video, VideoDecoder);
DECLNODE(dec_audio, AudioDecoder);
