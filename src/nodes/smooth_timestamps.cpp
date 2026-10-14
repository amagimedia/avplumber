#include "node_common.hpp"
#include "../mixer/primitives/MonotonicClock.hpp"
#include "../mixer/primitives/PaintStamping.hpp"
#include <cmath>
#include <deque>
#include <iomanip>
#include <optional>
#include <sstream>

// smooth_timestamps:
// - overwrites PTS with a "smoothed" monotonic timeline built from previous PTS + duration
// - resyncs to input timestamps when averaged drift grows too large (to maintain A/V sync)
//
// Video:
// - duration is user-controlled (as rational seconds), typically 1/FPS
// - output timebase == duration, and PTS increments by 1 each frame
// - with `latency_ms` (a source whose consumer draws slot k at time(k) + latency on the monotonic
//   clock the stamps are on, e.g. the mixer's browser chain): the count is corrected against that
//   deadline instead of the mean drift (PaintStamping): a frame within `guard_ms` of missing its
//   tick moves the numbering one slot later, once; a numbering more than `early_slots` ahead of the
//   clock for `early_frames` frames moves one slot back. `stamping` (node.object.get) counts them.
//
// Audio:
// - duration is derived from frame sample count and sample rate
// - output timebase == 1/sample_rate, and PTS increments by samplesCount

template<typename T>
class SmoothTimestamps : public NodeSISO<T, T>,
                         public NonBlockingNode<SmoothTimestamps<T>>,
                         public IInputReset,
                         public ITimeBaseSource,
                         public IFrameRateSource,
                         public IReturnsObjects {
protected:
    bool started_ = false;
    // Video with `latency_ms`: the numbering against the playout deadline, instead of the
    // counted timeline and mean-drift resync below.
    std::optional<avp::mixer::PaintStamping> paint_;
    uint64_t shift_events_ = 0;      // frames whose numbering was moved; the first and every 100th are logged
    // PTS to use for the next output frame (in current output timebase).
    // This makes the first emitted frame equal to (rescaled) first input PTS (or 0).
    av::Timestamp next_out_pts_ = NOTS;
    av::Timestamp last_in_pts_ = NOTS;

    // configuration
    av::Rational video_duration_tb_{0, 0}; // seconds per tick (video only)
    double resync_threshold_sec_ = 0.02;   // resync when |avg(drift)| exceeds this
    int min_samples_before_resync_ = 150;   // ignore early timestamps noise
    double discontinuity_sec_ = 2.0;       // hard reset if input discontinuity exceeds this
    // Video: when (re)aligning the smoothed timeline to an input timestamp, take
    // the next output tick instead of the nearest one. A consumer that renders
    // slot k at tick(k)+latency then always receives the frame before its tick,
    // instead of up to half a tick after it when the input's phase rounded down.
    bool round_up_ = false;

    // drift estimation (sliding window)
    int drift_window_ = 300;               // samples
    std::deque<double> drift_window_sec_; // last N drift samples (seconds)
    double drift_sum_sec_ = 0.0;

    static constexpr bool IsVideo = std::is_same_v<T, av::VideoFrame>;
    static constexpr bool IsAudio = std::is_same_v<T, av::AudioSamples>;

    av::Timestamp alignToInput(const av::Timestamp in_pts, const av::Rational out_tb) const {
        if (!round_up_ || !in_pts.isValid()) return rescaleTS(in_pts, out_tb);
        return { av_rescale_q_rnd(in_pts.timestamp(), in_pts.timebase().getValue(), out_tb.getValue(), AV_ROUND_UP), out_tb };
    }

    av::Rational currentOutTimebaseFor(const T& frame) const {
        if constexpr (IsVideo) {
            return video_duration_tb_;
        } else if constexpr (IsAudio) {
            int sr = frame.sampleRate();
            if (sr <= 0 && frame.raw()) {
                sr = frame.raw()->sample_rate;
            }
            if (sr <= 0) {
                // fallback: keep incoming timebase if it exists, otherwise "invalid"
                av::Rational tb = frame.pts().timebase();
                return tb;
            }
            return {1, sr};
        } else {
            return {0, 0};
        }
    }

    av::Timestamp deltaFor(const T& frame, const av::Rational out_tb) const {
        if constexpr (IsVideo) {
            (void)frame;
            return av::Timestamp(1, out_tb);
        } else if constexpr (IsAudio) {
            int64_t n = frame.samplesCount();
            if (n <= 0) {
                // best effort: treat as 0-length, avoid moving timeline
                n = 0;
            }
            return av::Timestamp(n, out_tb);
        } else {
            (void)frame;
            return NOTS;
        }
    }

    void resetState() {
        started_ = false;
        next_out_pts_ = NOTS;
        last_in_pts_ = NOTS;
        drift_window_sec_.clear();
        drift_sum_sec_ = 0.0;
        if (paint_) paint_->reset();
    }

    /// Video with `latency_ms`: number the frame against the deadline and hand it on. False when
    /// the sink has no room; the numbering is then left as it was, for the retry with the same frame.
    bool stampByDeadline(T &frame, const av::Timestamp in_pts, const bool in_pts_valid, const av::Rational out_tb) {
        const int64_t now_ns = avp::mixer::monotonicNs();
        const int64_t in_ns = in_pts_valid
            ? av_rescale_q(in_pts.timestamp(), in_pts.timebase().getValue(), AVRational{1, 1000000000}) : 0;
        const avp::mixer::PaintStamping saved = *paint_;
        const auto stamped = paint_->stamp(in_pts_valid, in_ns, now_ns);
        const av::Timestamp orig_pts = frame.pts();
        frame.setTimeBase(av::Rational());
        frame.setPts(av::Timestamp(stamped.slot, out_tb));
        if (!this->sink_->put(frame, true)) {
            frame.setTimeBase(av::Rational());
            frame.setPts(orig_pts);
            *paint_ = saved;
            return false;
        }
        if (stamped.shifted && (++shift_events_ == 1 || shift_events_ % 100 == 0)) {
            const auto &st = paint_->stats();
            std::ostringstream line;
            line << std::fixed << std::setprecision(1) << "smooth_timestamps: ";
            if (stamped.shifted > 0)
                line << "a frame within the guard of its slot's deadline: numbering moved " << stamped.shifted
                     << " slot(s) later";
            else
                line << "frames running more than " << paint_->config().early_ns / 1e6
                     << " ms ahead of their slots: numbering moved one slot back";
            line << " (slot " << stamped.slot << ", margin now " << paint_->marginNs(stamped.slot, now_ns) / 1e6
                 << " ms; " << shift_events_ << " frame(s) moved so far, " << st.shifts_later << " slot(s) later, "
                 << st.shifts_back << " back)";
            logstream << line.str();
        }
        return true;
    }

public:
    using NodeSISO<T, T>::NodeSISO;

    void resetInput() override {
        resetState();
    }

    Parameters getObject(const std::string key) override {
        if (key != "stamping") throw Error("smooth_timestamps: unknown object " + key);
        Parameters result = {{"mode", paint_ ? "deadline" : "drift"}};
        if (paint_) {
            const auto &st = paint_->stats();
            const auto &cfg = paint_->config();
            result["shifts_later"] = st.shifts_later;
            result["shifts_back"] = st.shifts_back;
            result["realignments"] = st.realignments;
            result["shift_events"] = shift_events_;
            result["next_slot"] = paint_->nextSlot();
            result["latency_ms"] = cfg.latency_ns / 1e6;
            result["guard_ms"] = cfg.guard_ns / 1e6;
            result["early_ms"] = cfg.early_ns / 1e6;
            result["early_frames"] = cfg.early_frames;
        }
        return result;
    }

    av::Rational timeBase() override {
        if constexpr (IsVideo) {
            return video_duration_tb_;
        } else {
            // for audio we only know after first frame; return "unknown" until then
            if (next_out_pts_.isValid()) {
                return next_out_pts_.timebase();
            }
            return {0, 1};
        }
    }

    av::Rational frameRate() override {
        if constexpr (IsVideo) {
            if (video_duration_tb_.getNumerator() == 0 || video_duration_tb_.getDenominator() == 0) {
                return {0, 1};
            }
            // duration = seconds per tick; fps = 1/duration
            return {video_duration_tb_.getDenominator(), video_duration_tb_.getNumerator()};
        } else {
            return {0, 1};
        }
    }

    void processNonBlocking(EventLoop&, bool ticks) override {
        bool process_next;
        do {
            process_next = false;

            T* dataptr = this->source_->peek(0);
            if (dataptr == nullptr) {
                if (!ticks) {
                    this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                }
                return;
            }

            T& frame = *dataptr;

            // EOF marker: pass through untouched
            if (isEofMarker(frame)) {
                if (!this->sink_->put(frame, true)) {
                    if (!ticks) {
                        this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                    }
                    return;
                }
                this->source_->pop();
                if (!ticks) this->yieldAndProcess();
                else process_next = true;
                continue;
            }

            av::Timestamp in_pts = frame.pts();
            const bool in_pts_valid = in_pts.isValid();

            // establish output timebase and delta
            av::Rational out_tb = currentOutTimebaseFor(frame);
            if constexpr (IsVideo) {
                if (out_tb.getNumerator() == 0 || out_tb.getDenominator() == 0) {
                    throw Error("smooth_timestamps: video requires valid 'duration' or 'fps' parameter");
                }
                if (paint_) {
                    if (!stampByDeadline(frame, in_pts, in_pts_valid, out_tb)) {
                        if (!ticks) {
                            this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                        }
                        return;
                    }
                    this->source_->pop();
                    if (!ticks) this->yieldAndProcess();
                    else process_next = true;
                    continue;
                }
            }

            // initialize state on first frame (prefer syncing to input PTS if available)
            if (!started_) {
                started_ = true;
                if (in_pts_valid) {
                    next_out_pts_ = alignToInput(in_pts, out_tb);
                    last_in_pts_ = in_pts;
                } else {
                    next_out_pts_ = av::Timestamp(0, out_tb);
                    last_in_pts_ = NOTS;
                }
                drift_window_sec_.clear();
                drift_sum_sec_ = 0.0;
            }

            // if output timebase changes (audio sample_rate change), keep next_out_pts_ consistent
            if (next_out_pts_.isValid() && next_out_pts_.timebase() != out_tb) {
                next_out_pts_ = rescaleTS(next_out_pts_, out_tb);
            }

            // detect big input discontinuity (hard reset to input)
            if (in_pts_valid && last_in_pts_.isValid()) {
                double in_jump = (in_pts - last_in_pts_).seconds();
                if (std::fabs(in_jump) > discontinuity_sec_) {
                    next_out_pts_ = alignToInput(in_pts, out_tb);
                    drift_window_sec_.clear();
                    drift_sum_sec_ = 0.0;
                }
            }

            av::Timestamp delta = deltaFor(frame, out_tb);
            av::Timestamp out_pts = next_out_pts_;

            // update drift averaging window
            if (in_pts_valid) {
                double drift_sec = addTS(in_pts, negateTS(out_pts)).seconds();
                if (drift_window_ > 0) {
                    drift_window_sec_.push_back(drift_sec);
                    drift_sum_sec_ += drift_sec;
                    while ((int)drift_window_sec_.size() > drift_window_) {
                        drift_sum_sec_ -= drift_window_sec_.front();
                        drift_window_sec_.pop_front();
                    }
                }
            }

            if (in_pts_valid && drift_window_ > 0 && (int)drift_window_sec_.size() >= min_samples_before_resync_) {
                const double drift_avg_sec = drift_sum_sec_ / (double)drift_window_sec_.size();
                if (std::fabs(drift_avg_sec) > resync_threshold_sec_) {
                    out_pts = alignToInput(in_pts, out_tb);
                    drift_window_sec_.clear();
                    drift_sum_sec_ = 0.0;
                }
            }

            // advance next PTS after deciding what to emit for this frame
            next_out_pts_ = addTS(out_pts, delta);

            av::Timestamp orig_pts = frame.pts();

            // apply PTS overwrite
            frame.setTimeBase(av::Rational());
            frame.setPts(out_pts);

            if (!this->sink_->put(frame, true)) {
                // restore if backpressure; we'll re-process same frame later
                frame.setTimeBase(av::Rational());
                frame.setPts(orig_pts);
                if (!ticks) {
                    this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                }
                return;
            }

            // commit state and consume input
            last_in_pts_ = in_pts_valid ? in_pts : last_in_pts_;
            this->source_->pop();
            if (!ticks) this->yieldAndProcess();
            else process_next = true;
        } while (process_next);
    }

    static std::shared_ptr<SmoothTimestamps> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<T, T>::template createCommon<SmoothTimestamps<T>>(edges, params);

        if constexpr (IsVideo) {
            if (params.count("duration")) {
                r->video_duration_tb_ = parseRatio(params.at("duration"));
            } else if (params.count("fps")) {
                av::Rational fps = parseRatio(params.at("fps"));
                if (fps.getNumerator() == 0 || fps.getDenominator() == 0) {
                    throw Error("smooth_timestamps: invalid fps");
                }
                r->video_duration_tb_ = av::Rational(fps.getDenominator(), fps.getNumerator());
            }

            if (r->video_duration_tb_.getNumerator() <= 0 || r->video_duration_tb_.getDenominator() <= 0) {
                throw Error("smooth_timestamps: video requires 'duration' (seconds per frame, rational) or 'fps' (frames per second, rational)");
            }
        }

        if (params.count("drift_window")) {
            r->drift_window_ = params.at("drift_window").get<int>();
        }
        if (params.count("resync_threshold")) {
            r->resync_threshold_sec_ = params.at("resync_threshold").get<double>();
        }
        if (params.count("min_samples_before_resync")) {
            r->min_samples_before_resync_ = params.at("min_samples_before_resync").get<int>();
        }
        if (params.count("round_up")) {
            if constexpr (!IsVideo) throw Error("smooth_timestamps: round_up applies to video only");
            r->round_up_ = params.at("round_up").get<bool>();
        }
        if (params.count("discontinuity_threshold")) {
            r->discontinuity_sec_ = params.at("discontinuity_threshold").get<double>();
        }

        if (r->drift_window_ < 0) {
            throw Error("smooth_timestamps: drift_window must be >= 0 (0 disables drift-based resync)");
        }
        if (r->resync_threshold_sec_ < 0.0) {
            throw Error("smooth_timestamps: resync_threshold must be >= 0");
        }
        if (r->min_samples_before_resync_ < 0) {
            throw Error("smooth_timestamps: min_samples_before_resync must be >= 0");
        }
        if (r->discontinuity_sec_ < 0.0) {
            throw Error("smooth_timestamps: discontinuity_threshold must be >= 0");
        }

        if (params.count("latency_ms")) {
            if constexpr (!IsVideo) throw Error("smooth_timestamps: latency_ms applies to video only");
            const double latency_ms = params.at("latency_ms").get<double>();
            const double guard_ms = params.value("guard_ms", 10.0);
            const double early_slots = params.value("early_slots", 1.5);
            const int early_frames = params.value("early_frames", 30);
            if (!(latency_ms > 0.0) || !(guard_ms >= 0.0) || !(guard_ms < latency_ms) || !(early_slots > 0.0) || early_frames < 1)
                throw Error("smooth_timestamps: latency_ms must be above 0, guard_ms from 0 to below latency_ms, "
                            "early_slots above 0 and early_frames at least 1");
            const double period_ms = 1000.0 * r->video_duration_tb_.getNumerator() / r->video_duration_tb_.getDenominator();
            avp::mixer::PaintStamping::Config cfg;
            cfg.fps = r->frameRate().getValue();
            cfg.latency_ns = std::llround(latency_ms * 1e6);
            cfg.guard_ns = std::llround(guard_ms * 1e6);
            cfg.early_ns = std::llround(early_slots * period_ms * 1e6);
            cfg.early_frames = early_frames;
            cfg.discontinuity_ns = std::llround(r->discontinuity_sec_ * 1e9);
            r->paint_.emplace(cfg);
        }

        return r;
    }
};

DECLNODE_ATD_RAW(smooth_timestamps, SmoothTimestamps);


