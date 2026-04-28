#include "node_common.hpp"
#include <deque>

// smooth_timestamps:
// - overwrites PTS with a "smoothed" monotonic timeline built from previous PTS + duration
// - resyncs to input timestamps when averaged drift grows too large (to maintain A/V sync)
//
// Video:
// - duration is user-controlled (as rational seconds), typically 1/FPS
// - output timebase == duration, and PTS increments by 1 each frame
//
// Audio:
// - duration is derived from frame sample count and sample rate
// - output timebase == 1/sample_rate, and PTS increments by samplesCount

template<typename T>
class SmoothTimestamps : public NodeSISO<T, T>,
                         public NonBlockingNode<SmoothTimestamps<T>>,
                         public IInputReset,
                         public ITimeBaseSource,
                         public IFrameRateSource {
protected:
    bool started_ = false;
    // PTS to use for the next output frame (in current output timebase).
    // This makes the first emitted frame equal to (rescaled) first input PTS (or 0).
    av::Timestamp next_out_pts_ = NOTS;
    av::Timestamp last_in_pts_ = NOTS;

    // configuration
    av::Rational video_duration_tb_{0, 0}; // seconds per tick (video only)
    double resync_threshold_sec_ = 0.02;   // resync when |avg(drift)| exceeds this
    int min_samples_before_resync_ = 150;   // ignore early timestamps noise
    double discontinuity_sec_ = 2.0;       // hard reset if input discontinuity exceeds this

    // drift estimation (sliding window)
    int drift_window_ = 300;               // samples
    std::deque<double> drift_window_sec_; // last N drift samples (seconds)
    double drift_sum_sec_ = 0.0;

    static constexpr bool IsVideo = std::is_same_v<T, av::VideoFrame>;
    static constexpr bool IsAudio = std::is_same_v<T, av::AudioSamples>;

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
    }

public:
    using NodeSISO<T, T>::NodeSISO;

    void resetInput() override {
        resetState();
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
            }

            // initialize state on first frame (prefer syncing to input PTS if available)
            if (!started_) {
                started_ = true;
                if (in_pts_valid) {
                    next_out_pts_ = rescaleTS(in_pts, out_tb);
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
                    next_out_pts_ = rescaleTS(in_pts, out_tb);
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
                    out_pts = rescaleTS(in_pts, out_tb);
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

        return r;
    }
};

DECLNODE_ATD_RAW(smooth_timestamps, SmoothTimestamps);


