#include "node_common.hpp"

template<typename T> class ForceFPS: public NonBlockingNode<ForceFPS<T>>, public NodeSISO<T, T>, public IFrameRateSource, public ITimeBaseSource {
private:
    av::Rational fps_;
    av::Timestamp frame_delta_;
    av::Timestamp last_ts_ = NOTS;
    av::Timestamp next_ts_ = NOTS;
    T last_frame_;
    av::Rational timebase_;
    bool frame_wasted_ = false;
    int dropped_ = 0;
    int duplicated_ = 0;
    int total_out_ = 0;
    int total_in_ = 0;
    av::Timestamp last_printed_stats_ = NOTS;
    void setLast(T &frm, bool wasted) {
        if (frame_wasted_) {
            // if previous frame wasn't used and we have to overwrite it, increase dropped frames count
            dropped_++;
        }
        last_frame_ = frm;
        frame_wasted_ = wasted;
    }
public:
    ForceFPS(std::unique_ptr<Source<T>> &&source, std::unique_ptr<Sink<T>> &&sink, const av::Rational fps, const av::Rational timebase): NodeSISO<T, T>(std::move(source), std::move(sink)), fps_(fps), timebase_(timebase) {
        if (timebase_.getDenominator()==0 || timebase_.getNumerator()==0) {
            timebase_ = av::Rational(fps_.getDenominator(), fps_.getNumerator());
            frame_delta_ = av::Timestamp(1, timebase_);
        } else {
            frame_delta_ = rescaleTS(av::Timestamp(1, {fps_.getDenominator(), fps_.getNumerator()}), timebase_);
        }
        logstream << "Set timebase " << timebase_ << ", frame rate " << fps_ << ", frame delta " << frame_delta_;
    }
    virtual void processNonBlocking(EventLoop& evl, bool ticks) override {
        bool process_next;
        do {
            process_next = false;
            T* ptr = this->source_->peek(0);
            if (ptr==nullptr) {
                // no data available in queue
                if (!ticks) {
                    // retry when we have packet in source queue
                    this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                }
                return;
            }
            
            T &pkt = *ptr;
            if (!pkt.isValid()) {
                process_next = true;
                continue;
            }
            
            if (last_printed_stats_.isNoPts()) {
                last_printed_stats_ = wallclock.ts();
            }
            
            av::Timestamp in_ts = pkt.pts();
            if ((in_ts.timebase() != timebase_) || (pkt.timeBase() != timebase_)) {
                pkt.setTimeBase(timebase_);
                in_ts = rescaleTS(in_ts, timebase_);
            }
            if (last_ts_.isValid()) {
                av::Timestamp delta_from_last = in_ts - last_ts_;
                bool discontinuity = (delta_from_last.seconds() > 0.5) || (delta_from_last.timestamp() < 0);
                if (discontinuity) {
                    logstream << "Discontinuity " << last_ts_ << " -> " << in_ts;
                }
                /*if (delta_from_last == frame_delta_) {
                    logstream << "Frame PTS = " << in_ts << " perfectly aligned";
                }*/
                if (! ( (delta_from_last == frame_delta_) || discontinuity ) ) {
                    // this scope is NOT executed if frame PTS is perfectly aligned
                    // or discontinuity occured
                    // because we don't want to duplicate or drop frames then
                    if (in_ts > next_ts_) {
                        // PTS too big
                        // input frame rate too low
                        // duplicate previous frame
                        while (in_ts > next_ts_) {
                            if (!frame_wasted_) {
                                // increase number of duplicated frames only if last_frame_ was already used
                                duplicated_++;
                            }
                            frame_wasted_ = false;
                            last_frame_.setPts(next_ts_);
                            total_out_++;

                            if (!this->sink_->put(last_frame_, true)) {
                                if (!ticks) {
                                    // retry when we have space in sink
                                    this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                                }
                                return;
                            }
                            next_ts_ = addTS(next_ts_, frame_delta_);
                        }
                        // now in_ts <= next_ts_
                        //  if in_ts == next_ts, all OK
                        //  if in_ts < next_ts_, it means that in_ts is too small and unaligned
                        //   aligning it is dangerous, so...
                        //   so continue to  the "PTS too small" logic!
                        //in_ts = next_ts_; // align
                        /*if (in_ts != next_ts_) {
                            logstream << " :/ unaligned";
                        } else {
                            logstream << " :) aligned";
                        }*/
                    }
                    if (in_ts < next_ts_) {
                        // PTS too small
                        // input frame rate too high
                        // save received frame as last_frame_
                        // and drop it!
                        setLast(pkt, true);
                        total_in_++;
                        this->source_->pop();
                        //logstream << "Dropping frame PTS = " << in_ts;
                        if (ticks) {
                            process_next = true;
                            continue;
                        } else {
                            this->yieldAndProcess();
                            break;
                        }
                    }
                }
            }
            last_ts_ = in_ts;
            next_ts_ = addTS(in_ts, frame_delta_);
            pkt.setPts(in_ts);
            
            if (!this->sink_->put(pkt, true)) {
                if (!ticks) {
                    // retry when we have space in sink
                    this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                }
                return;
            } else {
                setLast(pkt, false);
            }
            total_out_++;
            this->source_->pop();
            total_in_++;

            process_next = ticks;
            if (!ticks) {
                this->yieldAndProcess();
            }
        } while (process_next);

        av::Timestamp now = wallclock.ts();
        const float print_stats_every = 10;
        if ( (now-last_printed_stats_).seconds()>print_stats_every ) {
            if (dropped_>0 || duplicated_>0) {
                logstream << "statistics: in " << total_in_ << ", out " << total_out_ << ", duplicated " << duplicated_ << ", dropped " << dropped_ << ", est. input FPS " << (fps_.getDouble() * (double)total_in_ / (double)total_out_);
                last_printed_stats_ = now;
            }
            /*dropped_ = 0;
            duplicated_ = 0;
            total_in_ = 0;
            total_out_ = 0;*/
        }
    }
    virtual av::Rational frameRate() {
        return fps_;
    }
    virtual av::Rational timeBase() {
        return timebase_;
    }
    static std::shared_ptr<ForceFPS> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        av::Rational fps = parseRatio(params.at("fps"));
        av::Rational timebase = {0, 0};
        if (params.count("timebase")==1) {
            timebase = parseRatio(params["timebase"]);
        }
        return NodeSISO<T, T>::template createCommon<ForceFPS<T>>(edges, params, fps, timebase);
    }
};

class ForceVideoFPS: public ForceFPS<av::VideoFrame> {};

DECLNODE(force_fps, ForceVideoFPS);
