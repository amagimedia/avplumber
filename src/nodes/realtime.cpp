#include "node_common.hpp"
#include "../instance_shared.hpp"
#include "../EventLoop.hpp"

class RealTimeTeam: public InstanceShared<RealTimeTeam> {
protected:
    std::atomic<AVTS> offset_{AV_NOPTS_VALUE};
    std::mutex busy_;
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    AVRational timebase_ = {0, 0};
    std::atomic_bool flushing_ = false;
public:
    void checkTimeBase(AVRational tb) {
        auto lock = getLock();
        if (timebase_.num==0 && timebase_.den==0) {
            timebase_ = tb;
        } else {
            if ((tb.num != timebase_.num) || (tb.den != timebase_.den)) {
                throw Error("all realtime nodes in a team must have the same timebase (tick_source)");
            }
        }
    }
    AVTS updateOffset(AVTS local_offset) {
        auto lock = getLock();
        AVTS offset = offset_.load(std::memory_order_relaxed);
        // std::memory_order_relaxed because mutexed anyway
        if (offset == AV_NOPTS_VALUE) {
            offset_.store(local_offset, std::memory_order_relaxed);
            return local_offset;
        }
        // we want to synchronize to the smallest offset because it ensures that sufficient data is buffered
        // for smooth playback of all streams
        if (local_offset < offset) {
            logstream << "realtime team changing offset by " << (local_offset-offset);
            offset_.store(local_offset, std::memory_order_relaxed);
            return local_offset;
        } else {
            logstream << "realtime team ignoring offset diff " << (local_offset-offset);
            return offset;
        }
    }
    void reset() {
        auto lock = getLock();
        offset_.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        logstream << "realtime team reset";
    }
    AVTS getOffset(AVTS local_offset = AV_NOPTS_VALUE) {
        AVTS r = offset_.load(std::memory_order_acquire);
        if ((local_offset != AV_NOPTS_VALUE) && (r != AV_NOPTS_VALUE)) {
            if (r < local_offset) {
                logstream << "getting offset from team diff " << (r-local_offset);
            } else if (r > local_offset) {
                logstream << "STRANGE: local offset smaller than team offset by " << (r-local_offset);
            }
        }
        //return r!=AV_NOPTS_VALUE ? r : local_offset;
        return r;
    }
    void startFlushing() {
        flushing_ = true;
    }
    void stopFlushing() {
        flushing_ = false;
    }
    bool isFlushing() {
        return flushing_;
    }
};

template <typename T> class RealTimeSpeed: public NodeSISO<T, T>, public NonBlockingNode<RealTimeSpeed<T>> {
protected:
    bool ready_ = false;
    bool first_ = true;
    bool no_wait_notified_ = false;
    AVTS max_no_wait_period_ = INT64_MAX;
    AVTS last_wait_ = 0;
    AVTS offset_;
    AVTS tick_period_ = 0;
    AVTS now_ts_ = AV_NOPTS_VALUE;
    AVTS negative_time_tolerance_ = -250;
    AVTS negative_time_discard_ = AV_NOPTS_VALUE;
    AVTS discontinuity_threshold_ = 1000;
    AVTS jitter_margin_ = 0;
    AVTS initial_jitter_margin_ = 0;
    AVRational timebase_;
    AVRational tb_to_rescale_ts_;
    uint64_t tick_drifted_for_ = 0;
    std::shared_ptr<RealTimeTeam> team_;
    std::shared_ptr<EdgeBase> input_ts_queue_;
    std::list<std::shared_ptr<EdgeBase>> intermediate_queues_;
    float max_buffered_ = 5.5;
    float min_buffered_ = 0.5;
    bool is_master_ = true; // by default everyone is master and can resync
    // TODO: master election in case of failure of master specified by user

    std::string printDuration(AVTS duration) {
        if (duration==AV_NOPTS_VALUE) {
            return "NOTS";
        }
        return std::to_string(duration) +
            ((timebase_.num==1 && timebase_.den==1000) ? "ms" : ("*"+std::to_string(timebase_.num)+"/"+std::to_string(timebase_.den)));
    }
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) {
        AVTS now_ts_wclk = wallclock.pts();
        AVTS now_ts_wclk_scaled = rescaleTS({now_ts_wclk, wallclock.timeBase()}, timebase_).timestamp();
        if (now_ts_ == AV_NOPTS_VALUE) {
            now_ts_ = now_ts_wclk_scaled;
        } else if (tick_period_ && ticks) {
            now_ts_ += tick_period_;
            AVTS drift = now_ts_ - now_ts_wclk_scaled;
            if (abs(drift) >= tick_period_) {
                tick_drifted_for_++;
                if (tick_drifted_for_ >= 240) {
                    logstream << "tick clock drifted from wallclock by " << printDuration(drift) << ", resyncing tick clock";
                    now_ts_ = now_ts_wclk_scaled;
                    tick_drifted_for_ = 0;
                }
            } else {
                tick_drifted_for_ = 0;
            }
        } else {
            now_ts_ = now_ts_wclk_scaled;
        }
        
        bool process_next;
        do {
            process_next = false;
            bool emit = true;
            bool consume = true;
            T* dataptr = this->source_->peek(0);
            if (dataptr==nullptr) {
                if (!ticks) {
                    // retry when we have packet in source queue
                    this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                }
                return;
            }
            T &data = *dataptr;
            
            AVTS now_ts = now_ts_;
            AVTS pkt_ts = TSGetter<T>::get(data, tb_to_rescale_ts_);
            if ( (pkt_ts != AV_NOPTS_VALUE) && (pkt_ts != (AV_NOPTS_VALUE+1)) ) { // FIXME: why +1 ???
                if (input_ts_queue_ != nullptr) {
                    av::Timestamp input_ts = input_ts_queue_->lastTS();
                    bool anything_buffered = input_ts_queue_->occupied() > 0;
                    for (auto q: intermediate_queues_) {
                        if (anything_buffered) break;
                        anything_buffered |= (q->occupied() > 0);
                    }
                    if (input_ts.isValid() && team_) {
                        float buffered = anything_buffered ? addTS(input_ts, negateTS(TSGetter<T>::getWithTB(data))).seconds() : 0;
                        if ((buffered > max_buffered_) || (team_->isFlushing() && (buffered > min_buffered_))) {
                            if (!team_->isFlushing()) {
                                logstream << "too many seconds buffered: " << buffered << " > " << max_buffered_ << ", flushing";
                                team_->startFlushing();
                            }
                            ready_ = false;
                            first_ = true;
                            this->source_->pop();
                            this->yieldAndProcess();
                            return;
                        } else {
                            if (team_->isFlushing()) {
                                logstream << "done flushing";
                                if (is_master_) {
                                    team_->reset();
                                }
                                team_->stopFlushing();
                            }
                        }
                    }
                }
                if (emit && team_) {
                    if (ready_) {
                        offset_ = team_->getOffset(offset_);
                    } else {
                        offset_ = team_->getOffset();
                    }
                    ready_ = offset_ != AV_NOPTS_VALUE; // if offset was initialized by a member of our team, trust it
                }
                if (ready_) {
                    AVTS diff = (pkt_ts - offset_) - now_ts;
                    if (diff < negative_time_tolerance_) {
                        logstream << "negative time to wait " << printDuration(diff) << ", resyncing.";
                        ready_ = false;
                    } else if (diff < negative_time_discard_) {
                        logstream << "negative time to wait " << printDuration(diff) << ", discarding frame.";
                        emit = false;
                    } else if (diff < discontinuity_threshold_) {
                        if (now_ts - last_wait_ < max_no_wait_period_) {
                            if (no_wait_notified_) {
                                logstream << "input no longer queued, returning to realtime mode.";
                                no_wait_notified_ = false;
                            }
                            if (diff > 0) {
                                emit = false;
                                consume = false;
                                if (!ticks) {
                                    // retry after waiting
                                    this->scheduleProcess(av::Timestamp(now_ts + diff, timebase_));
                                }
                            }
                        } else {
                            if (!no_wait_notified_) {
                                logstream << "input queued constantly for " << printDuration(now_ts-last_wait_) << ", bypassing.";
                                no_wait_notified_ = true;
                            }
                            ready_ = false;
                            first_ = true;
                        }
                    } else {
                        // diff >= discontinuity_threshold_
                        logstream << "discontinuity detected, " << printDuration(diff) << ", resyncing.";
                        if (team_ && is_master_) {
                            team_->reset();
                        }
                        ready_ = false;
                        first_ = true;
                    }
                }
                if ((!ready_) && consume) {
                    // offset_ must be set only when consuming, because it marks sync point
                    offset_ = (pkt_ts - now_ts) - (first_ ? initial_jitter_margin_ : jitter_margin_);
                    first_ = false;
                    if (team_) {
                        offset_ = team_->updateOffset(offset_);
                    }
                    ready_ = true;
                }
            }
            if (emit) {
                if (!this->sink_->put(data, true)) {
                    if (!ticks) {
                        // retry when we have space in sink
                        this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                    }
                    consume = false;
                } else {
                    if (!ticks) {
                        // process next packet
                        this->yieldAndProcess();
                    } else {
                        process_next = true;
                    }
                }
            }
            if (consume) {
                this->source_->pop();

                // check whether there is next packet in the input queue 
                T* ptr = this->source_->peek(0);
                if (ptr==nullptr || last_wait_==0) {
                    // we'll probably wait for packet in next process() iteration
                    last_wait_ = wallclock.pts();
                }
            }
        } while (process_next);
    }
    static std::shared_ptr<RealTimeSpeed> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<RealTimeSpeed> r = NodeSISO<T, T>::template createCommon<RealTimeSpeed>(edges, params);
        AVRational timebase = wallclock.timeBase();
        if (params.count("tick_period")) {
            int tb_den_mult = 4;
            timebase = parseRatio(params["tick_period"]);
            r->tick_period_ = tb_den_mult;
            while ((tb_den_mult%2)==0 && (timebase.num%2)==0) {
                tb_den_mult /= 2;
                timebase.num /= 2;
            }
            timebase.den *= tb_den_mult;
            logstream << "tick period " << r->tick_period_ << ", timebase " << timebase.num << "/" << timebase.den;
        }
        r->timebase_ = timebase;
        r->tb_to_rescale_ts_ = timebase;
        auto secondsToTs = [timebase](float seconds) -> AVTS {
            return seconds*(float)timebase.den / (float)timebase.num + 0.5;
        };
        if (params.count("leak_after")) {
            r->max_no_wait_period_ = secondsToTs(params["leak_after"]);
        }
        if (params.count("speed")) {
            float speed = params["speed"];
            r->tb_to_rescale_ts_ = { timebase.num, int(float(timebase.den)/speed + 0.5f) };
        }
        if (params.count("negative_time_tolerance")) {
            r->negative_time_tolerance_ = -secondsToTs(params["negative_time_tolerance"]);
        }
        if (params.count("negative_time_discard")) {
            r->negative_time_discard_ = -secondsToTs(params["negative_time_discard"]);
        }
        if (r->negative_time_discard_ == AV_NOPTS_VALUE) {
            r->negative_time_discard_ = r->negative_time_tolerance_;
        }
        if (params.count("discontinuity_threshold")) {
            r->discontinuity_threshold_ = secondsToTs(params["discontinuity_threshold"]);
        }
        if (params.count("jitter_margin")) {
            r->jitter_margin_ = secondsToTs(params["jitter_margin"]);
        }
        if (params.count("initial_jitter_margin")) {
            r->initial_jitter_margin_ = secondsToTs(params["initial_jitter_margin"]);
        } else {
            r->initial_jitter_margin_ = r->jitter_margin_;
        }
        if (params.count("team")) {
            r->team_ = InstanceSharedObjects<RealTimeTeam>::get(nci.instance, params["team"]);
            r->team_->checkTimeBase(timebase);
        }
        if (params.count("master")) {
            r->is_master_ = params["master"];
        }
        if (params.count("input_ts_queue")) {
            r->input_ts_queue_ = edges.findAny(params["input_ts_queue"]);
            if (r->input_ts_queue_ == nullptr) {
                throw Error("input_ts_queue doesn't exist");
            }
        }
        if (params.count("intermediate_queues")) {
            for (const std::string &qname: jsonToStringList(params["intermediate_queues"])) {
                std::shared_ptr<EdgeBase> q = edges.findAny(qname);
                if (q==nullptr) {
                    throw Error("queue " + qname + " from intermediate_queues doesn't exist");
                }
                r->intermediate_queues_.push_back(q);
            }
        }
        if (params.count("max_buffered")) {
            r->max_buffered_ = params["max_buffered"];
        }
        if (params.count("min_buffered")) {
            r->min_buffered_ = params["min_buffered"];
        }
        return r;
    }
};

DECLNODE_ATD(realtime, RealTimeSpeed);
