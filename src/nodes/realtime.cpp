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
public:
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
            logstream << "realtime team changing offset by " << (local_offset-offset) << "ms";
            offset_.store(local_offset, std::memory_order_relaxed);
            return local_offset;
        } else {
            logstream << "realtime team ignoring offset diff " << (local_offset-offset) << "ms";
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
                logstream << "getting offset from team diff " << (r-local_offset) << "ms";
            } else if (r > local_offset) {
                logstream << "STRANGE: local offset smaller than team offset by " << (r-local_offset) << "ms";
            }
        }
        return r!=AV_NOPTS_VALUE ? r : local_offset;
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
    AVTS negative_time_tolerance_ = -250;
    AVTS negative_time_discard_;
    AVTS discontinuity_threshold_ = 1000;
    AVTS jitter_margin_ = 0;
    AVTS initial_jitter_margin_ = 0;
    AVRational tb_to_rescale_ts_ = wallclock.timeBase();
    std::shared_ptr<RealTimeTeam> team_;
    bool is_master_ = true; // by default everyone is master and can resync
    // TODO: master election in case of failure of master specified by user
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) {
        bool process_next;
        do {
            process_next = false;
            bool emit = true;
            bool consume = true;
            T* dataptr = this->source_->peek(0);
            if (dataptr==nullptr) {
                if (!ticks) {
                    // TODO write some wrapper function or macro because copy-pasting the whole weak_ptr->lambda->shared_ptr logic is tedious
                    std::weak_ptr<RealTimeSpeed> wthis(this->thisAsShared());
                    //logstream << "scheduling source retry";
                    evl.asyncWaitAndExecute(this->edgeSource()->edge()->producedEvent(), [wthis](EventLoop& evl) {
                        //logstream << "edgeSource wakeup";
                        std::shared_ptr<RealTimeSpeed> sthis = wthis.lock();
                        if (!sthis) return;
                        // retry when we have packet in source queue
                        sthis->processNonBlocking(evl, false);
                    });
                }
                return;
            }
            T &data = *dataptr;

            AVTS now_ts = wallclock.pts();
            
            AVTS pkt_ts = TSGetter<T>::get(data, tb_to_rescale_ts_);
            if ( (pkt_ts != AV_NOPTS_VALUE) && (pkt_ts != (AV_NOPTS_VALUE+1)) ) { // FIXME: why +1 ???
                if (team_) {
                    if (ready_) {
                        offset_ = team_->getOffset(offset_);
                    } else {
                        offset_ = team_->getOffset();
                        ready_ = offset_ != AV_NOPTS_VALUE; // if offset was initialized by a member of our team, trust it
                    }
                }
                if (ready_) {
                    AVTS diff = (pkt_ts - offset_) - now_ts;
                    if (diff < negative_time_tolerance_) {
                        logstream << "negative time to wait " << diff << "ms, resyncing.";
                        ready_ = false;
                    } else if (diff < negative_time_discard_) {
                        logstream << "negative time to wait " << diff << "ms, discarding frame.";
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
                                    //logstream << "need to wait " << diff;
                                    std::weak_ptr<RealTimeSpeed> wthis(this->thisAsShared());
                                    evl.schedule(av::Timestamp(now_ts + diff, wallclock.timeBase()), [wthis](EventLoop& evl) {
                                        //logstream << "wait wakeup";
                                        std::shared_ptr<RealTimeSpeed> sthis = wthis.lock();
                                        if (!sthis) return;
                                        // retry after waiting
                                        sthis->processNonBlocking(evl, false);
                                    });
                                }
                            }
                        } else {
                            if (!no_wait_notified_) {
                                logstream << "input queued constantly for " << (now_ts-last_wait_) << "ms, bypassing.";
                                no_wait_notified_ = true;
                            }
                            ready_ = false;
                            first_ = true;
                        }
                    } else {
                        // diff >= discontinuity_threshold_
                        logstream << "discontinuity detected, " << diff << "ms, resyncing.";
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
                        std::weak_ptr<RealTimeSpeed> wthis(this->thisAsShared());
                        evl.asyncWaitAndExecute(this->edgeSink()->edge()->consumedEvent(), [wthis](EventLoop& evl) {
                            std::shared_ptr<RealTimeSpeed> sthis = wthis.lock();
                            if (!sthis) return;
                            // retry when we have space in sink
                            sthis->processNonBlocking(evl, false);
                        });
                    }
                    consume = false;
                } else {
                    if (!ticks) {
                        std::weak_ptr<RealTimeSpeed> wthis(this->thisAsShared());
                        evl.execute([wthis](EventLoop& evl) {
                            std::shared_ptr<RealTimeSpeed> sthis = wthis.lock();
                            if (!sthis) return;
                            // process next packet
                            sthis->processNonBlocking(evl, false);
                        });
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
        if (params.count("leak_after")) {
            r->max_no_wait_period_ = wallclock.secondsToAVTS(params["leak_after"]);
        }
        if (params.count("speed")) {
            float speed = params["speed"];
            AVRational tb = wallclock.timeBase();
            r->tb_to_rescale_ts_ = { tb.num, int(float(tb.den)/speed + 0.5f) };
        }
        if (params.count("negative_time_tolerance")) {
            r->negative_time_tolerance_ = -wallclock.secondsToAVTS(params["negative_time_tolerance"]);
        }
        if (params.count("negative_time_discard")) {
            r->negative_time_discard_ = -wallclock.secondsToAVTS(params["negative_time_discard"]);
        } else {
            r->negative_time_discard_ = r->negative_time_tolerance_;
        }
        if (params.count("discontinuity_threshold")) {
            r->discontinuity_threshold_ = wallclock.secondsToAVTS(params["discontinuity_threshold"]);
        }
        if (params.count("jitter_margin")) {
            r->jitter_margin_ = wallclock.secondsToAVTS(params["jitter_margin"]);
        }
        if (params.count("initial_jitter_margin")) {
            r->initial_jitter_margin_ = wallclock.secondsToAVTS(params["initial_jitter_margin"]);
        } else {
            r->initial_jitter_margin_ = r->jitter_margin_;
        }
        if (params.count("team")) {
            r->team_ = InstanceSharedObjects<RealTimeTeam>::get(nci.instance, params["team"]);
        }
        if (params.count("master")) {
            r->is_master_ = params["master"];
        }
        return r;
    }
};

DECLNODE_ATD(realtime, RealTimeSpeed);
