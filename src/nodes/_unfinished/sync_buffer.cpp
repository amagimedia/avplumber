#include "node_common.hpp"
#include <queue>
#include "../instance_shared.hpp"

class SyncBufferCommon: public InstanceShared<SyncBufferCommon> {
protected:
    bool ready_ = false;
    AVTS offset_;
    AVRational tb_to_rescale_ts_ = wallclock.timeBase();
    std::recursive_mutex busy_;
public:
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    unsigned getWaitTime(AVTS pkt_ts, AVTS now_ts) {
        auto lock = getLock();
        if (ready_) {
            AVTS diff = (pkt_ts - offset_) - now_ts;
            if (diff < -250) {
                logstream << "negative time to wait " << diff << "ms, resetting.";
                ready_ = false;
            } else if (diff < 1000) {
                return (diff > 0) ? diff : 0;
            } else {
                // diff >= 1000, discontinuity
                logstream << "timestamps difference " << diff << "ms, resetting.";
                ready_ = false;
            }
        }
        if (!ready_) {
            offset_ = pkt_ts - now_ts;
            ready_ = true;
            return 0;
        }
        throw Error("reached inaccessible code - should never happen");
    }
    template<typename T> unsigned getWaitTimeFromData(const T& data, AVTS now_ts) {
        if (!TSGetter<T>::isValid(data)) {
            logstream << "WARNING: invalid data";
            return 0;
        }
        AVTS pkt_ts = TSGetter<T>::get(data, tb_to_rescale_ts_);
        return getWaitTime(pkt_ts, now_ts);
    }
};

//INSTANCE_SHARED(SyncBufferCommon);


template <typename T> class SyncBuffer: public NodeSISO<T, T>, public IFlushable {
protected:
    std::shared_ptr<SyncBufferCommon> common_;
    std::queue<T> queue_;
    bool outputting_ = false;
    unsigned min_queue_size_ = 1;
    float output_when_have_enqueued_ = 0.2;
    av::Timestamp enqueuedTime() {
        if (queue_.size() >= 2) {
            return addTS(TSGetter<T>::getWithTB(queue_.back()),
                         negateTS(TSGetter<T>::getWithTB(queue_.front()))
                        );
        } else {
            return {0, {1, 1}};
        }
    }
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        int timeout_ms = -1;
        while (outputting_ && (queue_.size() >= min_queue_size_)) {
            AVTS now_ts = wallclock.pts();
            timeout_ms = common_->getWaitTimeFromData(queue_.front(), now_ts);
            if (timeout_ms==0) {
                this->sink_->put(queue_.front());
                queue_.pop();
            } else {
                break;
            }
        }
        if (queue_.size() < min_queue_size_) {
            outputting_ = false; // wait for buffer to fill
        }

        T in_data = this->source_->get(timeout_ms);

        if (in_data.isComplete()) {
            if (TSGetter<T>::getWithTB(in_data).isNoPts()) {
                logstream << "NOPTS not supported, dropping";
                return;
            }

            queue_.push(in_data);

            if ((!outputting_) && (enqueuedTime().seconds() >= output_when_have_enqueued_)) {
                outputting_ = true;
            }
        }
    }
    virtual void flush() {
        while (queue_.size()) {
            this->sink_->put(queue_.front());
            queue_.pop();
        }
    }
    static std::shared_ptr<SyncBuffer> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<SyncBuffer> r = NodeSISO<T, T>::template createCommon<SyncBuffer>(edges, params);
        std::string sgname = "default";
        if (params.count("sync_group")==1) {
            sgname = params.at("sync_group");
        }
        r->common_ = InstanceSharedObjects<SyncBufferCommon>::get(nci.instance, sgname);
        return r;
    }
};

DECLNODE_ATD(sync_buffer, SyncBuffer);
