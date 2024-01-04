#pragma once
#include "graph_interfaces.hpp"
#include "util.hpp"
#include "avutils.hpp"
#include <json.hpp>
#include <readerwriterqueue/readerwriterqueue.h>
#include <unordered_map>
#include "Event.hpp"
#include "instance.hpp"
#include "EventLoop.hpp"
#include "edge_meta_utils.hpp"

class EdgeBase;

class Node: public std::enable_shared_from_this<Node> {
public:
    virtual void process() = 0;
    virtual void start() {};
    virtual std::weak_ptr<Node> sourceNode() {
        return {};
    }
    virtual std::shared_ptr<EdgeBase> sourceEdge() {
        return {};
    }
    /*virtual std::weak_ptr<Node> sinkNode() {
        return {};
    }*/
    virtual ~Node() {
    }
};

class NonBlockingNodeBase: virtual public Node {
public:
    virtual void process() {
        throw Error("process() called for non-blocking node. Use processNonBlocking(...)");
    }
    virtual void processNonBlocking(EventLoop&, bool ticks) = 0;
};

template<typename Child> class NonBlockingNode: public NonBlockingNodeBase {
protected:
    std::shared_ptr<Child> thisAsShared() {
        return std::dynamic_pointer_cast<Child>(this->shared_from_this());
    }
};

template <typename T> class Source {
public:
    using DataType = T;
    virtual T get(const int timeout_ms = -1) = 0;
    virtual bool tryGet(T& dest, const int timeout_ms = -1) = 0;
    virtual T* peek(const int timeout_ms = -1) = 0;
    virtual bool tryPeek(T& dest, const int timeout_ms = -1) = 0;
    virtual bool pop() = 0;
    virtual ~Source() {
    }
};
template <typename T> class Sink {
public:
    using DataType = T;
    virtual bool put(const T&, bool = false) = 0;
    virtual ~Sink() {
    }
};

template<typename T> class Edge;

template <typename T> class EdgeWrapper {
protected:
    std::shared_ptr<Edge<T>> edge_;
public:
    EdgeWrapper(std::shared_ptr<Edge<T>> e): edge_(e) {
    }
    std::shared_ptr<Edge<T>> edge() {
        return edge_;
    }
    ~EdgeWrapper() {
    }
};

template <typename T> class EdgeSource: public Source<T>, public EdgeWrapper<T> {
public:
    using EdgeWrapper<T>::EdgeWrapper;
    virtual T get(const int timeout_ms = -1) {
        T data;
        if (timeout_ms < 0) {
            this->edge_->wait_dequeue(data);
        } else {
            this->edge_->wait_dequeue_timed_ms(data, timeout_ms);
        }
        return data;
    };
    virtual bool tryGet(T& dest, const int timeout_ms = -1) {
        if (timeout_ms < 0) {
            this->edge_->wait_dequeue(dest);
            return true;
        } else {
            return this->edge_->wait_dequeue_timed_ms(dest, timeout_ms);
        }
    };
    virtual T* peek(const int timeout_ms = -1) {
        return this->edge_->wait_peek(timeout_ms);
    }
    virtual bool tryPeek(T& dest, const int timeout_ms = -1) {
        T* ptr = this->edge_->wait_peek(timeout_ms);
        if (ptr) {
            dest = *ptr;
            return true;
        } else {
            return false;
        }
    }
    virtual bool pop() {
        return this->edge_->pop();
    }
};

template <typename T> class EdgeSink: public Sink<T>, public EdgeWrapper<T> {
public:
    using EdgeWrapper<T>::EdgeWrapper;
    virtual bool put(const T &data, bool drop_if_full = false) {
        if (!data.pts()) {
            logstream << "Warning: putting NOPTS into sink";
        }
        if (drop_if_full) {
            if (!this->edge_->try_enqueue(data)) {
                logstream << "Enqueue failed, queue full, dropping!";
                return false;
            } else {
                return true;
            }
        } else {
            this->edge_->enqueue(data);
            return true;
        }
    };
};

class EdgeBase: public std::enable_shared_from_this<EdgeBase> {
protected:
    std::list<std::shared_ptr<EdgeMetadata>> metadata_;
    std::weak_ptr<Node> producer_;
    std::weak_ptr<Node> consumer_;
    Event produced_;
    Event consumed_;
    std::atomic_bool finish_producer_{false};
    std::atomic_bool finish_consumer_{false};
    static void setNodePointer(std::weak_ptr<Node> &dest, std::weak_ptr<Node> source, std::atomic_bool &flag_to_reset) {
        // TODO? here we don't protect against race conditions but they won't happen anyway
        // unless someone really screws up the graph and uses the same node name in different groups
        // or starts a node without its group
        if (dest.expired()) {
            dest = source;
            flag_to_reset = false;
        } else {
            // TODO test whether it works for consumers, it has had some problems
            if (dest.lock() == source.lock()) {
                logstream << "Warning: setting edge more than once!" << std::endl;
            } else {
                throw Error("Edge connected to other node!");
            }
        }
    }
public:
    std::weak_ptr<Node> producer() {
        return producer_;
    }
    std::weak_ptr<Node> consumer() {
        return consumer_;
    }
    void setProducer(std::weak_ptr<Node> prod) {
        setNodePointer(producer_, prod, finish_producer_);
    }
    void setConsumer(std::weak_ptr<Node> cons) {
        setNodePointer(consumer_, cons, finish_consumer_);
    }
    template<typename MD> std::shared_ptr<MD> metadata(bool create_if_empty = false) {
        // TODO? race conditions as in setNodePointer
        for (std::shared_ptr<EdgeMetadata> &mdptr: metadata_) {
            std::shared_ptr<MD> r = std::dynamic_pointer_cast<MD>(mdptr);
            if (r != nullptr) {
                return r;
            }
        }
        if (create_if_empty) {
            std::shared_ptr<MD> r = std::make_shared<MD>();
            metadata_.push_back(r);
            return r;
        } else {
            return nullptr;
        }
    }
    template<typename MD> std::shared_ptr<MD> findMetadataUp() {
        std::shared_ptr<MD> md = metadata<MD>();
        std::shared_ptr<EdgeBase> edge = shared_from_this();
        while(true) {
            if (md != nullptr) return md;
            std::shared_ptr<Node> node = edge->producer().lock();
            if (node==nullptr) return nullptr;
            edge = node->sourceEdge();
            md = edge->metadata<MD>();
        }
    }
    template<typename NodeType> std::shared_ptr<NodeType> findNodeUp() {
        // find Node which has type of NodeType, walking up the processing graph
        std::shared_ptr<Node> node_shr = producer_.lock();

        do {
            // check if source is defined:
            if (node_shr==nullptr) return nullptr;

            // check if source is NodeType:
            std::shared_ptr<NodeType> node_p = std::dynamic_pointer_cast<NodeType>(node_shr);
            if (node_p!=nullptr) {
                return node_p;
            }

            // else continue search...
            node_shr = node_shr->sourceNode().lock();
        } while(true);
    }
    virtual void waitEmpty() = 0;
    virtual ~EdgeBase() {
    }
};

template<typename T> class Edge: public EdgeBase {
public:
    using WiretapCallback = std::function<void(const T&)>;
    static constexpr size_t default_capacity = 63;
protected:
    moodycamel::ReaderWriterQueue<T> queue_;
    int queue_limit_;
    std::list<WiretapCallback> wiretap_callbacks_;
    std::atomic_int occupied_{0};
    av::Timestamp last_ts_ = NOTS;

    // try_func returns whether item appeared in the queue
    // wait_func waits and returns false if timeout, true otherwise
    // alt_finish indicates whether we're finishing without giving data
    //   (as ALTernative way of ending the wait)
    static inline bool waitDo(std::function<bool()> try_func, std::function<bool()> wait_func, std::atomic_bool &alt_finish, Event &signal_event) {
        bool finish = false;
        while (!( try_func() || (finish = alt_finish) )) {
            if (!wait_func()) {
                return false;
            }
        }
        if (!finish) {
            signal_event.signal();
            return true;
        } else {
            alt_finish = false; // reset flag
            return false;
        }
    }
    static inline void signalAltFinish(std::atomic_bool &flag, Event &event) {
        flag = true;
        event.signal();
    }

    bool try_dequeue(T &elem) {
        bool r = queue_.try_dequeue(elem);
        if (r) {
            /*if (occupied_ <= 0) {
                logstream << "BUG: decreasing occupied_ = " << occupied_;
            }*/ // warning disabled, gave false positives because of race conditions
            --occupied_;
            consumed_.signal();
        }
        return r;
    }
    std::shared_ptr<Edge<T>> thisAsShared() {
        return std::static_pointer_cast<Edge<T>>(this->shared_from_this());
    }
public:
    void addWiretapCallback(WiretapCallback cb) {
        wiretap_callbacks_.push_back(cb);
    }
    bool try_enqueue(const T &elem) {
        bool r = queue_.try_enqueue(elem);
        if (r) {
            last_ts_ = elem.pts();
            ++occupied_;
            if (occupied_ > queue_limit_) {
                queue_limit_ = occupied_;
                //logstream << "BUG: occupied_ = " << occupied_;
            }
            produced_.signal();
            for (WiretapCallback &cb: wiretap_callbacks_) {
                cb(elem);
            }
        }
        return r;
    }
    Edge(const size_t capacity): queue_(capacity), queue_limit_(capacity) {
    }
    size_t capacity() {
        return queue_limit_;
    }
    int occupied() {
        //return queue_.size_approx();
        return occupied_;
    }
    int free() {
        //return capacity() - occupied();
        return queue_limit_ - occupied_;
    }
    av::Timestamp lastTS() {
        return last_ts_;
    }
    decltype(queue_)& queue() {
        return queue_;
    }
    Event& producedEvent() {
        return produced_;
    }
    Event& consumedEvent() {
        return consumed_;
    }

    void finishProducer() {
        signalAltFinish(finish_producer_, consumed_);
    }
    void finishConsumer() {
        signalAltFinish(finish_consumer_, produced_);
    }

    std::unique_ptr<EdgeSource<T>> makeSource() {
        return make_unique<EdgeSource<T>>(thisAsShared());
    }
    std::unique_ptr<EdgeSink<T>> makeSink() {
        return make_unique<EdgeSink<T>>(thisAsShared());
    }
    /*bool enqueue(T &&elem) {
        waitDo([this, &elem](){ return queue_.try_enqueue(std::move(elem)); }, finish_producer_, consumed_, produced_); // warning: old synopsis, update when uncommenting!
    }*/
    // lambdas don't support move semantics so generally the above is useless
    bool enqueue(const T &elem) {
        last_ts_ = elem.pts();
        return waitDo([this, &elem](){ return try_enqueue(elem); }, [this]() { consumed_.wait(); return true; }, finish_producer_, produced_);
    }
    T* peek() {
        return queue_.peek();
    }
    T* wait_peek(const int timeout_ms = -1) {
        T* r = queue_.peek();
        if (timeout_ms==0) return r;
        if (r != nullptr) return r;
        AVTS remaining = timeout_ms;
        bool wait_inf = timeout_ms < 0;
        AVTS wait_till;
        if (!wait_inf) {
            wait_till = wallclock.pts() + timeout_ms;
        }
        do {
            produced_.wait(remaining);
            if (!wait_inf) remaining = wait_till - wallclock.pts();
        } while ( (!finish_consumer_) && ((r = queue_.peek()) == nullptr) && (remaining>0 || wait_inf) );
        return r;
    }
    bool pop() {
        if (queue_.pop()) {
            occupied_--;
            consumed_.signal();
            return true;
        } else {
            return false;
        }
    }
    void wait_dequeue(T &elem) {
        waitDo([this, &elem]() { return try_dequeue(elem); }, [this]() { produced_.wait(); return true; }, finish_consumer_, consumed_);
    }
    bool wait_dequeue_timed_ms(T &elem, const unsigned int msec) {
        return waitDo([this, &elem]() { return try_dequeue(elem); }, [this, msec]() { return produced_.wait(msec) > 0; }, finish_consumer_, consumed_);
    }
    virtual void waitEmpty() override {
        while (occupied_.load() > 0) {
            if (!consumer_.expired()) {
                // there IS consumer
                consumed_.wait();
            } else {
                // no consumer
                // so empty the queue artificially.
                logstream << "Warning: waitEmpty() called for queue without consumer. Discarding " << occupied_.load() << " items.";
                while (pop()) {};
            }
        }
    }
};

template<typename T> class EdgesOfType {
private:
    std::unordered_map<std::string, std::shared_ptr<Edge<T>>> edges_;
public:
    std::shared_ptr<Edge<T>> findInternal(const std::string &name, bool create_if_empty = true, const size_t capacity = Edge<T>::default_capacity) {
        if ( create_if_empty && (edges_[name]==nullptr) ) {
            edges_[name] = std::make_shared<Edge<T>>(capacity);
        }
        return edges_[name];
    };
    template <typename OStream> void printStats(OStream &ost, bool compact = false, const std::string prefix = "") {
        for (auto &kv: edges_) {
            const std::string name = prefix + kv.first;
            std::shared_ptr<Edge<T>> edge = kv.second;
            if (edge==nullptr) continue;
            std::stringstream ss;
            int capacity = edge->capacity();
            int occupied = edge->occupied();
            if (occupied > capacity) occupied = capacity;
            int free = capacity - occupied;
            //if (free < 0) free = 0;
            if (compact) {
                ss << name << ':' << occupied << '/' << capacity << ", ";
                ost << ss.str();
            } else {
                ss << '[' << std::string(occupied, '#') << std::string(free, '.') << "]  " << edge->lastTS() << "=" << edge->lastTS().seconds() << "   " << name;
                ost << ss.str() << std::endl;
            }
        }
    }
};

class EdgeManager {
protected:
    static EdgeManager global_edge_manager_;
    edge_meta_utils::MultiContainer<EdgesOfType> storage_;
    std::unordered_map<std::string, std::string> edge_types_;
    std::unordered_map<std::string, size_t> planned_capacities_;
    size_t default_capacity_ = 63;
    std::recursive_mutex busy_;
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    static bool isNameGlobal(const std::string name) {
        return name.length()>0 && name[0]=='@';
    }
public:
    EdgeManager() {
    }
    EdgeManager(const EdgeManager&) = delete;
    template<typename T> std::shared_ptr<Edge<T>> find(const std::string &name) {
        if (isNameGlobal(name)) {
            return global_edge_manager_.find<T>(name.substr(1));
        }
        auto lock = getLock();
        std::string &stored_type = edge_types_[name];
        std::string new_type = typeid(T).name();
        if (stored_type.empty() || (stored_type==new_type)) {
            // all OK
            if (stored_type.empty()) {
                stored_type = new_type;
            }
            size_t capacity = default_capacity_;
            if (planned_capacities_.count(name)>0) {
                capacity = planned_capacities_[name];
            }
            auto r = storage_.get<T>()->findInternal(name, true, capacity);
            return r;
        } else {
            throw Error(std::string("Edge ") + name + " has type " + stored_type + ", not " + new_type);
        }
    }
    std::shared_ptr<EdgeBase> findAny(const std::string &name) {
        if (isNameGlobal(name)) {
            return global_edge_manager_.findAny(name.substr(1));
        }
        auto lock = getLock();
        std::shared_ptr<EdgeBase> r = nullptr;
        storage_.some([&name, &r](auto edges) -> bool {
            r = edges->findInternal(name, false);
            return r != nullptr;
        });
        return r;
    }
    void planCapacity(const std::string &name, const size_t capacity) {
        if (isNameGlobal(name)) {
            global_edge_manager_.planCapacity(name.substr(1), capacity);
            return;
        }
        auto lock = getLock();
        if (name == std::string("*")) {
            default_capacity_ = capacity;
        } else {
            planned_capacities_[name] = capacity;
        }
    }
    template<typename T> bool exists(const std::string &name) {
        if (isNameGlobal(name)) {
            return global_edge_manager_.exists<T>(name.substr(1));
        }
        auto lock = getLock();
        return storage_.get<T>()->findInternal(name, false) != nullptr;
    }
    template <typename OStream> void printEdgesStats(OStream &ost, bool compact = false) {
        bool we_are_global = this==&global_edge_manager_;
        const std::string prefix = we_are_global ? "@" : "";
        if (!we_are_global) {
            global_edge_manager_.printEdgesStats(ost, compact);
        }
        auto lock = getLock();
        storage_.forEach([&ost, compact, &prefix](auto edges) {
            edges->printStats(ost, compact, prefix);
        });
    }
};

struct NodeCreationInfo {
    EdgeManager &edges;
    const Parameters &params;
    InstanceData &instance;
};
