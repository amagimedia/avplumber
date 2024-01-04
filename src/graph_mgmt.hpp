#pragma once
#include <functional>
#include <unordered_map>
#include <list>
#include <string>
#include <mutex>
#include "Event.hpp"
#include "graph_core.hpp"
#include "graph_factory.hpp"
#include "instance.hpp"

class NodeFactory {
public:
    using NodeFactoryFunction = ::NodeFactoryFunction;
protected:
    std::unordered_map<std::string, NodeFactoryFunction> factories_;
    std::shared_ptr<EdgeManager> edges_;
    InstanceData &instance_;
public:
    std::shared_ptr<Node> produce(const Parameters &params);
    NodeFactory(std::shared_ptr<EdgeManager> edgeman, InstanceData &inst);
};

class NodeManager;
class NodeGroup;
class TickSource;

class NodeWrapper: public std::enable_shared_from_this<NodeWrapper> {
public:
    // cb(this->shared_from_this(), exit_requested)
    using OnFinishedHandler = std::function<void(std::shared_ptr<NodeWrapper>, bool)>;
protected:
    std::string name_;
    std::shared_ptr<Node> node_;
    std::unique_ptr<std::thread> thread_;
    std::shared_ptr<NodeManager> manager_;
    std::shared_ptr<NodeGroup> group_;
    std::shared_ptr<TickSource> tick_source_;
    std::atomic_bool dowork_ {false};
    std::atomic_bool finished_;
    std::atomic_bool stop_requested_;

    std::list<OnFinishedHandler> on_finished_;
    std::string last_error_;
    Parameters params_;
    std::recursive_mutex start_stop_mutex_;
    void threadFunction();
    inline bool threadWorks() {
        return ((thread_!=nullptr) && (!finished_));
    }
    bool isNonBlocking() {
        return std::dynamic_pointer_cast<NonBlockingNodeBase>(node_) != nullptr;
    }
public:
    void createNode();
    inline decltype(group_)& group() {
        return group_;
    }
    inline Parameters& parameters() {
        return params_;
    }
    decltype(node_) node() {
        return node_;
    }
    inline void onFinished(OnFinishedHandler on_finished) {
        on_finished_.push_back(on_finished);
    }
    inline const std::string& name() {
        return name_;
    }
    inline bool isWorking() {
        return threadWorks() || (isNonBlocking() && node_!=nullptr && dowork_);
    }
    inline void doLocked(std::function<void()> cb) {
        std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
        cb();
    }
    bool start();
    bool stop(bool inhibit_actions = true);
    bool interrupt(bool optional = false);
    Parameters getObject(const std::string);

    bool stopAndWait();
    void join();
    NodeWrapper(std::shared_ptr<NodeManager> manager, const Parameters &params, const bool early_create = true);
    virtual ~NodeWrapper();
};

class NodeGroup: public std::enable_shared_from_this<NodeGroup> {
public:
    using SolidItem = std::shared_ptr<NodeWrapper>;
    using Item = std::weak_ptr<NodeWrapper>;
    NodeGroup(NodeManager* manager, const std::string name);
    ~NodeGroup();
    NodeGroup(const NodeGroup&) = delete;
protected:
    std::string name_;
    NodeManager* manager_;
    std::list<Item> nodes_;
    std::list<Item> sorted_nodes_;
    std::recursive_mutex busy_;
    std::atomic_uint64_t start_id_{0};
    enum class State {
        EMPTY,
        STOPPED,
        STARTED,
        MIXED,
        RESTART,
        FINISH_THREAD
    };
    std::atomic<State> desired_state_ = State::EMPTY;
    Event mgmt_thread_wakeup_;
    void goToState(State desired) {
        desired_state_ = desired;
        start_id_++;
        mgmt_thread_wakeup_.signal();
    }
    std::thread mgmt_thread_;
    bool is_sorted_ = false;
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
private:
    void sort();
    bool doWithNodes(std::function<void(NodeWrapper&)> cb, const bool retry_single, const std::string &operation_desc);
    void stopNodesInternal();
    decltype(start_id_)::value_type startNodesInternal();
    void restartNodesInternal();
    State currentState();
public:
    void collectGarbage();
    void add(SolidItem node);
    void stopNodes() {
        goToState(State::STOPPED);
    }
    void stopNodesAndFinishThread() {
        logstream << "Shutting down group " << name_;
        goToState(State::FINISH_THREAD);
        logstream << "Waiting for management thread to finish";
        mgmt_thread_.join();
        logstream << "Management thread finished";
    }
    void startNodes() {
        goToState(State::STARTED);
    }
    void restartNodes() {
        goToState(State::RESTART);
    }
    const std::list<Item>& sortedNodes();
};

class NodeManager: public std::enable_shared_from_this<NodeManager> {
    friend class NodeWrapper;
protected:
    std::shared_ptr<EdgeManager> edges_;
    std::shared_ptr<NodeFactory> factory_;
    InstanceData instance_;
    std::unordered_map<std::string, std::shared_ptr<NodeGroup>> groups_;
    std::unordered_map<std::string, std::shared_ptr<NodeWrapper>> nodes_index_;
    std::recursive_mutex busy_;
    std::atomic_bool should_work_ {true};
    Event shutdown_complete_;
    
    bool nodeExists(const std::string &name);
    std::shared_ptr<NodeWrapper> getNodeByName(const std::string &name);
    void panic();
public:
    NodeManager(const NodeManager&) = delete;
    std::shared_ptr<NodeWrapper> createNode(Parameters &params, const bool early_create, const bool start);
    void deleteNode(const std::string &name);
    void interrupt();
    void shutdown(); // do not use NodeManager after calling it
    Event &shutdownCompleteEvent() { return shutdown_complete_; }
    NodeManager(): edges_(std::make_shared<EdgeManager>()), factory_(std::make_shared<NodeFactory>(edges_, instance_)) {
    }
    std::shared_ptr<NodeWrapper> node(const std::string &name) {
        std::shared_ptr<NodeWrapper> p = getNodeByName(name);
        if (p==nullptr) {
            throw Error("Node " + name + " doesn't exist.");
        }
        return p;
    }
    std::shared_ptr<NodeGroup> group(const std::string name) {
        auto lock = getLock();
        auto iter = groups_.find(name);
        if (iter == groups_.end()) {
            throw Error("Group " + name + " doesn't exist.");
        }
        return (*iter).second;
    }
    decltype(edges_)& edges() {
        return edges_;
    }
    InstanceData& instanceData() { return instance_; }
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    bool shouldWork() const {
        return should_work_;
    }
};
