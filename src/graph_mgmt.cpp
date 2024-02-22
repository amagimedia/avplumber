#include "graph_mgmt.hpp"
#include "Event.hpp"
#include "EventLoop.hpp"
#include "TickSource.hpp"
#include "graph_core.hpp"
#include "graph_factory.hpp"
#include "instance_shared.hpp"
#include <atomic>
#include <exception>
#include <limits>
#include <memory>

///////////////////////////////////////////////////////////
////// NodeFactory

NodeFactory::NodeFactory(std::shared_ptr< EdgeManager > edgeman, InstanceData &inst): edges_(edgeman), instance_(inst) {
    initFactories(factories_);
}

std::shared_ptr< Node > NodeFactory::produce(const Parameters& params) {
    std::string type = params["type"];
    //logstream << "Producing " << name << std::endl;
    if (factories_.count(type)!=1) {
        throw Error("Unknown node type: " + type);
    }
    NodeFactoryFunction func = factories_[type];
    if (func==nullptr) {
        throw Error("Invalid factory function for " + type);
    }
    NodeCreationInfo nci {*edges_, params, instance_};
    std::shared_ptr<Node> r = func(nci);
    return r;
}


///////////////////////////////////////////////////////////
////// NodeWrapper

bool NodeWrapper::start() {
    std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
    if (!isWorking()) {
        last_error_ = "";
        createNode();
        if (node_==nullptr) {
            logstream << "Node " << name_ << " creation failed, not starting.";
            return true;
        }
        dowork_ = true;
        finished_ = false;
        stop_requested_ = false;
        try {
            if (thread_ && thread_->joinable()) thread_->join();
        } catch (std::system_error&) {
        }

        std::shared_ptr<NonBlockingNodeBase> nbnode = std::dynamic_pointer_cast<NonBlockingNodeBase>(node_);
        if (nbnode) {
            nbnode->start();
            if (tick_source_!=nullptr) {
                tick_source_->add(std::weak_ptr<NonBlockingNodeBase>(nbnode));
            } else {
                if (event_loop_==nullptr) {
                    event_loop_ = InstanceSharedObjects<EventLoop>::get(manager_->instanceData(), "default");
                }
                event_loop_->execute([nbnode](EventLoop &evl) {
                    nbnode->processNonBlocking(evl, false);
                });
            }
        } else {
            if (tick_source_!=nullptr || event_loop_!=nullptr) {
                throw Error("tick_source or event_loop can't be specified for blocking (threaded) node");
            }
            // this is blocking Node so it requires separate thread
            thread_ = make_unique<std::thread>(start_thread(name_, [this]() {
                this->threadFunction();
            }));
        }
        return true;
    } else {
        return false;
    }
}

void NodeWrapper::createNode() {
    std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
    if (node_==nullptr) {
        auto produceObject = [&]() {
            node_ = manager_->factory_->produce(params_);
            if (node_==nullptr) {
                throw Error("Node factory returned nullptr");
            }
            std::shared_ptr<IInitAfterCreate> node_init = std::dynamic_pointer_cast<IInitAfterCreate>(node_);
            if (node_init) {
                try {
                    node_init->init(*manager_->edges(), params_);
                } catch (std::exception &e) {
                    logstream << "Node " << name_ << " init failed: " << e.what();
                    node_ = nullptr;
                    throw;
                }
            }
        };
        if (params_.count("optional") && params_["optional"].get<bool>()) {
            try {
                produceObject();
            } catch(std::exception &e) {
                logstream << "Ignoring exception when creating optional node " << name_ << ": " << e.what();
            }
        } else {
            produceObject();
        }
    }
}

Parameters NodeWrapper::getObject(const std::string object_name) {
    std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
    if (node_==nullptr) {
        throw Error("Node not created");
    }
    IReturnsObjects* retobj = dynamic_cast<IReturnsObjects*>(node_.get());
    if (retobj==nullptr) {
        throw Error("Node doesn't return objects");
    }
    return retobj->getObject(object_name);
}

bool NodeWrapper::stopAndWait() {
    std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
    bool r = interrupt(true);
    if (!r) {
        r = stop();
    }
    join();

    return r;
}

void NodeWrapper::join() {
    if (thread_ && thread_->joinable()) thread_->join();
}


bool NodeWrapper::stop(bool inhibit_actions) {
    std::lock_guard<decltype(start_stop_mutex_)> lock(start_stop_mutex_);
    if (threadWorks()) {
        std::shared_ptr<IStoppable> node_stoppable = std::dynamic_pointer_cast<IStoppable>(node_);
        if (node_stoppable) {
            if (inhibit_actions) stop_requested_ = true;
            dowork_ = false;
            node_stoppable->stop();
        } else {
            throw Error("NodeWrapper::stop() called for node which doesn't have stopping interface!");
        }
        std::shared_ptr<IWaitsSinksEmpty> node_sinks = std::dynamic_pointer_cast<IWaitsSinksEmpty>(node_);
        if (node_sinks) {
            node_sinks->stopSinks();
        }
        return true;
    } else {
        dowork_ = false;
        // node is destroyed in thread function only if it was started, so a special case for not-yet-started nodes
        // and non-blocking nodes is necessary:
        if (node_ != nullptr) {
            std::shared_ptr<IFlushable> node_flushable = std::dynamic_pointer_cast<IFlushable>(node_);
            if (node_flushable) {
                logstream << "Flushing node " << name_ << " from stop()";
                node_flushable->flush();
            }
            logstream << "Destroying node " << name_ << " from stop()";
            node_ = nullptr;
            finished_ = true;
        }
        return false;
    }
}

bool NodeWrapper::interrupt(bool optional) {
    std::shared_ptr<Node> node = node_;
    if (node) {
        std::shared_ptr<IInterruptible> node_interruptible = std::dynamic_pointer_cast<IInterruptible>(node);
        if (node_interruptible) {
            stop_requested_ = true;
            node_interruptible->interrupt();
        } else if (optional) {
            return false;
        } else {
            throw Error("NodeWrapper::interrupt() called for node which doesn't have interrupting interface!");
        }
        std::shared_ptr<IWaitsSinksEmpty> node_sinks = std::dynamic_pointer_cast<IWaitsSinksEmpty>(node);
        if (node_sinks) {
            node_sinks->stopSinks();
        }
        //node_ = nullptr;
        return true;
    } else {
        return false;
    }
}

void NodeWrapper::threadFunction() {
    decltype(node_) node = node_;
    if (node==nullptr) {
        logstream << "BUG: race condition detected, node_==nullptr in threadFunction() !!!";
        return;
    }
    IReportsFinish *node_finishable = dynamic_cast<IReportsFinish*>(node.get());
    IFlushable *node_flushable = dynamic_cast<IFlushable*>(node.get());
    try {
        logstream << "Node " << name_ << " started." << std::endl;
        node->start();
        if (node_finishable) {
            // Node signals that it finished work
            while (!node_finishable->finished()) {
                if (node_==nullptr) {
                    logstream << "BUG: race condition detected, node_==nullptr in threadFunction() !!! (node_finishable loop)";
                    return;
                }
                if (dowork_) {
                    node->process();
                } else if (node_flushable) {
                    // told to finish work
                    // and Node is IFlushable
                    // so flush it
                    node_flushable->flush();
                } else {
                    // node should finish work
                    // but doesn't have flushing interface
                    // so to give it a chance to finish,
                    // call process()...
                    node_->process();
                }
            }
            logstream << "Node " << name_ << " reported that it finished processing.";
        } else {
            // dumb Node
            while (dowork_) {
                if (node_==nullptr) {
                    logstream << "BUG: race condition detected, node_==nullptr in threadFunction() !!! (dumb Node loop)";
                    return;
                }
                node->process();
            }
            if (node_flushable) {
                node_flushable->flush();
            }
            logstream << "Node " << name_ << " stopped processing because it was told to do so.";
        }
    } catch (std::exception &e) {
        logstream << "Node " << name_ << " failed: " << e.what();
        last_error_ = e.what();
    }
    try {
        node_ = nullptr;
    } catch (std::exception &e) {
        logstream << "Destroying node " << name_ << " failed: " << e.what();
    }

    finished_ = true;
    logstream << "Node " << name_ << " finished." << std::endl;
    if ((!on_finished_.empty()) && manager_->shouldWork()) {
        set_thread_name("~"+name_);
        try {
            thread_->detach(); // detach myself to call on_finished_ independently
        } catch (std::exception &e) {
            logstream << "unable to detach thread: " << e.what();
        }
        for (auto &cb: on_finished_) {
            try {
                cb(this->shared_from_this(), stop_requested_);
            } catch (std::exception &e) {
                logstream << "on_finished callback failed: " << e.what();
            }
        }
    }
}



///////////////////////////////////////////////////////////
////// NodeManager

std::shared_ptr< NodeWrapper > NodeManager::createNode(Parameters& params, const bool early_create, const bool start) {
    auto nw = std::make_shared<NodeWrapper>(this->shared_from_this(), params, false);

    { // begin lock
        auto lock = getLock();

        if (nodeExists(nw->name())) {
            throw Error("Name busy: " + nw->name());
        }

        bool in_group = false;
        if (params.count("group") > 0) {
            std::string groupname = params["group"];
            std::shared_ptr<NodeGroup> group = groups_[groupname];
            if (group==nullptr) {
                group = std::make_shared<NodeGroup>(this, groupname);
                groups_[groupname] = group;
            }
            groups_[groupname]->add(nw);
            nw->group() = groups_[groupname];
            in_group = true;
        }

        std::string auto_restart = "off";
        if (params.count("auto_restart") > 0) {
            auto_restart = params["auto_restart"];
        }
        if (auto_restart == "on") {
            nw->onFinished([](std::shared_ptr<NodeWrapper> n, bool requested) {
                if (requested) return;
                logstream << "Node " << n->name() << " finished, restarting.";
                n->start();
            });
        } else if (auto_restart == "group") {
            if (!in_group) {
                throw Error("Node must belong to group to use auto_restart=group");
            }
            nw->onFinished([](std::shared_ptr<NodeWrapper> n, bool requested) {
                if (requested) return;
                logstream << "Node " << n->name() << " initiated group auto-restart";
                n->group()->restartNodes();
                logstream << "Auto-restart scheduled.";
            });
        } else if (auto_restart == "panic") {
            nw->onFinished([this](std::shared_ptr<NodeWrapper> n, bool requested) {
                if (requested) return;
                logstream << "Node " << n->name() << " finished but it should never finish (declared as auto_restart=panic)";
                this->panic();
            });
        } else if (auto_restart == "off") {
            // don't do anything
        } else {
            throw Error("Invalid auto_restart mode, should be on/group/panic/off");
        }
        nodes_index_[nw->name()] = nw;
        logstream << "NodeWrapper " << nw->name() << " added to manager.";
    } // end lock

    if (early_create) {
        nw->createNode();
    }
    if (start) {
        nw->start();
    }
    return nw;
}

void NodeManager::deleteNode(const std::string& name) {
    std::shared_ptr<NodeWrapper> nw;
    { // begin lock
        auto lock = getLock();
        auto iter = nodes_index_.find(name);
        if (iter == nodes_index_.end()) {
            throw Error(name + " not found");
        }
        nw = iter->second;
        nodes_index_.erase(iter);
    } // end lock
    // and destroy the node outside of the lock
    if (!nw) return;
    nw->stopAndWait();
}

void NodeManager::interrupt() {
    auto lock = getLock();
    for (auto &entry: nodes_index_) {
        std::shared_ptr<NodeWrapper> n = entry.second;
        try {
            n->interrupt(true);
        } catch (std::exception &e) {
            logstream << "Error interrupting " << n->name() << ": " << e.what();
        }
    }
}

void NodeManager::shutdown() {
    auto lock = getLock();
    if (!should_work_) {
        return; // already shut(ting) down
    }
    should_work_ = false;
    logstream << "Stopping all groups";
    for (auto &entry: groups_) {
        entry.second->stopNodesAndFinishThread();
    }
    logstream << "Stopping all remaining nodes";
    for (auto &entry: nodes_index_) {
        std::shared_ptr<NodeWrapper> n = entry.second;
        try {
            if (!n->interrupt(true)) {
                n->stop();
            }
        } catch (std::exception &e) {
            logstream << "Error stopping " << n->name() << ": " << e.what();
        }
    }
    logstream << "Stop request sent to all nodes (unless there was an error), waiting for them";
    for (auto &entry: nodes_index_) {
        std::shared_ptr<NodeWrapper> n = entry.second;
        n->join();
    }
    logstream << "All nodes threads finished";
    nodes_index_.clear();
    logstream << "All nodes destroyed";
    shutdown_complete_.signal();
}

void NodeManager::panic() {
    logstream << "Critical error. Shutting down.";
    shutdown();
}

std::shared_ptr< NodeWrapper > NodeManager::getNodeByName(const std::string& name) {
    auto lock = getLock();
    auto iter = nodes_index_.find(name);
    if (iter == nodes_index_.end()) {
        return nullptr;
    } else {
        return iter->second;
    }
}


bool NodeManager::nodeExists(const std::string& name) {
    return getNodeByName(name) != nullptr;
}


const std::list<NodeGroup::Item>& NodeGroup::sortedNodes() {
    auto lock = getLock();
    if (!is_sorted_) sort();
    return sorted_nodes_;
}


///////////////////////////////////////////////////////////
////// NodeGroup

void NodeGroup::add(SolidItem node) {
    collectGarbage();
    {
        auto lock = getLock();
        nodes_.push_back(node);
        is_sorted_ = false;
        sorted_nodes_.clear();
    }
}

// TODO remove retry_single argument, not used anymore
bool NodeGroup::doWithNodes(std::function<void (NodeWrapper &)> callback, const bool retry_single, const std::string &operation_desc) {
    auto lock = getLock();
    auto nodes_list = sortedNodes();
    for (std::weak_ptr<NodeWrapper> &weak_node: nodes_list) {
        std::shared_ptr<NodeWrapper> node = weak_node.lock();
        if (!node) continue;
        bool success = false;
        do {
            try {
                logstream << operation_desc << " node " << node->name() << " ...";
                callback(*node);
                success = true;
                logstream << operation_desc << " node " << node->name() << " done.";
            } catch (std::exception &e) {
                logstream << operation_desc << " node " << node->name() << " failed: " << e.what();
                if (retry_single) {
                    wallclock.sleepms(1000);
                } else {
                    throw;
                }
            }
        } while ((!success) && retry_single && manager_->shouldWork());
        if (!success) {
            return false;
        }
    }
    return true;
}

void NodeGroup::stopNodesInternal() {
    doWithNodes([](NodeWrapper &n) {
        n.stopAndWait();
    }, false, "Stopping");
}

decltype(NodeGroup::start_id_)::value_type NodeGroup::startNodesInternal() {
    try {
        auto lock = getLock();
        decltype(start_id_)::value_type start_id = start_id_.load(std::memory_order_acquire);
        doWithNodes([this, start_id](NodeWrapper &n) {
            if (start_id == start_id_.load()) {
                n.createNode();
            } else {
                logstream << "Another start of the group requested (while creating)";
                throw NotReallyError("Another start of the group requested");
            }
        }, false, "Creating");
        doWithNodes([this, start_id](NodeWrapper &n) {
            if (start_id == start_id_.load()) {
                n.start();
            } else {
                logstream << "Another start of the group requested (while starting)";
                throw NotReallyError("Another start of the group requested");
            }
        }, false, "Starting");
        return start_id;
    } catch (std::exception &e) {
        doWithNodes([](NodeWrapper &n) {
            try {
                n.stopAndWait();
            } catch (std::exception &e) {
            }
        }, false, "Stopping (after failure)");
        throw;
    }
}

void NodeGroup::restartNodesInternal() {
    auto lock = getLock();
    logstream << "Restarting group nodes";
    stopNodesInternal();
    startNodesInternal();
}

void NodeGroup::collectGarbage() {
    auto lock = getLock();
    auto is_garbage = [](Item &item) {
        return item.expired();
    };
    nodes_.remove_if(is_garbage);
    sorted_nodes_.remove_if(is_garbage);
}

namespace NodeGroupUtils {
    static std::list<std::string> offers(Parameters& params) {
        std::list<std::string> we_offer;
        if (params["type"]=="demux") {
            Parameters routing = params["routing"];
            for (Parameters::iterator route = routing.begin(); route != routing.end(); ++route) {
                we_offer.push_back(route.value());
            }
        } else if (params.count("dst")==1) {
            we_offer = jsonToStringList(params["dst"]);
        }
        return we_offer;
    }
    static std::list<std::string> offers(NodeGroup::Item& weak_node) {
        NodeGroup::SolidItem node = weak_node.lock();
        if (node==nullptr) return {};
        return offers(node->parameters());
    }

    static std::list<std::string> needs(Parameters& params) {
        if (params.count("src")==1) {
            return jsonToStringList(params["src"]);
        } else {
            return {};
        }
    }
    static std::list<std::string > needs(NodeGroup::Item& weak_node) {
        NodeGroup::SolidItem node = weak_node.lock();
        if (node==nullptr) return {};
        return needs(node->parameters());
    }


    // sorting algorithm source: https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
    enum SortNodeMark {
        None, Temporary, Permanent
    };
    struct SortNode {
        NodeGroup::Item item;
        SortNodeMark mark;
        SortNode(NodeGroup::Item a_item): item(a_item), mark(None) {
        }
    };
    static void sortVisit(SortNode& n, std::list<SortNode>& all_nodes, std::list<NodeGroup::Item> &target_list) {
        if (n.mark==Permanent) return;
        if (n.mark==Temporary) {
            throw Error("Graph has cycle!");
        }
        auto n_offers = offers(n.item);
        if (!n_offers.empty()) {
            n.mark = Temporary;
            for (SortNode &m: all_nodes) {
                if (m.mark != None) continue; // WARNING: this optimization may sabotage cycle checking
                auto m_needs = needs(m.item);
                if (m_needs.empty()) continue; // skip the whole loop
                bool has_edge = false;
                // FIXME: the following loop-in-loop is inefficient!!!
                for (const std::string &offer: n_offers) {
                    for (const std::string &need: m_needs) {
                        if (offer==need) {
                            has_edge = true;
                            break;
                        }
                    }
                    if (has_edge) break;
                }
                if (has_edge) {
                    sortVisit(m, all_nodes, target_list);
                }
            }
        }
        n.mark = Permanent;
        target_list.push_front(n.item);
    }
    static void sort(std::list<NodeGroup::Item> &target_list, std::list<NodeGroup::Item> &source_list) {
        target_list.clear();
        std::list<SortNode> all_nodes;
        for (NodeGroup::Item &elem: source_list) {
            all_nodes.push_back(SortNode(elem));
        }
        for (SortNode &n: all_nodes) {
            sortVisit(n, all_nodes, target_list);
        }
    }
};

void NodeGroup::sort() {
    collectGarbage();
    {
        auto lock = getLock();
        NodeGroupUtils::sort(sorted_nodes_, nodes_);
    }
}

NodeGroup::State NodeGroup::currentState() {
    auto lock = getLock();
    State s = State::EMPTY;
    for (auto &node: nodes_) {
        auto n = node.lock();
        if (!n) continue;
        State ns = n->isWorking() ? State::STARTED : State::STOPPED;
        if (s != ns) {
            if (s == State::EMPTY) {
                s = ns;
            } else {
                s = State::MIXED;
                break;
            }
        }
    }
    return s;
}

NodeGroup::NodeGroup(NodeManager* manager, const std::string name):
    manager_(manager),
    name_(name) {
    mgmt_thread_ = start_thread(std::string("GM:") + name, [this]() {
        bool dowork = true;
        bool retry = false;
        while(dowork) {
            if (!retry) {
                mgmt_thread_wakeup_.wait();
            }
            State desired = desired_state_;
            State cur = currentState();
            if (cur != desired) {
                try {
                    if (desired==State::STARTED) {
                        startNodesInternal();
                    } else if (desired==State::STOPPED) {
                        stopNodesInternal();
                    } else if (desired==State::RESTART) {
                        desired_state_ = State::STARTED;
                        restartNodesInternal();
                    } else if (desired==State::FINISH_THREAD) {
                        stopNodesInternal();
                        dowork = false;
                    } else {
                        throw Error("BUG: unsupported desired state");
                    }
                    retry = false;
                } catch (std::exception &e) {
                    if (currentState() == desired) {
                        logstream << "BUG: state change caused exception but left currentState() == desired";
                    }
                    logstream << "Error while changing state: " << e.what() << ", retrying";
                    wallclock.sleepms(1000);
                    retry = true;
                }
            } else if (retry) {
                logstream << "BUG: currentState() != desired_state_ but retry==true, fixing to avoid infinite loop";
                retry = false;
            }
        }
    });
}

NodeGroup::~NodeGroup() {
    if (mgmt_thread_.joinable()) {
        logstream << "Waiting for group " << name_ << " management thread to finish";
        mgmt_thread_.join();
        logstream << "Group " << name_ << " management thread finished";
    }
}


///////////////////////////////////////////////////////////
////// NodeWrapper:

NodeWrapper::~NodeWrapper() {
    logstream << "Destroying NodeWrapper " << name_;
    stopAndWait();
    logstream << "Destroyed NodeWrapper " << name_;
}


NodeWrapper::NodeWrapper(std::shared_ptr< NodeManager > manager, const Parameters& params, const bool early_create):
    manager_(manager), finished_(false), params_(params) {
    // SET NAME:
    if (params_.count("name") > 0) {
        name_ = params_["name"];
    } else {
        std::stringstream name;
        name << params_["type"].get<std::string>() << "@";
        name << std::hex << reinterpret_cast<std::uintptr_t>(this);
        name_ = name.str();
    }
    if (params_.count("tick_source") > 0) {
        tick_source_ = InstanceSharedObjects<TickSource>::get(manager->instanceData(), params["tick_source"]);
    }
    if (params_.count("event_loop") > 0) {
        event_loop_ = InstanceSharedObjects<EventLoop>::get(manager->instanceData(), params["event_loop"]);
    }
    if (tick_source_ && event_loop_) {
        throw Error("tick_source and event_loop can't be specified at the same time");
    }

    if (early_create) {
        createNode();
    }
}


