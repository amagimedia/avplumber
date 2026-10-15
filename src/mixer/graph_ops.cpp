#include "NodeManagerGraph.hpp"
#include "../graph_interfaces.hpp"
#include <algorithm>
#include <cctype>
#include <iterator>
#include <vector>

namespace avp::mixer {

NodeManagerMixerGraph::NodeManagerMixerGraph(const std::shared_ptr<NodeManager>& manager)
    : edges_(manager->edges()), manager_(manager), instance_(manager->instanceData()) {
    for (const auto& [name, node] : manager->allNodes()) {
        nodes_.emplace(name, node);
        if (node->parameters().contains("group"))
            groups_[node->parameters().at("group").get<std::string>()] = node->group();
    }
}

std::shared_ptr<NodeWrapper> NodeManagerMixerGraph::node_if_exists(const std::string& name) const {
    auto it = nodes_.find(name);
    return it == nodes_.end() ? nullptr : it->second.lock();
}

std::shared_ptr<NodeWrapper> NodeManagerMixerGraph::node(const std::string& name) const {
    auto result = node_if_exists(name);
    if (!result) throw Error("Mixer node no longer exists: " + name);
    return result;
}

std::shared_ptr<NodeGroup> NodeManagerMixerGraph::group(const std::string& name) const {
    auto it = groups_.find(name);
    auto result = it == groups_.end() ? nullptr : it->second.lock();
    if (!result) throw Error("Mixer group no longer exists: " + name);
    return result;
}

bool NodeManagerMixerGraph::shouldWork() const {
    auto manager = manager_.lock();
    return manager && manager->shouldWork();
}

bool NodeManagerMixerGraph::nodeExists(const std::string& name) const {
    return node_if_exists(name) != nullptr;
}

std::string NodeManagerMixerGraph::nodeType(const std::string& name) const {
    return node(name)->parameters().value("type", std::string());
}

bool NodeManagerMixerGraph::hasCreatedParam(const std::string& name, const std::string& key) const {
    auto live = node(name);
    return live->node() && live->parameters().contains(key);
}

void NodeManagerMixerGraph::setObject(const std::string& name, const std::string& key, const Parameters& value) {
    auto control = std::dynamic_pointer_cast<IInputsObjects>(node(name)->node());
    if (!control) throw Error("node has no runtime controls");
    control->setObject(key, value);
}

bool NodeManagerMixerGraph::setObjectIfCreated(const std::string& name, const std::string& key,
                                               const Parameters& value) {
    auto wrapper = node_if_exists(name);
    if (!wrapper)
        throw Error("Node " + name + " doesn't exist.");

    wrapper->parameters()[key] = value;
    if (!wrapper->node())
        return false;

    try {
        auto control = std::dynamic_pointer_cast<IInputsObjects>(wrapper->node());
        if (!control) throw Error("Node has no runtime controls: " + name);
        control->setObject(key, value);
        return true;
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "Node not created")
            return false;
        throw;
    }
}

Parameters NodeManagerMixerGraph::getObject(const std::string& name, const std::string& key) const {
    return node(name)->getObject(key);
}

bool NodeManagerMixerGraph::tryGetObject(const std::string& name, const std::string& key, Parameters& out) const {
    auto wrapper = node_if_exists(name);
    if (!wrapper)
        return false;
    return wrapper->getObjectTry(key, out);
}

void NodeManagerMixerGraph::setParam(const std::string& name, const std::string& key, const std::string& value) {
    node(name)->parameters()[key] = value;
}

void NodeManagerMixerGraph::startGroup(const std::string& name) {
    group(name)->startNodes();
}

void NodeManagerMixerGraph::stopGroup(const std::string& name) {
    group(name)->stopNodes();
}

void NodeManagerMixerGraph::stopGroupAndWait(const std::string& name) {
    group(name)->stopNodesAndWait();
}

void NodeManagerMixerGraph::stopNode(const std::string& name, bool inhibit) {
    node(name)->stop(inhibit);
}

void NodeManagerMixerGraph::resetInput(const std::string& name) {
    if (name.empty())
        return;
    auto wrapper = node_if_exists(name);
    if (!wrapper || !wrapper->node())
        return;
    if (auto reset = std::dynamic_pointer_cast<IInputReset>(wrapper->node()))
        reset->resetInput();
}

namespace {

bool nodeConsumesEdge(const std::shared_ptr<NodeWrapper>& node, const std::string& edge_name) {
    if (!node)
        return false;
    const auto& params = node->parameters();
    if (!params.count("src"))
        return false;
    for (const auto& src_name : jsonToStringList(params["src"])) {
        if (src_name == edge_name)
            return true;
    }
    return false;
}

} // namespace

std::shared_ptr<NodeWrapper> NodeManagerMixerGraph::workingConsumerForEdge(const std::string& edge_name) const {
    for (const auto& [name, weak] : nodes_) {
        auto node = weak.lock();
        if (node && node->isWorking() && nodeConsumesEdge(node, edge_name))
            return node;
    }
    return nullptr;
}

void NodeManagerMixerGraph::clearEdgeIfIdle(const std::string& name) {
    auto edge = edges_->findAny(name);
    if (!edge)
        return;

    int occupied = edge->occupied();
    if (occupied <= 0)
        return;

    auto active_consumer = workingConsumerForEdge(name);
    if (active_consumer) {
        logstream << "mixer: leaving active wipe edge " << name
                  << " unflushed (" << occupied << " queued, consumer="
                  << active_consumer->name() << ")";
        return;
    }

    logstream << "mixer: flushing stale wipe edge " << name
              << " (" << occupied << " queued)";
    edge->clear();
}

av::Timestamp NodeManagerMixerGraph::edgeLastTimestamp(const std::string& name) const {
    if (name.empty())
        return NOTS;
    auto edge = edges_->findAny(name);
    return edge ? edge->lastTS() : NOTS;
}

std::string NodeManagerMixerGraph::firstDstEdgeName(const std::string& node_name) const {
    auto live = node_if_exists(node_name);
    if (!live)
        return "";
    const auto& params = live->parameters();
    if (!params.count("dst"))
        return "";
    auto names = jsonToStringList(params["dst"]);
    return names.empty() ? "" : names.front();
}

std::string NodeManagerMixerGraph::edgeNameAt(const std::string& node_name, const std::string& param_name,
                                              size_t index) const {
    auto live = node_if_exists(node_name);
    if (!live)
        return "";
    const auto& params = live->parameters();
    if (!params.count(param_name))
        return "";
    auto names = jsonToStringList(params[param_name]);
    if (index >= names.size())
        return "";
    auto it = names.begin();
    std::advance(it, index);
    return *it;
}

int NodeManagerMixerGraph::routerOutputCount(const std::string& router_name) const {
    auto wrapper = node_if_exists(router_name);
    if (!wrapper)
        throw Error("mixer.init_routes: node " + router_name + " doesn't exist");
    const auto& params = wrapper->parameters();
    if (!params.count("dst"))
        throw Error("mixer.init_routes: router " + router_name + " has no dst parameter");
    return static_cast<int>(jsonToStringList(params["dst"]).size());
}

int NodeManagerMixerGraph::routerOutputIndexFromLabel(const std::string& router_name,
                                                      const std::string& label) const {
    auto wrapper = node_if_exists(router_name);
    if (!wrapper)
        throw Error("mixer.routed_source: node " + router_name + " doesn't exist");
    const auto& params = wrapper->parameters();
    if (!params.count("dst"))
        throw Error("mixer.routed_source: router " + router_name + " has no dst parameter");
    if (!params.count("labels"))
        throw Error("mixer.routed_source: router " + router_name + " has no labels parameter");

    auto dst = jsonToStringList(params["dst"]);
    auto labels_list = jsonToStringList(params["labels"]);
    std::vector<std::string> labels(labels_list.begin(), labels_list.end());
    if (dst.size() != labels.size()) {
        throw Error("mixer.routed_source: router " + router_name + " labels size " +
                    std::to_string(labels.size()) + " does not match dst size " +
                    std::to_string(dst.size()));
    }
    if (!label.empty() && std::all_of(label.begin(), label.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        int output_index = std::stoi(label);
        if (output_index < 0 || output_index >= (int)labels.size()) {
            throw Error("mixer.routed_source: router " + router_name +
                        " output index " + label + " is out of range");
        }
        return output_index;
    }
    auto it = std::find(labels.begin(), labels.end(), label);
    if (it == labels.end()) {
        throw Error("mixer.routed_source: router " + router_name +
                    " has no output label " + label);
    }
    return static_cast<int>(std::distance(labels.begin(), it));
}

std::shared_ptr<CutLatencyObserver> NodeManagerMixerGraph::cutLatencyObserver(const std::string& name) const {
    return std::dynamic_pointer_cast<CutLatencyObserver>(node(name)->node());
}

} // namespace avp::mixer
