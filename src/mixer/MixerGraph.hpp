#pragma once
#include "../graph_mgmt.hpp"

namespace avp::mixer {

// Resolve the graph on the control thread. Frame processing must not acquire
// NodeManager's lock: shutdown holds it while joining the processing threads.
// Weak handles also let the graph be destroyed without a selector/state cycle.
class MixerGraph {
    std::unordered_map<std::string, std::weak_ptr<NodeWrapper>> nodes_;
    std::unordered_map<std::string, std::weak_ptr<NodeGroup>> groups_;
    std::shared_ptr<EdgeManager> edges_;
    std::weak_ptr<NodeManager> manager_;
    InstanceData& instance_;
public:
    explicit MixerGraph(const std::shared_ptr<NodeManager>& manager)
        : edges_(manager->edges()), manager_(manager), instance_(manager->instanceData()) {
        for (const auto& [name, node] : manager->allNodes()) {
            nodes_.emplace(name, node);
            if (node->parameters().contains("group"))
                groups_[node->parameters().at("group").get<std::string>()] = node->group();
        }
    }
    std::shared_ptr<NodeWrapper> node_if_exists(const std::string& name) const {
        auto it = nodes_.find(name);
        return it == nodes_.end() ? nullptr : it->second.lock();
    }
    std::shared_ptr<NodeWrapper> node(const std::string& name) const {
        auto result = node_if_exists(name);
        if (!result) throw Error("Mixer node no longer exists: " + name);
        return result;
    }
    std::shared_ptr<NodeGroup> group(const std::string& name) const {
        auto it = groups_.find(name);
        auto result = it == groups_.end() ? nullptr : it->second.lock();
        if (!result) throw Error("Mixer group no longer exists: " + name);
        return result;
    }
    auto allNodes() const {
        std::unordered_map<std::string, std::shared_ptr<NodeWrapper>> result;
        for (const auto& [name, weak] : nodes_)
            if (auto node = weak.lock()) result.emplace(name, std::move(node));
        return result;
    }
    const auto& edges() const { return edges_; }
    InstanceData& instanceData() const { return instance_; }
    bool shouldWork() const {
        auto manager = manager_.lock();
        return manager && manager->shouldWork();
    }
};

} // namespace avp::mixer
