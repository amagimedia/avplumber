#pragma once
// NodeManager-backed MixerGraph. Compiled into the C++ binary only.
#include "MixerGraph.hpp"
#include "../graph_mgmt.hpp"
#include <unordered_map>

namespace avp::mixer {

class NodeManagerMixerGraph : public MixerGraph {
    std::unordered_map<std::string, std::weak_ptr<NodeWrapper>> nodes_;
    std::unordered_map<std::string, std::weak_ptr<NodeGroup>> groups_;
    std::shared_ptr<EdgeManager> edges_;
    std::weak_ptr<NodeManager> manager_;
    InstanceData& instance_;

    std::shared_ptr<NodeWrapper> node_if_exists(const std::string& name) const;
    std::shared_ptr<NodeWrapper> node(const std::string& name) const;
    std::shared_ptr<NodeGroup> group(const std::string& name) const;
    std::shared_ptr<NodeWrapper> workingConsumerForEdge(const std::string& edge_name) const;

public:
    explicit NodeManagerMixerGraph(const std::shared_ptr<NodeManager>& manager);

    InstanceData& instanceData() const override { return instance_; }
    bool shouldWork() const override;
    bool nodeExists(const std::string& name) const override;
    std::string nodeType(const std::string& name) const override;
    bool hasCreatedParam(const std::string& name, const std::string& key) const override;

    void setObject(const std::string& name, const std::string& key, const Parameters& value) override;
    bool setObjectIfCreated(const std::string& name, const std::string& key, const Parameters& value) override;
    Parameters getObject(const std::string& name, const std::string& key) const override;
    bool tryGetObject(const std::string& name, const std::string& key, Parameters& out) const override;
    void setParam(const std::string& name, const std::string& key, const std::string& value) override;

    void startGroup(const std::string& name) override;
    void stopGroup(const std::string& name) override;
    void stopGroupAndWait(const std::string& name) override;
    void stopNode(const std::string& name, bool inhibit) override;
    void resetInput(const std::string& name) override;
    void clearEdgeIfIdle(const std::string& edge) override;

    av::Timestamp edgeLastTimestamp(const std::string& edge) const override;
    std::string firstDstEdgeName(const std::string& node) const override;
    std::string edgeNameAt(const std::string& node, const std::string& param, size_t index) const override;
    int routerOutputCount(const std::string& router) const override;
    int routerOutputIndexFromLabel(const std::string& router, const std::string& label) const override;

    std::shared_ptr<CutLatencyObserver> cutLatencyObserver(const std::string& name) const override;
};

} // namespace avp::mixer
