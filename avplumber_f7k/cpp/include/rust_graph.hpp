#pragma once
// MixerGraph over the Rust instance. NodeManager is not linked.
#include "mixer/MixerGraph.hpp"
#include "avplumber_core.h"

#include <string>
#include <unordered_map>

namespace avp::mixer {

class RustMixerGraph : public MixerGraph {
    AvpCore* core_;
    std::unordered_map<std::string, Parameters> params_;

    void call_set(const std::string& name, const std::string& key, const Parameters& value);
    Parameters call_get(const std::string& name, const std::string& key) const;

public:
    explicit RustMixerGraph(AvpCore* core) : core_(core) {}

    InstanceData& instanceData() const override;
    bool shouldWork() const override { return true; }
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
