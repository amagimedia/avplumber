#pragma once
// Operations the orchestrator uses. The C++ process implements them on
// NodeManagerMixerGraph; a Rust graph implements the same virtuals over the C ABI.
// This header does not mention NodeManager, so a Rust link does not pull it in.
#include "primitives/CutLatencyProbe.hpp"
#include <cstddef>
#include <memory>
#include <string>

class InstanceData;

namespace avp::mixer {

class MixerGraph {
public:
    virtual ~MixerGraph() = default;

    virtual InstanceData& instanceData() const = 0;
    virtual bool shouldWork() const = 0;
    virtual bool nodeExists(const std::string& name) const = 0;
    virtual std::string nodeType(const std::string& name) const = 0;
    virtual bool hasCreatedParam(const std::string& name, const std::string& key) const = 0;

    virtual void setObject(const std::string& name, const std::string& key, const Parameters& value) = 0;
    /// Stores `value` on the node parameters, then applies it. False when the live
    /// object is not created yet (including a "Node not created" exception).
    virtual bool setObjectIfCreated(const std::string& name, const std::string& key,
                                    const Parameters& value) = 0;
    virtual Parameters getObject(const std::string& name, const std::string& key) const = 0;
    virtual bool tryGetObject(const std::string& name, const std::string& key, Parameters& out) const = 0;
    virtual void setParam(const std::string& name, const std::string& key, const std::string& value) = 0;

    virtual void startGroup(const std::string& name) = 0;
    virtual void stopGroup(const std::string& name) = 0;
    virtual void stopGroupAndWait(const std::string& name) = 0;
    virtual void stopNode(const std::string& name, bool inhibit) = 0;
    virtual void resetInput(const std::string& name) = 0;

    /// Wipe flush of one edge. An active consumer is left in place and logged.
    virtual void clearEdgeIfIdle(const std::string& edge) = 0;

    virtual av::Timestamp edgeLastTimestamp(const std::string& edge) const = 0;
    virtual std::string firstDstEdgeName(const std::string& node) const = 0;
    virtual std::string edgeNameAt(const std::string& node, const std::string& param, size_t index) const = 0;
    virtual int routerOutputCount(const std::string& router) const = 0;
    virtual int routerOutputIndexFromLabel(const std::string& router, const std::string& label) const = 0;

    virtual std::shared_ptr<CutLatencyObserver> cutLatencyObserver(const std::string& name) const = 0;
};

} // namespace avp::mixer
