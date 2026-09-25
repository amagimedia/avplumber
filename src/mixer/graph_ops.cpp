#include "graph_ops.hpp"
#include "../graph_interfaces.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iterator>
#include <thread>

namespace avp::mixer::graph {

void resetInputIf(const std::shared_ptr<NodeManager>& nodes, const std::string& name) {
    if (name.empty())
        return;
    auto w = nodes->node_if_exists(name);
    if (!w || !w->node())
        return;
    if (auto r = std::dynamic_pointer_cast<IInputReset>(w->node()))
        r->resetInput();
}

void resetSlotNormFps(const std::shared_ptr<NodeManager>& nodes, const MixerState& st) {
    resetInputIf(nodes, st.slot_a.norm_ts_name);
    resetInputIf(nodes, st.slot_b.norm_ts_name);
}

bool nodeWorkingIfExists(const std::shared_ptr<NodeManager>& nodes, const std::string& name) {
    if (name.empty())
        return false;
    auto w = nodes->node_if_exists(name);
    return w && w->isWorking();
}

static bool nodeConsumesEdge(const std::shared_ptr<NodeWrapper>& node, const std::string& edge_name) {
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

std::shared_ptr<NodeWrapper> workingConsumerForEdge(const std::shared_ptr<NodeManager>& nodes,
                                                    const std::string& edge_name) {
    for (const auto& [_, node] : nodes->allNodes()) {
        if (node && node->isWorking() && nodeConsumesEdge(node, edge_name))
            return node;
    }
    return nullptr;
}

bool setNodeObjectIfCreated(const std::shared_ptr<NodeManager>& nodes,
                            const std::string& node_name,
                            const std::string& key,
                            const Parameters& value) {
    auto wrapper = nodes->node_if_exists(node_name);
    if (!wrapper)
        throw Error("Node " + node_name + " doesn't exist.");

    wrapper->parameters()[key] = value;
    if (!wrapper->node())
        return false;

    try {
        wrapper->setObject(key, value);
        return true;
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "Node not created")
            return false;
        throw;
    }
}

int edgeOccupiedIfExists(const std::shared_ptr<NodeManager>& nodes, const std::string& name) {
    if (name.empty())
        return 0;
    auto e = nodes->edges()->findAny(name);
    return e ? e->occupied() : 0;
}

std::string firstDstEdgeName(const std::shared_ptr<NodeManager>& nodes, const std::string& node_name) {
    auto node = nodes->node_if_exists(node_name);
    if (!node)
        return "";
    const auto& params = node->parameters();
    if (!params.count("dst"))
        return "";
    auto names = jsonToStringList(params["dst"]);
    return names.empty() ? "" : names.front();
}

std::string edgeNameAt(const std::shared_ptr<NodeManager>& nodes,
                       const std::string& node_name,
                       const std::string& param_name,
                       size_t index) {
    auto node = nodes->node_if_exists(node_name);
    if (!node)
        return "";
    const auto& params = node->parameters();
    if (!params.count(param_name))
        return "";
    auto names = jsonToStringList(params[param_name]);
    if (index >= names.size())
        return "";
    auto it = names.begin();
    std::advance(it, index);
    return *it;
}

av::Timestamp edgeLastTsIfExists(const std::shared_ptr<NodeManager>& nodes, const std::string& name) {
    if (name.empty())
        return NOTS;
    auto e = nodes->edges()->findAny(name);
    return e ? e->lastTS() : NOTS;
}

WipeReadyResult waitForWipeOverlayReady(const std::shared_ptr<NodeManager>& nodes,
                                        const std::string& edge_name,
                                        av::Timestamp initial_ts,
                                        int64_t earliest_visible_pts_ms,
                                        const std::shared_ptr<MixerState>& state, uint64_t generation,
                                        int64_t timeout_ms) {
    WipeReadyResult result;
    while (result.waited_ms < timeout_ms) {
        if (state->transition_generation.load() != generation) return result;
        const bool time_ready = wallclock.pts() >= earliest_visible_pts_ms;
        av::Timestamp ts = edgeLastTsIfExists(nodes, edge_name);
        const bool frame_ready = ts.isValid() && (!initial_ts.isValid() || ts > initial_ts);
        if (time_ready && frame_ready) {
            result.ready = true;
            result.ready_ts = ts;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        result.waited_ms += kPollMs;
    }
    result.ready_ts = edgeLastTsIfExists(nodes, edge_name);
    return result;
}

bool overlayCommandCurrent(const std::shared_ptr<MixerState>& state, uint64_t generation) {
    return state->overlay_generation.load(std::memory_order_acquire) == generation;
}

OverlayReadyResult waitForOverlayBranchReady(const std::shared_ptr<NodeManager>& nodes,
                                             const std::shared_ptr<MixerState>& state,
                                             uint64_t generation,
                                             const std::string& edge_name,
                                             av::Timestamp initial_ts,
                                             av::Timestamp minimum_ts,
                                             int64_t timeout_ms,
                                             int64_t poll_ms) {
    OverlayReadyResult result;
    timeout_ms = std::max<int64_t>(0, timeout_ms);
    poll_ms = std::max<int64_t>(1, poll_ms);
    while (result.waited_ms < timeout_ms) {
        if (!overlayCommandCurrent(state, generation)) {
            result.cancelled = true;
            return result;
        }
        av::Timestamp ts = edgeLastTsIfExists(nodes, edge_name);
        const bool fresh = ts.isValid() && (!initial_ts.isValid() || ts > initial_ts);
        const bool monotonic = !minimum_ts.isValid() || (ts.isValid() && !(ts < minimum_ts));
        if (fresh && monotonic) {
            result.ready = true;
            result.ready_ts = ts;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        result.waited_ms += poll_ms;
    }
    result.ready_ts = edgeLastTsIfExists(nodes, edge_name);
    return result;
}

static std::shared_ptr<NodeWrapper> requireNodeWrapper(const std::shared_ptr<NodeManager>& nodes,
                                                const std::string& node_name,
                                                const std::string& context) {
    auto wrapper = nodes->node_if_exists(node_name);
    if (!wrapper)
        throw Error(context + ": node " + node_name + " doesn't exist");
    return wrapper;
}

static std::vector<std::string> routerLabels(const std::shared_ptr<NodeManager>& nodes, const std::string& router_name) {
    auto wrapper = requireNodeWrapper(nodes, router_name, "mixer.routed_source");
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
    return labels;
}

int routerOutputCount(const std::shared_ptr<NodeManager>& nodes, const std::string& router_name) {
    auto wrapper = requireNodeWrapper(nodes, router_name, "mixer.init_routes");
    const auto& params = wrapper->parameters();
    if (!params.count("dst"))
        throw Error("mixer.init_routes: router " + router_name + " has no dst parameter");
    return static_cast<int>(jsonToStringList(params["dst"]).size());
}

int routerOutputIndexFromLabel(const std::shared_ptr<NodeManager>& nodes,
                               const std::string& router_name,
                               const std::string& label) {
    auto labels = routerLabels(nodes, router_name);
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

}
