#include "mixer_orchestrator.hpp"
#include "avutils.hpp"
#include "graph_interfaces.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <iterator>
#include <sstream>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <utility>

struct MixerTransitionScheduler::Impl {
    struct Task {
        std::chrono::steady_clock::time_point when;
        uint64_t seq = 0;
        std::string label;
        std::function<void()> run;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Task> tasks;
    bool stopping = false;
    uint64_t next_seq = 0;
    std::thread worker;

    static bool before(const Task& a, const Task& b) {
        if (a.when != b.when)
            return a.when < b.when;
        return a.seq < b.seq;
    }

    void workerLoop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            if (tasks.empty()) {
                if (stopping)
                    break;
                cv.wait(lock, [&] { return stopping || !tasks.empty(); });
                continue;
            }

            auto next = std::min_element(tasks.begin(), tasks.end(), before);
            auto now = std::chrono::steady_clock::now();
            if (next->when > now) {
                cv.wait_until(lock, next->when);
                continue;
            }

            Task task = std::move(*next);
            tasks.erase(next);
            lock.unlock();
            try {
                task.run();
            } catch (const std::exception& e) {
                logstream << "mixer transition task " << task.label << " failed: " << e.what();
            } catch (...) {
                logstream << "mixer transition task " << task.label << " failed with unknown exception";
            }
            lock.lock();
        }
    }
};

MixerTransitionScheduler::MixerTransitionScheduler()
    : impl_(std::make_unique<Impl>()) {
    impl_->worker = start_thread("mixer transitions", [impl = impl_.get()] {
        impl->workerLoop();
    });
}

MixerTransitionScheduler::~MixerTransitionScheduler() {
    shutdown();
}

void MixerTransitionScheduler::post(std::string label, std::function<void()> task) {
    postAfter(std::move(label), 0, std::move(task));
}

void MixerTransitionScheduler::postAfter(std::string label, int64_t delay_ms, std::function<void()> task) {
    if (!task)
        throw Error("mixer transition scheduler: empty task");

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping)
        throw Error("mixer transition scheduler is stopped");

    Impl::Task queued;
    queued.when = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(std::max<int64_t>(0, delay_ms));
    queued.seq = impl_->next_seq++;
    queued.label = std::move(label);
    queued.run = std::move(task);
    impl_->tasks.push_back(std::move(queued));
    impl_->cv.notify_one();
}

void MixerTransitionScheduler::shutdown() {
    if (!impl_)
        return;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
        impl_->tasks.clear();
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable())
        impl_->worker.join();
}

namespace {

constexpr int64_t kWipeSwitchGraceMs = 500;
constexpr int64_t kFadeColdPrepMs = 700;
constexpr int64_t kWipeReadyPollMs = 5;
constexpr int64_t kWipeReadyTimeoutMs = 5000;
constexpr int kOverlayDirectInput = 0;
constexpr int kOverlayCompositedInput = 1;

void resetInputIf(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return;
    auto w = nodes->node_if_exists(name);
    if (!w || !w->node())
        return;
    if (auto r = std::dynamic_pointer_cast<IInputReset>(w->node()))
        r->resetInput();
}

/// After a PGM path change, slot compositors may have been idle (`active_inputs=0`) while cameras
/// advanced in PTS; `force_fps` on `norm_*` would otherwise compare the next real frame to a stale
/// grid and produce a big `Discontinuity` jump with a duplicate burst across the cut.
///
/// Deliberately NOT resetting the final post-`out_sel` `force_fps` (e.g. `mixer_norm_fps`): it is
/// the last VFR guard before the encoder and resetting it would drop that guarantee.
void resetSlotNormFps(std::shared_ptr<NodeManager> nodes, const MixerState& st) {
    resetInputIf(nodes, st.slot_a.norm_ts_name);
    resetInputIf(nodes, st.slot_b.norm_ts_name);
}

bool nodeWorkingIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return false;
    auto w = nodes->node_if_exists(name);
    return w && w->isWorking();
}

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

std::shared_ptr<NodeWrapper> workingConsumerForEdge(std::shared_ptr<NodeManager> nodes,
                                                    const std::string& edge_name) {
    for (const auto& [_, node] : nodes->allNodes()) {
        if (node && node->isWorking() && nodeConsumesEdge(node, edge_name))
            return node;
    }
    return nullptr;
}

bool setNodeObjectIfCreated(std::shared_ptr<NodeManager> nodes,
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

int edgeOccupiedIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return 0;
    auto e = nodes->edges()->findAny(name);
    return e ? e->occupied() : 0;
}

std::string firstDstEdgeName(std::shared_ptr<NodeManager> nodes, const std::string& node_name) {
    auto node = nodes->node_if_exists(node_name);
    if (!node)
        return "";
    const auto& params = node->parameters();
    if (!params.count("dst"))
        return "";
    auto names = jsonToStringList(params["dst"]);
    return names.empty() ? "" : names.front();
}

std::string edgeNameAt(std::shared_ptr<NodeManager> nodes,
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

av::Timestamp edgeLastTsIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return NOTS;
    auto e = nodes->edges()->findAny(name);
    return e ? e->lastTS() : NOTS;
}

struct WipeReadyResult {
    bool ready = false;
    int64_t waited_ms = 0;
    av::Timestamp ready_ts = NOTS;
};

struct OverlayReadyResult {
    bool ready = false;
    bool cancelled = false;
    int64_t waited_ms = 0;
    av::Timestamp ready_ts = NOTS;
};

WipeReadyResult waitForWipeOverlayReady(std::shared_ptr<NodeManager> nodes,
                                        const std::string& edge_name,
                                        av::Timestamp initial_ts,
                                        int64_t earliest_visible_pts_ms) {
    WipeReadyResult result;
    while (result.waited_ms < kWipeReadyTimeoutMs) {
        const bool time_ready = wallclock.pts() >= earliest_visible_pts_ms;
        av::Timestamp ts = edgeLastTsIfExists(nodes, edge_name);
        const bool frame_ready = ts.isValid() && (!initial_ts.isValid() || ts > initial_ts);
        if (time_ready && frame_ready) {
            result.ready = true;
            result.ready_ts = ts;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kWipeReadyPollMs));
        result.waited_ms += kWipeReadyPollMs;
    }
    result.ready_ts = edgeLastTsIfExists(nodes, edge_name);
    return result;
}

bool overlayCommandCurrent(const std::shared_ptr<MixerState>& state, uint64_t generation) {
    return state->overlay_generation.load(std::memory_order_acquire) == generation;
}

OverlayReadyResult waitForOverlayBranchReady(std::shared_ptr<NodeManager> nodes,
                                             std::shared_ptr<MixerState> state,
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

std::shared_ptr<NodeWrapper> requireNodeWrapper(std::shared_ptr<NodeManager> nodes,
                                                const std::string& node_name,
                                                const std::string& context) {
    auto wrapper = nodes->node_if_exists(node_name);
    if (!wrapper)
        throw Error(context + ": node " + node_name + " doesn't exist");
    return wrapper;
}

std::vector<std::string> routerLabels(std::shared_ptr<NodeManager> nodes, const std::string& router_name) {
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

int routerOutputCount(std::shared_ptr<NodeManager> nodes, const std::string& router_name) {
    auto wrapper = requireNodeWrapper(nodes, router_name, "mixer.init_routes");
    const auto& params = wrapper->parameters();
    if (!params.count("dst"))
        throw Error("mixer.init_routes: router " + router_name + " has no dst parameter");
    return static_cast<int>(jsonToStringList(params["dst"]).size());
}

int routerOutputIndexFromLabel(std::shared_ptr<NodeManager> nodes,
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

Parameters routesToParameters(const std::vector<int>& routes) {
    Parameters arr = Parameters::array();
    for (int input_index : routes)
        arr.push_back(input_index);
    return arr;
}

void ensureRouteTableSize(MixerState& st, const std::string& router_name) {
    int count = st.router_output_counts[router_name];
    auto& routes = st.router_routes[router_name];
    if ((int)routes.size() != count)
        routes.assign(count, -1);
}

std::unordered_map<std::string, std::vector<int>> currentRouterTables(MixerState& st) {
    std::unordered_map<std::string, std::vector<int>> tables;
    for (const auto& [router_name, count] : st.router_output_counts) {
        ensureRouteTableSize(st, router_name);
        tables[router_name] = st.router_routes[router_name];
        if ((int)tables[router_name].size() != count)
            tables[router_name].assign(count, -1);
    }
    return tables;
}

int routeOutputForSlot(const MixerState::SourceInfo& info, bool is_slot_a) {
    return is_slot_a ? info.route_output_a : info.route_output_b;
}

std::string routeOutputLabelForSlot(const MixerState::SourceInfo& info, bool is_slot_a) {
    return is_slot_a ? info.route_output_label_a : info.route_output_label_b;
}

void setRoutedSlotInTables(MixerState& st,
                           std::unordered_map<std::string, std::vector<int>>& tables,
                           bool is_slot_a,
                           const SceneDefinition* scene) {
    for (const auto& [src_name, info] : st.sources) {
        if (!info.routed)
            continue;
        const int output_index = routeOutputForSlot(info, is_slot_a);
        if (output_index < 0)
            continue;

        auto& routes = tables[info.router_node_name];
        const int count = st.router_output_counts[info.router_node_name];
        if ((int)routes.size() != count)
            routes.assign(count, -1);
        if (output_index >= (int)routes.size())
            throw Error("mixer: routed source " + src_name + " output index " +
                        std::to_string(output_index) + " (" +
                        routeOutputLabelForSlot(info, is_slot_a) + ") exceeds router " +
                        info.router_node_name + " route table");

        routes[output_index] = -1;
        if (!scene || !scene->sources.count(src_name))
            continue;

        auto route_it = scene->routes.find(src_name);
        if (route_it == scene->routes.end()) {
            throw Error("mixer: scene " + scene->name + " uses routed source " +
                        src_name + " without an explicit route");
        }
        routes[output_index] = route_it->second;
    }
}


/// One cuda_rect_overlay layer per compositor src index (see mixer.source). Omitted sources use a dummy rect.
Parameters compositorLayersFromScene(const MixerState& st, const SceneDefinition& scene) {
    static const Parameters kUnusedLayer = Parameters({{"dst_x", 0}, {"dst_y", 0}});

    int max_idx = -1;
    for (const auto& [_, info] : st.sources)
        max_idx = std::max(max_idx, info.input_index);

    Parameters arr = Parameters::array();
    for (int i = 0; i <= max_idx; ++i) {
        std::string name_at;
        for (const auto& [name, info] : st.sources) {
            if (info.input_index == i) {
                name_at = name;
                break;
            }
        }
        if (name_at.empty()) {
            arr.push_back(kUnusedLayer);
            continue;
        }
        auto it = scene.sources.find(name_at);
        if (it == scene.sources.end())
            arr.push_back(kUnusedLayer);
        else
            arr.push_back(it->second.layer);
    }
    return arr;
}

class TransitionPrepGuard {
    std::shared_ptr<MixerState> state_;
    bool active_ = true;

public:
    explicit TransitionPrepGuard(std::shared_ptr<MixerState> state)
        : state_(std::move(state)) {}

    ~TransitionPrepGuard() {
        if (active_)
            state_->transition_mode = MixerState::TransitionMode::Idle;
    }

    void release() {
        active_ = false;
    }
};

bool transitionIsCurrent(const std::shared_ptr<MixerState>& state,
                         uint64_t generation,
                         MixerState::TransitionMode mode) {
    return state->transition_generation.load(std::memory_order_acquire) == generation &&
           state->transition_mode.load(std::memory_order_acquire) == mode;
}

} // namespace

MixerOrchestrator::MixerOrchestrator(
    std::shared_ptr<NodeManager> nodes,
    std::shared_ptr<MixerState> state,
    std::shared_ptr<SharedTimeline> timeline,
    std::shared_ptr<MixerTransitionScheduler> scheduler)
    : nodes_(std::move(nodes)),
      state_(std::move(state)),
      timeline_(std::move(timeline)),
      scheduler_(std::move(scheduler)) {}

void MixerOrchestrator::postTransitionTask(std::string label, int64_t delay_ms, std::function<void()> task) {
    if (!scheduler_)
        throw Error("mixer: transition scheduler is not configured");
    scheduler_->postAfter(std::move(label), delay_ms, std::move(task));
}

void MixerOrchestrator::setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value) {
    try {
        auto node = nodes_->node(node_name);
        node->setObject(key, value);
    } catch (const std::exception& e) {
        throw Error("mixer: set " + node_name + "." + key + " failed: " + e.what());
    }
}

void MixerOrchestrator::publishRuntimeObject(const std::string& node_name,
                                             const std::string& key,
                                             const Parameters& value) {
    timeline_->clearKey(node_name, key);
    if (!setNodeObjectIfCreated(nodes_, node_name, key, value)) {
        logstream << "mixer: queued " << node_name << "." << key
                  << " for node not created yet";
    }
    timeline_->set(node_name, key, wallclock.pts(), value);
}

void MixerOrchestrator::publishCameraOtmOutputs(const std::string& otm_name, uint32_t mask) {
    // `one_to_many` with `timeline` uses tlGetRaw("outputs") whenever any entry matches; stale
    // rows (e.g. an old T_cleanup) would override setObject. We drop only the "outputs" key on
    // this OTM channel — not post-scene otms, not source_switcher, not other keys here.
    // cut/fade append new `outputs` rows at T_cleanup *after* loadSceneIntoSlot returns, so
    // those are not cleared by this call. Overlapping mixer commands are rejected by ensureIdle().
    timeline_->clearKey(otm_name, "outputs");
    if (!setNodeObjectIfCreated(nodes_, otm_name, "outputs", Parameters(mask))) {
        logstream << "mixer: queued " << otm_name << ".outputs=" << mask
                  << " for node not created yet";
    }
    timeline_->set(otm_name, "outputs", wallclock.pts(), Parameters(mask));
}

void MixerOrchestrator::setNodeParam(const std::string& node_name, const std::string& param, const std::string& value) {
    auto node = nodes_->node(node_name);
    auto& params = node->parameters();
    params[param] = value;
}

void MixerOrchestrator::autoRestartNode(const std::string& node_name) {
    auto node = nodes_->node(node_name);
    node->stop(false);
}

void MixerOrchestrator::createAndStartNode(const Parameters& params) {
    Parameters p = params;
    nodes_->createNode(p, true, true);
}

void MixerOrchestrator::deleteNodeIfExists(const std::string& name) {
    auto node = nodes_->node_if_exists(name);
    if (node) {
        nodes_->deleteNode(name);
    }
}

void MixerOrchestrator::startGroup(const std::string& group_name) {
    nodes_->group(group_name)->startNodes();
}

void MixerOrchestrator::stopGroup(const std::string& group_name) {
    nodes_->group(group_name)->stopNodes();
}

void MixerOrchestrator::flushWipeEdges() {
    for (const auto& name : state_->wipe_flush_edges) {
        auto edge = nodes_->edges()->findAny(name);
        if (!edge)
            continue;

        int occupied = edge->occupied();
        if (occupied <= 0)
            continue;

        auto active_consumer = workingConsumerForEdge(nodes_, name);
        if (active_consumer) {
            logstream << "mixer: leaving active wipe edge " << name
                      << " unflushed (" << occupied << " queued, consumer="
                      << active_consumer->name() << ")";
            continue;
        }

        logstream << "mixer: flushing stale wipe edge " << name
                  << " (" << occupied << " queued)";
        edge->clear();
    }
}

void MixerOrchestrator::flushSlotEdges(bool is_slot_a) {
    const auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;

    auto clearEdge = [this](const std::string& name) {
        if (name.empty())
            return;
        auto edge = nodes_->edges()->findAny(name);
        if (edge && edge->occupied() > 0) {
            // readerwriterqueue is SPSC: clearing from this control thread would
            // consume the queue concurrently with the node that owns the edge.
            if (!edge->consumer().expired()) {
                logstream << "mixer: leaving live slot edge " << name
                          << " unflushed (" << edge->occupied() << " queued)";
                return;
            }
            logstream << "mixer: flushing stale slot edge " << name
                      << " (" << edge->occupied() << " queued)";
            edge->clear();
        }
    };

    for (const auto& [_, info] : state_->sources) {
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;
        auto node = nodes_->node_if_exists(cs_node);
        if (!node)
            continue;
        const auto& params = node->parameters();
        if (params.count("src")) {
            for (const auto& edge_name : jsonToStringList(params["src"]))
                clearEdge(edge_name);
        }
        if (params.count("dst")) {
            for (const auto& edge_name : jsonToStringList(params["dst"]))
                clearEdge(edge_name);
        }
    }

    for (const std::string& node_name : {slot.compositor_name, slot.norm_ts_name, slot.post_otm_name}) {
        auto node = nodes_->node_if_exists(node_name);
        if (!node)
            continue;
        const auto& params = node->parameters();
        if (params.count("dst")) {
            for (const auto& edge_name : jsonToStringList(params["dst"]))
                clearEdge(edge_name);
        }
    }
}

void MixerOrchestrator::ensureIdle() const {
    auto mode = state_->transition_mode.load();
    if (mode != MixerState::TransitionMode::Idle)
        throw Error("mixer: transition already in progress");
}

int64_t MixerOrchestrator::resolveTransitionStartPts(int64_t requested_start_pts_ms) const {
    int64_t now = wallclock.pts();
    if (requested_start_pts_ms < 0)
        return now;
    int64_t earliest = now + state_->switch_margin_ms;
    if (requested_start_pts_ms < earliest)
        throw Error("mixer: start_pts_ms must be at least " + std::to_string(state_->switch_margin_ms) +
                    "ms in the future");
    return requested_start_pts_ms;
}

void MixerOrchestrator::defineSource(const std::string& name, const std::string& otm_node, int input_index,
                                      const std::string& cs_node_a, const std::string& cs_node_b) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    MixerState::SourceInfo info;
    info.otm_node_name = otm_node;
    info.input_index = input_index;
    info.cs_node_a = cs_node_a;
    info.cs_node_b = cs_node_b;
    state_->sources[name] = std::move(info);
}

void MixerOrchestrator::defineRoutedSource(const std::string& name, const std::string& router_node,
                                           int input_index,
                                           const std::string& route_output_label_a,
                                           const std::string& route_output_label_b,
                                           const std::string& cs_node_a, const std::string& cs_node_b) {
    const int output_count = routerOutputCount(nodes_, router_node);
    const int route_output_a = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_a);
    const int route_output_b = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_b);

    std::lock_guard<std::mutex> lock(state_->mutex);
    MixerState::SourceInfo info;
    info.input_index = input_index;
    info.cs_node_a = cs_node_a;
    info.cs_node_b = cs_node_b;
    info.routed = true;
    info.router_node_name = router_node;
    info.route_output_label_a = route_output_label_a;
    info.route_output_label_b = route_output_label_b;
    info.route_output_a = route_output_a;
    info.route_output_b = route_output_b;
    state_->sources[name] = std::move(info);

    int& stored_output_count = state_->router_output_counts[router_node];
    if (stored_output_count != 0 && stored_output_count != output_count) {
        throw Error("mixer.routed_source: router " + router_node + " output count changed from " +
                    std::to_string(stored_output_count) + " to " +
                    std::to_string(output_count));
    }
    stored_output_count = output_count;
    ensureRouteTableSize(*state_, router_node);
}

void MixerOrchestrator::defineScene(const std::string& name, const SceneDefinition& def) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->scenes[name] = def;
}

void MixerOrchestrator::applyRoutedSceneRoutesForSlot(bool is_slot_a, const SceneDefinition& scene,
                                                       int64_t at_pts_ms, bool immediate) {
    auto tables = currentRouterTables(*state_);
    setRoutedSlotInTables(*state_, tables, is_slot_a, &scene);

    for (const auto& [router_name, routes] : tables) {
        Parameters value = routesToParameters(routes);
        if (immediate) {
            timeline_->clearKey(router_name, "routes");
            if (!setNodeObjectIfCreated(nodes_, router_name, "routes", value)) {
                logstream << "mixer: queued " << router_name
                          << ".routes for router node not created yet";
            }
            state_->router_routes[router_name] = routes;
        }
        timeline_->set(router_name, "routes", at_pts_ms, value);
    }
}

void MixerOrchestrator::publishRoutedRoutesForProgramOnly(bool pgm_is_slot_a,
                                                           const SceneDefinition& scene,
                                                           int64_t at_pts_ms,
                                                           bool immediate) {
    auto tables = currentRouterTables(*state_);
    setRoutedSlotInTables(*state_, tables, true, nullptr);
    setRoutedSlotInTables(*state_, tables, false, nullptr);
    setRoutedSlotInTables(*state_, tables, pgm_is_slot_a, &scene);

    for (const auto& [router_name, routes] : tables) {
        Parameters value = routesToParameters(routes);
        if (immediate) {
            timeline_->clearKey(router_name, "routes");
            if (!setNodeObjectIfCreated(nodes_, router_name, "routes", value)) {
                logstream << "mixer: queued " << router_name
                          << ".routes for router node not created yet";
            }
            state_->router_routes[router_name] = routes;
        }
        timeline_->set(router_name, "routes", at_pts_ms, value);
    }
}

void MixerOrchestrator::initializeRoutedRoutes() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto& [router_name, expected_count] : state_->router_output_counts) {
        const int actual_count = routerOutputCount(nodes_, router_name);
        if (expected_count != actual_count) {
            throw Error("mixer.init_routes: router " + router_name + " expected output count " +
                        std::to_string(expected_count) + " but node dst has " +
                        std::to_string(actual_count));
        }
        ensureRouteTableSize(*state_, router_name);
    }
    if (state_->pgm_scene_name.empty() || !state_->scenes.count(state_->pgm_scene_name))
        return;
    publishRoutedRoutesForProgramOnly(
        state_->pgm_is_slot_a,
        state_->scenes.at(state_->pgm_scene_name),
        wallclock.pts(),
        true);
}

void MixerOrchestrator::rewriteCameraOutputsForSlot(uint32_t slot_bit, const SceneDefinition& scene) {
    uint32_t active = state_->computeActiveInputsMask(scene);
    for (const auto& [src_name, info] : state_->sources) {
        if (info.routed)
            continue;
        Parameters current_val;
        uint32_t mask = nodes_->node(info.otm_node_name)->getObjectTry("outputs", current_val)
                            ? current_val.get<uint32_t>()
                            : 0u;
        mask &= ~slot_bit;
        if (scene.sources.count(src_name) && (active & (1u << (unsigned)info.input_index)))
            mask |= slot_bit;
        publishCameraOtmOutputs(info.otm_node_name, mask);
    }
}

void MixerOrchestrator::loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name) {
    auto& scene = state_->scenes.at(scene_name);
    const auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;

    // The slot being loaded is the broadcast-inactive PVW slot.  Its compositor
    // was previously idled with active_inputs=0, so it may still hold frames on
    // its input edges. If left there, the next activation starts by rendering
    // stale frames and appears to lag behind the scene switch.
    flushSlotEdges(is_slot_a);

    for (const auto& [src_name, layout] : scene.sources) {
        auto src_it = state_->sources.find(src_name);
        if (src_it == state_->sources.end()) continue;
        const auto& info = src_it->second;
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;

        // Only restart the crop/scale node when the graph string actually changed.
        // Restarting a filter_video node tears down and rebuilds its FFmpeg filter
        // graph, which briefly stops producing frames and allocates a new
        // hw_frames_ctx pool.  Downstream filter_video nodes now absorb pool
        // rotations via a semantic hw_frames_ctx comparison so this no longer
        // causes a mid-wipe EXT_NULL gap, but the restart is still a wasted
        // stall and a frame-timing hiccup when the graph string is unchanged.
        const auto& node_params = nodes_->node(cs_node)->parameters();
        const std::string old_graph = node_params.value("graph", std::string(""));
        if (old_graph == layout.crop_scale_graph) {
            logstream << "loadSceneIntoSlot: " << cs_node
                      << " graph unchanged – skipping restart to avoid spurious hw_frames_ctx change";
        } else {
            logstream << "loadSceneIntoSlot: " << cs_node
                      << " graph changed (\"" << old_graph << "\" -> \""
                      << layout.crop_scale_graph << "\") – restarting";
            setNodeParam(cs_node, "graph", layout.crop_scale_graph);
            autoRestartNode(cs_node);
        }
    }

    setNodeObject(slot.compositor_name, "layers", compositorLayersFromScene(*state_, scene));

    uint32_t active_mask = state_->computeActiveInputsMask(scene);
    // Same pattern as camera otms: cuda_rect_overlay reads "active_inputs" from timeline only.
    // clearKey does not touch "layers" or other keys on this compositor channel.
    timeline_->clearKey(slot.compositor_name, "active_inputs");
    setNodeObject(slot.compositor_name, "active_inputs", Parameters(active_mask));
    timeline_->set(slot.compositor_name, "active_inputs", wallclock.pts(), Parameters(active_mask));

    // Drop slot bit for every camera, then enable only sources in scene with active_inputs set.
    // Keeps `outputs` consistent with compositor consumption (no frames into unused inputs).
    const uint32_t slot_bit = is_slot_a ? 1u : 2u;
    rewriteCameraOutputsForSlot(slot_bit, scene);
    applyRoutedSceneRoutesForSlot(is_slot_a, scene, wallclock.pts(), true);

    state_->pvw_scene_name = scene_name;
}

void MixerOrchestrator::scheduleSceneControls(const SceneDefinition& scene, int64_t at_pts_ms) {
    for (const auto& control : scene.controls) {
        timeline_->set(control.node_name, control.key, at_pts_ms, control.value);
        logstream << "mixer scene control: " << control.node_name << "." << control.key
                  << " at " << at_pts_ms << " -> " << control.value;
    }
}

void MixerOrchestrator::preview(const std::string& scene_name) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();
    if (!state_->scenes.count(scene_name))
        throw Error("mixer.preview: unknown scene: " + scene_name);

    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    const auto& slot = state_->pvwSlot();

    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer preview: scene already loaded in PVW: " << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
        resetInputIf(nodes_, slot.norm_ts_name);
    }

    int64_t T_prep = wallclock.pts();
    timeline_->clearKey(slot.post_otm_name, "outputs");
    setNodeObject(slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(slot.post_otm_name, "outputs", T_prep, Parameters(1u));
    logstream << "mixer preview armed: scene=" << scene_name
              << " slot=" << (pvw_is_slot_a ? 'A' : 'B')
              << " post_otm " << slot.post_otm_name << "->1";
}

// ---------------------------------------------------------------------------
// cutInternal: the graph-level work for a hard cut, without touching
// transition_mode or pgm_is_slot_a.  All values are read from pre-flip state.
// Caller must hold state_->mutex.
// Returns the earliest cut PTS (wallclock ms). Cold cuts are gated until the
// incoming direct edge has produced a fresh frame; preloaded PVW cuts only wait
// for the scheduled PTS.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::cutInternal(const std::string& scene_name, int64_t start_pts_ms) {
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer cut: reusing preloaded PVW scene=" << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
    }

    int64_t T_prep = wallclock.pts();
    int64_t T_cut = start_pts_ms;
    const auto& new_slot = state_->pvwSlot();

    // Pre-warm the hidden direct branch before the visible `out_sel` switch. Without this,
    // enabling `post_otm` and switching `out_sel` at the same PTS leaves the newly selected
    // path one pipeline-latency late, so the final encoder-side force_fps repeats the last
    // visible frame for a few ticks across the cut. `one_to_many` with timeline runs in
    // drop_dynamic_ mode, so feeding an inactive direct branch here is safe: `source_switcher`
    // drains and drops those pre-roll frames instead of back-pressuring the slot.
    timeline_->clearKey(new_slot.post_otm_name, "outputs");
    setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(new_slot.post_otm_name, "outputs", T_prep, Parameters(1u));

    logstream << "mixer cut armed: scene=" << scene_name << " earliest T_cut=" << T_cut
              << " post_otm prep " << new_slot.post_otm_name << "->1";

    resetSlotNormFps(nodes_, *state_);

    return T_cut;
}

// ---------------------------------------------------------------------------
// deferredCleanup: flips internal bookkeeping and deletes nodes that can't be
// removed via timeline. The scheduler decides when this runs.
// ---------------------------------------------------------------------------
void MixerOrchestrator::deferredCleanup(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        uint64_t transition_generation,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        std::vector<std::string> nodes_to_delete) {
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Crossfade))
        return;

    for (const auto& name : nodes_to_delete) {
        try {
            auto node = nodes->node_if_exists(name);
            if (node) nodes->deleteNode(name);
        } catch (const std::exception& e) {
            logstream << "mixer: deferred cleanup error deleting " << name << ": " << e.what();
        }
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Crossfade))
        return;
    try {
        const uint32_t pgm_bit = new_pgm_is_slot_a ? 1u : 2u;
        const auto scene_it = state->scenes.find(new_pgm_scene);
        if (scene_it != state->scenes.end()) {
            const SceneDefinition& scene = scene_it->second;
            const uint32_t active = state->computeActiveInputsMask(scene);
            for (const auto& [src_name, info] : state->sources) {
                if (info.routed)
                    continue;
                const bool in_scene = scene.sources.count(src_name) > 0;
                const bool active_input = (active & (1u << (unsigned)info.input_index)) != 0;
                const uint32_t mask = (in_scene && active_input) ? pgm_bit : 0u;
                timeline->clearKey(info.otm_node_name, "outputs");
                setNodeObjectIfCreated(nodes, info.otm_node_name, "outputs", Parameters(mask));
            }
            MixerOrchestrator orch(nodes, state, timeline);
            orch.publishRoutedRoutesForProgramOnly(new_pgm_is_slot_a, scene, wallclock.pts(), true);
            const auto& new_slot = new_pgm_is_slot_a ? state->slot_a : state->slot_b;
            const auto& old_slot = new_pgm_is_slot_a ? state->slot_b : state->slot_a;
            timeline->clearKey(new_slot.post_otm_name, "outputs");
            timeline->clearKey(old_slot.post_otm_name, "outputs");
            timeline->clearKey(new_slot.compositor_name, "active_inputs");
            timeline->clearKey(old_slot.compositor_name, "active_inputs");
            timeline->clearKey(state->source_switcher_name, "active");
            nodes->node(new_slot.post_otm_name)->setObject("outputs", Parameters(1u));
            nodes->node(old_slot.post_otm_name)->setObject("outputs", Parameters(0u));
            nodes->node(new_slot.compositor_name)->setObject("active_inputs", Parameters(active));
            nodes->node(old_slot.compositor_name)->setObject("active_inputs", Parameters(0u));
            nodes->node(state->source_switcher_name)->setObject(
                "active", Parameters(new_pgm_is_slot_a ? 0 : 1));
        }
    } catch (const std::exception& e) {
        logstream << "mixer: deferred cleanup error restoring routing: " << e.what();
    }
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

void MixerOrchestrator::readyCutTask(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        uint64_t transition_generation,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        std::string ready_edge_name,
        av::Timestamp ready_edge_initial_ts,
        int64_t earliest_switch_pts_ms,
        bool require_new_ready_frame) {
    constexpr int64_t kPollMs = 5;
    int64_t waited_ms = 0;
    while (true) {
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Cut))
            return;
        const bool time_ready = wallclock.pts() >= earliest_switch_pts_ms;
        bool edge_ready = !require_new_ready_frame;
        if (require_new_ready_frame) {
            auto edge = nodes->edges()->findAny(ready_edge_name);
            if (!edge) {
                logstream << "mixer ready cut: missing ready edge " << ready_edge_name
                          << " for scene=" << new_pgm_scene;
                std::lock_guard<std::mutex> lock(state->mutex);
                if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Cut))
                    state->transition_mode = MixerState::TransitionMode::Idle;
                return;
            }
            av::Timestamp ts = edge->lastTS();
            edge_ready = ts.isValid() && (!ready_edge_initial_ts.isValid() || ts > ready_edge_initial_ts);
        }
        if (edge_ready && time_ready)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        waited_ms += kPollMs;
    }
    if (waited_ms > 0 || require_new_ready_frame) {
        logstream << "mixer ready cut: scene=" << new_pgm_scene
                  << " waited_ms=" << waited_ms
                  << " require_new_ready_frame=" << (require_new_ready_frame ? "true" : "false");
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Cut))
        return;
    try {
        const uint32_t pgm_bit = new_pgm_is_slot_a ? 1u : 2u;
        const auto scene_it = state->scenes.find(new_pgm_scene);
        if (scene_it != state->scenes.end()) {
            const SceneDefinition& scene = scene_it->second;
            const uint32_t active = state->computeActiveInputsMask(scene);
            const auto& new_slot = new_pgm_is_slot_a ? state->slot_a : state->slot_b;
            const auto& old_slot = new_pgm_is_slot_a ? state->slot_b : state->slot_a;

            timeline->clearKey(state->source_switcher_name, "active");
            nodes->node(state->source_switcher_name)->setObject(
                "active", Parameters(new_pgm_is_slot_a ? 0 : 1));

            for (const auto& [src_name, info] : state->sources) {
                if (info.routed)
                    continue;
                const bool in_scene = scene.sources.count(src_name) > 0;
                const bool active_input = (active & (1u << (unsigned)info.input_index)) != 0;
                const uint32_t mask = (in_scene && active_input) ? pgm_bit : 0u;
                timeline->clearKey(info.otm_node_name, "outputs");
                setNodeObjectIfCreated(nodes, info.otm_node_name, "outputs", Parameters(mask));
            }
            MixerOrchestrator orch(nodes, state, timeline);
            orch.publishRoutedRoutesForProgramOnly(new_pgm_is_slot_a, scene, wallclock.pts(), true);
            timeline->clearKey(new_slot.post_otm_name, "outputs");
            timeline->clearKey(old_slot.post_otm_name, "outputs");
            timeline->clearKey(new_slot.compositor_name, "active_inputs");
            timeline->clearKey(old_slot.compositor_name, "active_inputs");
            nodes->node(new_slot.post_otm_name)->setObject("outputs", Parameters(1u));
            nodes->node(old_slot.post_otm_name)->setObject("outputs", Parameters(0u));
            nodes->node(new_slot.compositor_name)->setObject("active_inputs", Parameters(active));
            nodes->node(old_slot.compositor_name)->setObject("active_inputs", Parameters(0u));
        }
    } catch (const std::exception& e) {
        logstream << "mixer: ready cut error restoring routing: " << e.what();
    }
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

// ---------------------------------------------------------------------------
// cut: PTS-scheduled hard cut.  Graph work + timeline entries happen now;
// state flip is deferred until the timeline entries have taken effect.
// ---------------------------------------------------------------------------
void MixerOrchestrator::cut(const std::string& scene_name, int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    int64_t T_cut = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Cut;
    uint64_t transition_generation = ++state_->transition_generation;
    TransitionPrepGuard prep_guard(state_);
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    bool was_preloaded = state_->pvw_scene_name == scene_name;

    scheduleSceneControls(state_->scenes.at(scene_name), T_cut);
    cutInternal(scene_name, T_cut);

    const auto& new_slot = state_->pvwSlot();
    std::string ready_edge_name = firstDstEdgeName(nodes_, new_slot.post_otm_name);
    auto ready_edge = nodes_->edges()->findAny(ready_edge_name);
    av::Timestamp ready_edge_initial_ts = ready_edge ? ready_edge->lastTS() : NOTS;
    postTransitionTask("mixer.cut.ready", T_cut - wallclock.pts(),
        [nodes = nodes_, state = state_, timeline = timeline_, transition_generation,
         pvw_is_slot_a, scene_name, ready_edge_name, ready_edge_initial_ts, T_cut, was_preloaded] {
            readyCutTask(nodes, state, timeline, transition_generation,
                         pvw_is_slot_a, scene_name, ready_edge_name, ready_edge_initial_ts,
                         T_cut, !was_preloaded);
        });
    prep_guard.release();
}

// ---------------------------------------------------------------------------
// fade: crossfade transition.  All timeline values are computed from the
// pre-flip state.  The state flip + transition node deletion are deferred.
// ---------------------------------------------------------------------------
void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec, int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    int64_t T_start = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Crossfade;
    uint64_t transition_generation = ++state_->transition_generation;
    TransitionPrepGuard prep_guard(state_);

    // Capture all needed values from pre-flip state
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    bool was_preloaded = state_->pvw_scene_name == scene_name;
    uint32_t pvw_bit = state_->pvwOutputBit();
    const auto& target_slot = pvw_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot    = pvw_is_slot_a ? state_->slot_b : state_->slot_a;
    int pvw_sw_idx = state_->pvwSourceSwitcherIndex();

    if (!was_preloaded) {
        T_start = std::max(T_start, wallclock.pts() + kFadeColdPrepMs);
    }

    // 1. Ensure target scene is loaded into PVW slot.
    if (was_preloaded) {
        logstream << "mixer fade: reusing preloaded PVW scene=" << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
    }

    auto& target_scene = state_->scenes.at(scene_name);
    scheduleSceneControls(target_scene, T_start);

    // 2. Create transition_cuda with direction-dependent alpha expression
    std::string progress_expr = "clip((t-" + std::to_string(T_start / 1000.0) +
        ")/" + std::to_string(duration_sec) + "\\,0\\,1)";
    std::string alpha_expr = pvw_is_slot_a ? "1-" + progress_expr : progress_expr;

    std::string slot_a_trans_edge = edgeNameAt(nodes_, state_->slot_a.post_otm_name, "dst", 1);
    std::string slot_b_trans_edge = edgeNameAt(nodes_, state_->slot_b.post_otm_name, "dst", 1);
    std::string transition_out_edge = edgeNameAt(
        nodes_, state_->source_switcher_name, "src", MixerState::transSourceSwitcherIndex());
    if (slot_a_trans_edge.empty() || slot_b_trans_edge.empty() || transition_out_edge.empty())
        throw Error("mixer.fade: graph is missing transition edges");

    std::string transition_node_name = state_->source_switcher_name.empty()
        ? transition_node_name_
        : state_->source_switcher_name + "_transition";

    Parameters trans_params;
    trans_params["type"] = "filter_video";
    trans_params["name"] = transition_node_name;
    trans_params["src"] = Parameters::array({slot_a_trans_edge, slot_b_trans_edge});
    trans_params["dst"] = transition_out_edge;
    trans_params["graph"] = "transition_cuda=alpha='" + alpha_expr + "':eval=frame";
    trans_params["hwaccel"] = state_->hwaccel_name;
    trans_params["defer_preliminary_init"] = true;
    trans_params["group"] = "mixer_trans";
    createAndStartNode(trans_params);

    // 3. Camera routing: applied in loadSceneIntoSlot via rewriteCameraOutputsForSlot

    // 4–5. Timeline: priming post-scene otms (direct+trans) then visible-path switches
    int64_t T_prep = T_start - (was_preloaded ? state_->switch_margin_ms : kFadeColdPrepMs);
    timeline_->set(state_->slot_a.post_otm_name, "outputs", T_prep, Parameters(3u)); // 0b11 warmup
    timeline_->set(state_->slot_b.post_otm_name, "outputs", T_prep, Parameters(3u));

    int64_t T_end = T_start + (int64_t)(duration_sec * 1000);

    // At T_start: switch output to transition
    timeline_->set(state_->source_switcher_name, "active", T_start,
                   Parameters(MixerState::transSourceSwitcherIndex()));
    timeline_->set(state_->slot_a.post_otm_name, "outputs", T_start, Parameters(2u)); // 0b10 trans only
    timeline_->set(state_->slot_b.post_otm_name, "outputs", T_start, Parameters(2u));

    // At T_end: switch output to new PGM direct
    timeline_->set(state_->source_switcher_name, "active", T_end, Parameters(pvw_sw_idx));
    timeline_->set(target_slot.post_otm_name, "outputs", T_end, Parameters(1u));  // 0b01 direct only
    timeline_->set(old_slot.post_otm_name, "outputs", T_end, Parameters(0u));     // idle

    // Camera cleanup at T_cleanup: converge to new-PGM-only bitmasks
    // pvw_bit == post-flip PGM bit (the PVW slot becomes the new PGM)
    int64_t T_cleanup = T_end + 100;
    for (const auto& [src_name, info] : state_->sources) {
        if (info.routed)
            continue;
        uint32_t new_mask = target_scene.sources.count(src_name) ? pvw_bit : 0u;
        timeline_->set(info.otm_node_name, "outputs", T_cleanup, Parameters(new_mask));
    }
    publishRoutedRoutesForProgramOnly(pvw_is_slot_a, target_scene, T_cleanup, false);
    timeline_->set(old_slot.compositor_name, "active_inputs", T_cleanup, Parameters(0u));

    // 6. Deferred cleanup: delete transition node + flip state
    int64_t flip_delay = (T_cleanup - wallclock.pts()) + 300;
    postTransitionTask("mixer.fade.cleanup", flip_delay,
        [nodes = nodes_, state = state_, timeline = timeline_, transition_generation,
         pvw_is_slot_a, scene_name, transition_node_name] {
            deferredCleanup(nodes, state, timeline, transition_generation,
                            pvw_is_slot_a, scene_name,
                            std::vector<std::string>{transition_node_name});
        });
    prep_guard.release();
}

// ---------------------------------------------------------------------------
// runWipeMidpointAndCleanup:
// Phase 1 (midpoint): PVW slot prep + timeline source_switcher (hidden under opaque wipe).
// Phase 2 (end): routing cleanup, tear down wipe chain, flip state.
// ---------------------------------------------------------------------------
void MixerOrchestrator::runWipeMidpointAndCleanup(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        uint64_t transition_generation,
        std::string scene_name,
        bool new_pgm_is_slot_a,
        int64_t remaining_ms) {

    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
        return;

    // --- Phase 1: midpoint - do invisible scene switch under the fully-opaque wipe ---
    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline);

        // Reconfigure PVW slot for the target scene
        orch.loadSceneIntoSlot(new_pgm_is_slot_a, scene_name);

        // Same post-scene OTM flip as cutInternal: out_sel will read PVW `sc*_direct`, so that slot's
        // `one_to_many` must have outputs=1. If it stays 0 (idle default), frames are popped from
        // norm_* with nowhere to go and the wipe path freezes. Stop the old PGM branch to avoid backup.
        int64_t Tw = wallclock.pts();
        const auto& new_slot = state->pvwSlot();
        const auto& old_slot = state->pgmSlot();
        timeline->clearKey(old_slot.post_otm_name, "outputs");
        orch.setNodeObject(old_slot.post_otm_name, "outputs", Parameters(0u));
        timeline->set(old_slot.post_otm_name, "outputs", Tw, Parameters(0u));
        timeline->clearKey(new_slot.post_otm_name, "outputs");
        orch.setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
        timeline->set(new_slot.post_otm_name, "outputs", Tw, Parameters(1u));

        // Switch source_switcher (invisible behind wipe overlay); timeline for consistency with other switches
        int sw = state->pvwSourceSwitcherIndex();
        timeline->set(state->source_switcher_name, "active", Tw, Parameters(sw));
        logstream << "mixer wipe midpoint: Tw=" << Tw << " scene=" << scene_name << " out_sel.active=" << sw
                  << " new_slot post_otm=" << new_slot.post_otm_name << " old_slot post_otm="
                  << old_slot.post_otm_name;
        resetSlotNormFps(nodes, *state);
    } catch (const std::exception& e) {
        logstream << "mixer: wipe midpoint error: " << e.what();
    }

    // --- Phase 2: wipe end – tear down and flip ---
    // Stage 2a: wait for the wipe source to EOF (or until the planned duration elapses).
    bool hit_input_eof = false;
    if (remaining_ms > 0) {
        int64_t waited_ms = 0;
        constexpr int64_t kPollMs = 10;
        while (waited_ms < remaining_ms) {
            if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
                return;
            if (!nodeWorkingIfExists(nodes, state->wipe_input_node_name)) {
                logstream << "mixer wipe: cleanup pulled to wipe EOF after " << waited_ms
                          << "ms of remaining tail";
                hit_input_eof = true;
                break;
            }
            int64_t step_ms = std::min<int64_t>(kPollMs, remaining_ms - waited_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
            waited_ms += step_ms;
        }
    }

    // Stage 2b: after source EOF, the tail of the wipe is still propagating through
    // wipe_demux → wipe_dec → wipe_fmt → wipe_rt → wipe_rt_fps → wipe_overlay (six
    // queues + filter internal buffering). Flipping `wipe_sel` now would cut those
    // tail frames. Wait until the last pre-overlay edge (`wipe_tail_edge`) has been
    // drained by the overlay, then give the overlay a short grace to emit the final
    // blended frames through `wipe_overlay_out` to `wipe_sel`.
    if (hit_input_eof && !state->wipe_tail_edge.empty()) {
        constexpr int64_t kWipeDrainTimeoutMs = 1000;
        constexpr int64_t kWipeDrainPollMs = 10;
        constexpr int64_t kWipeOverlayTailMs = 120; // ~3-4 frames @30fps
        int64_t waited = 0;
        while (waited < kWipeDrainTimeoutMs) {
            if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
                return;
            if (edgeOccupiedIfExists(nodes, state->wipe_tail_edge) == 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kWipeDrainPollMs));
            waited += kWipeDrainPollMs;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kWipeOverlayTailMs));
        logstream << "mixer wipe: drained " << state->wipe_tail_edge << " in " << waited
                  << "ms + " << kWipeOverlayTailMs << "ms overlay tail";
    }

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline);

        int64_t Tw = wallclock.pts();
        if (!state->wipe_otm_name.empty()) {
            timeline->clearKey(state->wipe_otm_name, "outputs");
            orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(1u));
            timeline->set(state->wipe_otm_name, "outputs", Tw, Parameters(1u));
        }
        if (!state->wipe_selector_name.empty()) {
            timeline->clearKey(state->wipe_selector_name, "active");
            orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));
            timeline->set(state->wipe_selector_name, "active", Tw, Parameters(0));
        }
        logstream << "mixer wipe cleanup: Tw=" << Tw << " otm_final.outputs=1 wipe_sel.active=0"
                  << " stop_wipe_group_in_ms=" << kWipeSwitchGraceMs;

        uint32_t new_pgm_bit = state->pvwOutputBit();
        auto& scene = state->scenes.at(scene_name);
        for (const auto& [src_name, info] : state->sources) {
            if (info.routed)
                continue;
            uint32_t mask = scene.sources.count(src_name) ? new_pgm_bit : 0u;
            timeline->set(info.otm_node_name, "outputs", Tw, Parameters(mask));
        }
        orch.publishRoutedRoutesForProgramOnly(new_pgm_is_slot_a, scene, Tw, true);

        const auto& old_slot = state->pgmSlot();
        timeline->set(old_slot.compositor_name, "active_inputs", Tw, Parameters(0u));
    } catch (const std::exception& e) {
        logstream << "mixer: wipe cleanup error: " << e.what();
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kWipeSwitchGraceMs));
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
        return;

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline);

        // Stop the pre-created wipe subgraph only after the direct-path switch
        // has had time to land on the frame timeline. Tearing it down at the
        // same wallclock instant as the switch starves wipe_sel/final_out for a
        // few ticks and the encoder-side force_fps visibly repeats the last wipe frame.
        if (!state->wipe_group_name.empty()) {
            orch.stopGroup(state->wipe_group_name);
            // Release any frames still sitting in wipe pipeline edges so they
            // don't replay at the start of the next wipe.
            orch.flushWipeEdges();
        }

        state->pgm_is_slot_a = new_pgm_is_slot_a;
        state->pgm_scene_name = scene_name;
        state->pvw_scene_name = "";
        state->transition_mode = MixerState::TransitionMode::Idle;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe teardown error: " << e.what();
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
    }
}

// ---------------------------------------------------------------------------
// wipe: media wipe transition.  Uses the static wipe_otm + wipe_selector
// nodes to route through the overlay without edge rewiring.
// The wipe subgraph (group wipe_group_name) is pre-created but not running
// in steady state; it is started here and stopped at the end of the wipe.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::prepareWipe(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        uint64_t transition_generation,
        std::string scene_name,
        std::string wipe_file,
        double duration_sec,
        bool new_pgm_is_slot_a,
        int64_t earliest_visible_pts_ms) {
    std::string overlay_edge_name;
    av::Timestamp overlay_initial_ts = NOTS;
    int64_t T_prep = wallclock.pts();

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return -1;
        MixerOrchestrator orch(nodes, state, timeline);

        overlay_edge_name = edgeNameAt(nodes, state->wipe_selector_name, "src", 1);
        overlay_initial_ts = edgeLastTsIfExists(nodes, overlay_edge_name);

        nodes->node(state->wipe_input_node_name)->stop(true);
        orch.setNodeParam(state->wipe_input_node_name, "url", wipe_file);
        orch.flushWipeEdges();
        resetInputIf(nodes, state->wipe_base_fps_name);
        orch.startGroup(state->wipe_group_name);

        T_prep = wallclock.pts();
        timeline->clearKey(state->wipe_otm_name, "outputs");
        orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(3u));       // 0b11 both direct + wipe_in
        timeline->set(state->wipe_otm_name, "outputs", T_prep, Parameters(3u));
        timeline->clearKey(state->wipe_selector_name, "active");
        orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));    // direct branch while prerolling
        timeline->set(state->wipe_selector_name, "active", T_prep, Parameters(0));

        int64_t total_ms = (int64_t)(duration_sec * 1000);
        int64_t midpoint_ms = total_ms / 2;
        logstream << "mixer wipe: scene=" << scene_name << " file=" << wipe_file << " T_prep=" << T_prep
                  << " requested_visible=" << earliest_visible_pts_ms << " total_ms=" << total_ms << " midpoint_ms=" << midpoint_ms
                  << " new_pgm_slot_" << (new_pgm_is_slot_a ? 'A' : 'B');
    } catch (const std::exception& e) {
        logstream << "mixer: wipe prep error: " << e.what();
        std::lock_guard<std::mutex> lock(state->mutex);
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
        return -1;
    }

    WipeReadyResult ready = waitForWipeOverlayReady(
        nodes, overlay_edge_name, overlay_initial_ts, earliest_visible_pts_ms);
    if (!ready.ready) {
        logstream << "mixer: wipe overlay did not become ready within " << kWipeReadyTimeoutMs
                  << "ms: edge=" << overlay_edge_name << " last_ts=" << ready.ready_ts;
        try {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe)) {
                MixerOrchestrator orch(nodes, state, timeline);
                int64_t Tw = wallclock.pts();
                timeline->clearKey(state->wipe_otm_name, "outputs");
                orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(1u));
                timeline->set(state->wipe_otm_name, "outputs", Tw, Parameters(1u));
                timeline->clearKey(state->wipe_selector_name, "active");
                orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));
                timeline->set(state->wipe_selector_name, "active", Tw, Parameters(0));
                orch.stopGroup(state->wipe_group_name);
                orch.flushWipeEdges();
                state->transition_mode = MixerState::TransitionMode::Idle;
            }
        } catch (const std::exception& e) {
            logstream << "mixer: wipe prep cleanup error: " << e.what();
            if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
                state->transition_mode = MixerState::TransitionMode::Idle;
        }
        return -1;
    }

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return -1;
        MixerOrchestrator orch(nodes, state, timeline);
        int64_t T_visible = wallclock.pts();
        orch.setNodeObject(state->wipe_selector_name, "active", Parameters(1));    // wipe_overlay_out
        timeline->set(state->wipe_selector_name, "active", T_visible, Parameters(1));
        logstream << "mixer wipe overlay ready: edge=" << overlay_edge_name
                  << " waited_ms=" << ready.waited_ms
                  << " ready_ts=" << ready.ready_ts
                  << " T_visible=" << T_visible
                  << " requested_visible=" << earliest_visible_pts_ms;
        return T_visible;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe visible switch error: " << e.what();
        std::lock_guard<std::mutex> lock(state->mutex);
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
        return -1;
    }
}

void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec,
                             int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty())
        throw Error("mixer: wipe requires wipe_otm and wipe_selector nodes (see mixer.init)");
    if (state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
        throw Error("mixer: wipe requires wipe_group and wipe_input_node (see mixer.init)");

    int64_t T_start = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Wipe;
    uint64_t transition_generation = ++state_->transition_generation;
    TransitionPrepGuard prep_guard(state_);
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    scheduleSceneControls(state_->scenes.at(scene_name), T_start);

    int64_t total_ms = (int64_t)(duration_sec * 1000);
    int64_t midpoint_ms = total_ms / 2;
    int64_t remaining_ms = total_ms - midpoint_ms;
    int64_t now_ms = wallclock.pts();
    postTransitionTask("mixer.wipe.prepare", T_start - now_ms,
        [scheduler = scheduler_, nodes = nodes_, state = state_, timeline = timeline_, transition_generation,
         scene_name, wipe_file, duration_sec, pvw_is_slot_a, T_start, midpoint_ms, remaining_ms] {
            int64_t T_visible = prepareWipe(nodes, state, timeline, transition_generation,
                                            scene_name, wipe_file, duration_sec, pvw_is_slot_a,
                                            T_start);
            if (T_visible < 0)
                return;

            scheduler->postAfter("mixer.wipe.midpoint", midpoint_ms,
                [nodes, state, timeline, transition_generation, scene_name, pvw_is_slot_a, remaining_ms] {
                    runWipeMidpointAndCleanup(nodes, state, timeline, transition_generation,
                                              scene_name, pvw_is_slot_a, remaining_ms);
                });
        });
    prep_guard.release();
}

void MixerOrchestrator::setOverlayEnabled(bool enabled, int64_t ready_timeout_ms) {
    std::string source_otm_name;
    std::string overlay_otm_name;
    std::string selector_name;
    std::string candidate_edge_name;
    std::string selector_output_edge_name;
    av::Timestamp visible_ts = NOTS;
    av::Timestamp initial_candidate_ts = NOTS;
    uint64_t generation = 0;
    int64_t timeout_ms = 0;
    int64_t poll_ms = 0;
    const int candidate_input = enabled ? kOverlayCompositedInput : kOverlayDirectInput;

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->overlay_source_otm_name.empty() ||
            state_->overlay_otm_name.empty() ||
            state_->overlay_selector_name.empty()) {
            throw Error("mixer.overlay: overlay nodes are not configured");
        }
        source_otm_name = state_->overlay_source_otm_name;
        overlay_otm_name = state_->overlay_otm_name;
        selector_name = state_->overlay_selector_name;
        timeout_ms = ready_timeout_ms >= 0 ? ready_timeout_ms : state_->overlay_ready_timeout_ms;
        poll_ms = state_->overlay_ready_poll_ms;
        generation = ++state_->overlay_generation;

        selector_output_edge_name = firstDstEdgeName(nodes_, selector_name);
        candidate_edge_name = edgeNameAt(nodes_, selector_name, "src", candidate_input);
        visible_ts = edgeLastTsIfExists(nodes_, selector_output_edge_name);
        initial_candidate_ts = edgeLastTsIfExists(nodes_, candidate_edge_name);

        if (candidate_edge_name.empty()) {
            throw Error("mixer.overlay: selector " + selector_name + " does not expose input " +
                        std::to_string(candidate_input));
        }

        if (!setNodeObjectIfCreated(nodes_, selector_name, "drop_non_monotonic", Parameters(true))) {
            logstream << "mixer.overlay: queued " << selector_name
                      << ".drop_non_monotonic for node not created yet";
        }

        if (enabled) {
            publishRuntimeObject(selector_name, "active", Parameters(kOverlayDirectInput));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(3u));
        } else {
            // Keep both legs fed until the direct leg has caught up with the last
            // visible frame. The selector flips only after the wait below.
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(3u));
        }

        logstream << "mixer.overlay: armed " << (enabled ? "enable" : "disable")
                  << " selector=" << selector_name
                  << " candidate_edge=" << candidate_edge_name
                  << " visible_ts=" << visible_ts
                  << " initial_candidate_ts=" << initial_candidate_ts
                  << " timeout_ms=" << timeout_ms;
    }

    OverlayReadyResult ready = waitForOverlayBranchReady(
        nodes_, state_, generation, candidate_edge_name, initial_candidate_ts,
        visible_ts, timeout_ms, poll_ms);
    if (ready.cancelled) {
        logstream << "mixer.overlay: " << (enabled ? "enable" : "disable")
                  << " superseded before visible switch";
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!overlayCommandCurrent(state_, generation)) {
            logstream << "mixer.overlay: " << (enabled ? "enable" : "disable")
                      << " superseded before finalizing";
            return;
        }

        if (!ready.ready) {
            publishRuntimeObject(selector_name, "active", Parameters(kOverlayDirectInput));
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(0u));
            state_->overlay_enabled = false;
            std::ostringstream msg;
            msg << "mixer.overlay: " << (enabled ? "overlay" : "direct")
                << " branch did not reach monotonic PTS before timeout; edge="
                << candidate_edge_name << " last_ts=" << ready.ready_ts;
            throw Error(msg.str());
        }

        publishRuntimeObject(selector_name, "active", Parameters(candidate_input));
        if (!enabled) {
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(0u));
        }
        state_->overlay_enabled = enabled;
        logstream << "mixer.overlay: " << (enabled ? "enabled" : "disabled")
                  << " waited_ms=" << ready.waited_ms
                  << " ready_ts=" << ready.ready_ts
                  << " visible_ts=" << visible_ts;
    }
}

std::vector<std::string> MixerOrchestrator::sceneNames() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::vector<std::string> names;
    names.reserve(state_->scenes.size());
    for (const auto& [name, _] : state_->scenes)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

Parameters MixerOrchestrator::status() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    Parameters s;
    s["pgm_scene"] = state_->pgm_scene_name;
    s["pvw_scene"] = state_->pvw_scene_name;
    s["pgm_slot"] = state_->pgm_is_slot_a ? "A" : "B";
    s["switch_margin_ms"] = state_->switch_margin_ms;
    s["now_pts_ms"] = wallclock.pts();
    if (!state_->overlay_selector_name.empty()) {
        s["overlay_enabled"] = state_->overlay_enabled;
        s["overlay_selector"] = state_->overlay_selector_name;
    }
    auto mode = state_->transition_mode.load();
    switch (mode) {
        case MixerState::TransitionMode::Idle: s["transition"] = "idle"; break;
        case MixerState::TransitionMode::Cut: s["transition"] = "cut"; break;
        case MixerState::TransitionMode::Crossfade: s["transition"] = "crossfade"; break;
        case MixerState::TransitionMode::Wipe: s["transition"] = "wipe"; break;
    }
    return s;
}
