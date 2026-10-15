#include "rust_graph.hpp"
#include "avutils.hpp"

#include <iterator>

extern "C" InstanceData* avp_cpp_instance(void* rust_instance);

namespace {

void free_rust(char* text) {
    if (text) {
        avp_string_free(text);
    }
}

[[noreturn]] void fail(char* err, const std::string& fallback) {
    std::string msg = err && err[0] ? err : fallback;
    free_rust(err);
    throw Error(msg);
}

} // namespace

namespace avp::mixer {

InstanceData& RustMixerGraph::instanceData() const {
    return *avp_cpp_instance(core_);
}

bool RustMixerGraph::nodeExists(const std::string& name) const {
    return avp_node_exists(core_, name.c_str()) != 0;
}

std::string RustMixerGraph::nodeType(const std::string& name) const {
    if (!nodeExists(name)) {
        throw Error("Mixer node no longer exists: " + name);
    }
    return "";
}

bool RustMixerGraph::hasCreatedParam(const std::string& name, const std::string& key) const {
    if (!nodeExists(name)) {
        throw Error("Mixer node no longer exists: " + name);
    }
    auto it = params_.find(name);
    return it != params_.end() && it->second.contains(key);
}

void RustMixerGraph::call_set(const std::string& name, const std::string& key, const Parameters& value) {
    char* err = nullptr;
    if (avp_node_object_set(core_, name.c_str(), key.c_str(), value.dump().c_str(), &err) != 0) {
        fail(err, "mixer: set " + name + "." + key + " failed");
    }
}

Parameters RustMixerGraph::call_get(const std::string& name, const std::string& key) const {
    char* json = nullptr;
    char* err = nullptr;
    if (avp_node_object_get(core_, name.c_str(), key.c_str(), &json, &err) != 0) {
        fail(err, "mixer: get " + name + "." + key + " failed");
    }
    Parameters parsed = json ? Parameters::parse(json) : Parameters(nullptr);
    free_rust(json);
    return parsed;
}

void RustMixerGraph::setObject(const std::string& name, const std::string& key, const Parameters& value) {
    params_[name][key] = value;
    call_set(name, key, value);
}

bool RustMixerGraph::setObjectIfCreated(const std::string& name, const std::string& key,
                                        const Parameters& value) {
    if (!nodeExists(name)) {
        throw Error("Node " + name + " doesn't exist.");
    }
    params_[name][key] = value;
    try {
        call_set(name, key, value);
        return true;
    } catch (const std::exception& ex) {
        if (std::string(ex.what()) == "Node not created") {
            return false;
        }
        throw;
    }
}

Parameters RustMixerGraph::getObject(const std::string& name, const std::string& key) const {
    return call_get(name, key);
}

bool RustMixerGraph::tryGetObject(const std::string& name, const std::string& key, Parameters& out) const {
    if (!nodeExists(name)) {
        return false;
    }
    try {
        out = call_get(name, key);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void RustMixerGraph::setParam(const std::string& name, const std::string& key, const std::string& value) {
    if (!nodeExists(name)) {
        throw Error("Mixer node no longer exists: " + name);
    }
    params_[name][key] = value;
}

void RustMixerGraph::startGroup(const std::string& name) {
    char* err = nullptr;
    if (avp_group_start_named(core_, name.c_str(), &err) != 0) {
        fail(err, "Mixer group no longer exists: " + name);
    }
}

void RustMixerGraph::stopGroup(const std::string& name) {
    char* err = nullptr;
    if (avp_group_stop_named(core_, name.c_str(), &err) != 0) {
        fail(err, "Mixer group no longer exists: " + name);
    }
}

void RustMixerGraph::stopGroupAndWait(const std::string& name) {
    stopGroup(name);
}

void RustMixerGraph::stopNode(const std::string&, bool) {
    throw Error("mixer: stopping one node is not available");
}

void RustMixerGraph::resetInput(const std::string&) {
    throw Error("mixer: resetInput is not available");
}

void RustMixerGraph::clearEdgeIfIdle(const std::string&) {
    throw Error("mixer: flushing edges is not available");
}

av::Timestamp RustMixerGraph::edgeLastTimestamp(const std::string&) const {
    return NOTS;
}

std::string RustMixerGraph::firstDstEdgeName(const std::string&) const {
    return "";
}

std::string RustMixerGraph::edgeNameAt(const std::string& node, const std::string& param, size_t index) const {
    auto it = params_.find(node);
    if (it == params_.end() || !it->second.contains(param)) {
        return "";
    }
    auto names = jsonToStringList(it->second[param]);
    if (index >= names.size()) {
        return "";
    }
    auto cursor = names.begin();
    std::advance(cursor, static_cast<std::ptrdiff_t>(index));
    return *cursor;
}

int RustMixerGraph::routerOutputCount(const std::string& router) const {
    throw Error("mixer.init_routes: node " + router + " doesn't exist");
}

int RustMixerGraph::routerOutputIndexFromLabel(const std::string& router, const std::string&) const {
    throw Error("mixer.routed_source: node " + router + " doesn't exist");
}

std::shared_ptr<CutLatencyObserver> RustMixerGraph::cutLatencyObserver(const std::string& name) const {
    if (!nodeExists(name)) {
        throw Error("Mixer node no longer exists: " + name);
    }
    return nullptr;
}

} // namespace avp::mixer

// avutils.cpp is not linked into this crate.
Wallclock wallclock;
