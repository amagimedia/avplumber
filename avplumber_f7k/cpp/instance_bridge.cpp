// The C++ nodes look devices up in InstanceSharedObjects<HWAccelDevice>, keyed
// by an InstanceData address. Rust's hwaccel service owns the real
// AVHWDeviceContext. This maps a Rust Instance onto one InstanceData and
// stores an extra reference to that device under the same name.

#include "instance.hpp"
#include "hwaccel.hpp"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

std::unordered_map<const InstanceData*, std::list<std::function<void()>>>
    InstanceSharedObjectsDestructors::destructors_;
std::unordered_map<const InstanceData*, std::list<std::function<void()>>>
    InstanceSharedObjectsDestructors::pre_shutdown_hooks_;
std::mutex InstanceSharedObjectsDestructors::busy_;

namespace {

std::mutex g_mu;
std::unordered_map<std::uintptr_t, std::unique_ptr<InstanceData>> g_by_inner;
std::unordered_map<std::uintptr_t, InstanceData*> g_by_wrapper;

} // namespace

extern "C" void avp_cpp_bind_instance(void* wrapper, void* inner) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto& slot = g_by_inner[reinterpret_cast<std::uintptr_t>(inner)];
    if (!slot) {
        slot = std::make_unique<InstanceData>();
    }
    g_by_wrapper[reinterpret_cast<std::uintptr_t>(wrapper)] = slot.get();
}

extern "C" InstanceData* avp_cpp_instance(void* wrapper) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto key = reinterpret_cast<std::uintptr_t>(wrapper);
    if (const auto it = g_by_wrapper.find(key); it != g_by_wrapper.end()) {
        return it->second;
    }
    auto& slot = g_by_inner[key];
    if (!slot) {
        slot = std::make_unique<InstanceData>();
    }
    g_by_wrapper[key] = slot.get();
    return slot.get();
}

extern "C" void avp_cpp_instance_drop(void* inner) {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto key = reinterpret_cast<std::uintptr_t>(inner);
    const auto it = g_by_inner.find(key);
    if (it == g_by_inner.end()) {
        return;
    }
    InstanceData* inst = it->second.get();
    for (auto alias = g_by_wrapper.begin(); alias != g_by_wrapper.end();) {
        if (alias->second == inst) {
            alias = g_by_wrapper.erase(alias);
        } else {
            ++alias;
        }
    }
    g_by_inner.erase(it);
}

extern "C" int avp_cpp_hwaccel_put(void* wrapper, const char* name, AVBufferRef* device) {
    if (!wrapper || !name || name[0] == '\0' || !device) {
        return -1;
    }
    InstanceData* inst = avp_cpp_instance(wrapper);
    AVBufferRef* owned = av_buffer_ref(device);
    if (!owned) {
        return -1;
    }
    try {
        auto hw = std::make_shared<HWAccelDevice>(owned);
        InstanceSharedObjects<HWAccelDevice>::put(
            *inst, std::string(name), std::move(hw),
            InstanceSharedObjects<HWAccelDevice>::PolicyIfExists::Ignore);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "hwaccel bridge: %s\n", ex.what());
        return -1;
    }
}
