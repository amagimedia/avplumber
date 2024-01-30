#pragma once
#include "util.hpp"
#include <mutex>
#include <memory>
#include <unordered_map>
#include <list>
#include <string>
#include <functional>

class InstanceData;

template<typename Child>
class InstanceShared/*: public std::enable_shared_from_this<InstanceShared<Child>>*/ {
};

class InstanceSharedObjectsDestructors {
private:
    friend class InstanceData;
    static std::unordered_map<const InstanceData*, std::list<std::function<void()>>> destructors_;
    static std::mutex busy_;
    static void callDestructors(const InstanceData* instance) {
        std::unique_lock<decltype(busy_)> lock(busy_);
        for (auto &fn: destructors_[instance]) {
            fn();
        }
        destructors_.erase(instance);
    }
public:
    static void addDestructor(const InstanceData* instance, std::function<void()> destructor) {
        std::unique_lock<decltype(busy_)> lock(busy_);
        destructors_[instance].push_back(destructor);
    }
};

struct SharedConstructorHelper {
    template <typename Object, typename std::enable_if<std::is_constructible_v<Object>>::type* = nullptr> static std::shared_ptr<Object> tryImplicitCreate(const std::string) {
        return std::make_shared<Object>();
    }
    template <typename Object, typename std::enable_if<!std::is_constructible_v<Object>>::type* = nullptr> static std::shared_ptr<Object> tryImplicitCreate(const std::string name) {
        throw Error("This shared object can't be created implicitly: " + name);
    }
};

template<typename Object>
class InstanceSharedObjects {
private:
    static std::unordered_map<const InstanceData*, std::unordered_map<std::string, std::shared_ptr<Object>>> objects_;
    static std::mutex busy_;

    static std::shared_ptr<Object>& find(const InstanceData &instance, const std::string id) {
        if (id.length()<1) {
            throw Error("too short shared object id: " + id);
        }
        bool is_global = id[0]=='@';
        const InstanceData* instance_ptr = is_global ? nullptr : &instance;
        std::string object_id = is_global ? id.substr(1) : id;
        if (is_global && object_id.length()<1) {
            throw Error("too short global object id: " + id);
        }
        
        // we need to track created objects in InstanceSharedObjectsDestructors,
        // so we track inner maps as their destruction will also destroy all objects inside.
        // let's add the destructor to InstanceSharedObjectsDestructors once per inner map.
        bool existed = objects_.count(instance_ptr);
        if (!existed) {
            InstanceSharedObjectsDestructors::addDestructor(instance_ptr, [instance_ptr]() {
                std::unique_lock<decltype(busy_)> lock2(busy_);
                objects_.erase(instance_ptr);
            });
        }
        
        return objects_[instance_ptr][object_id];
    }
public:
    static std::shared_ptr<Object> get(const InstanceData &instance, const std::string id) {
        std::unique_lock<decltype(busy_)> lock(busy_);
        std::shared_ptr<Object> &r = find(instance, id);
        if (!r) {
            r = SharedConstructorHelper::tryImplicitCreate<Object>(id);
        }
        return r;
    }
    enum class PolicyIfExists {
        Overwrite,
        Ignore,
        Throw
    };
    // FIXME DRY
    static void put(const InstanceData &instance, const std::string id, std::shared_ptr<Object> pobj, PolicyIfExists policy = PolicyIfExists::Overwrite) {
        std::unique_lock<decltype(busy_)> lock(busy_);
        std::shared_ptr<Object> &pref = find(instance, id);
        if (pref && (policy != PolicyIfExists::Overwrite)) {
            if (policy == PolicyIfExists::Throw) {
                throw Error("already have shared object " + id);
            }
        } else {
            pref = pobj;
        }
    }
    template<typename ... Args> static void emplace(const InstanceData &instance, const std::string id, PolicyIfExists policy, Args&&...args) {
        std::unique_lock<decltype(busy_)> lock(busy_);
        std::shared_ptr<Object> &pref = find(instance, id);
        if (pref && (policy != PolicyIfExists::Overwrite)) {
            if (policy == PolicyIfExists::Throw) {
                throw Error("already have shared object " + id);
            }
        } else {
            pref = std::make_shared<Object>(std::forward<Args>(args)...);
        }
    }
};

template<typename Object>
std::unordered_map<const InstanceData*, std::unordered_map<std::string, std::shared_ptr<Object>>> InstanceSharedObjects<Object>::objects_;

template<typename Object>
std::mutex InstanceSharedObjects<Object>::busy_;
