#include "instance_shared.hpp"

std::unordered_map<const InstanceData*, std::list<std::function<void()>>> InstanceSharedObjectsDestructors::destructors_;
std::unordered_map<const InstanceData*, std::list<std::function<void()>>> InstanceSharedObjectsDestructors::pre_shutdown_hooks_;
std::mutex InstanceSharedObjectsDestructors::busy_;
