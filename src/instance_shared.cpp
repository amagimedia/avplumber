#include "instance_shared.hpp"

std::unordered_map<const InstanceData*, std::list<std::function<void()>>> InstanceSharedObjectsDestructors::destructors_;
std::mutex InstanceSharedObjectsDestructors::busy_;
