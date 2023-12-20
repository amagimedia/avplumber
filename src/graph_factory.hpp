#pragma once
#include <functional>
#include <memory>
#include <unordered_map>
#include "graph_core.hpp"

using NodeFactoryFunction = std::function<std::shared_ptr<Node>(NodeCreationInfo&)>;

void initFactories(std::unordered_map<std::string, NodeFactoryFunction> &factories);
