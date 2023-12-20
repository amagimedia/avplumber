#pragma once
#include "graph_mgmt.hpp"
#include "util.hpp"
#include <memory>


template<template<typename> class Tpl> std::shared_ptr<Node> createNodeATD(NodeCreationInfo&);
template<typename T> std::shared_ptr<Node> createNode(NodeCreationInfo&);
