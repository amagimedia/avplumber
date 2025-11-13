#pragma once
#include "../util.hpp"
#include "../avutils.hpp"
#include "../graph_core.hpp"
#include "../graph_base.hpp"
#include "../graph_interfaces.hpp"
#include "../node_create_function.hpp"
#include "../edge_types.hpp"

namespace {
    template<typename T> bool checkParamEdge(NodeCreationInfo &nci, const std::string &pname) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        if (params.count(pname) != 1) {
            return false;
        }
        
        Parameters param = params[pname];
        if (param.is_string()) {
            return edges.exists<T>(param.get<std::string>());
        } else if (param.is_array()) {
            for (auto &item: param) {
                if (edges.exists<T>(item.get<std::string>())) {
                    return true;
                }
            }
            return false;
        } else {
            return false;
        }
    }
    template<typename T> bool worksWithType(NodeCreationInfo &nci) {
        return checkParamEdge<T>(nci, "src") || checkParamEdge<T>(nci, "dst");
    }
};
// ATD = Automatic (data) Type Detection
template<template<typename> class Tpl> std::shared_ptr<Node> createNodeATD(NodeCreationInfo &nci) {
	#define X(T) if (worksWithType<T>(nci)) { \
		logstream << "Detected " << nci.params["type"].get<std::string>() << " as <" << #T << ">"; \
		return createNode<Tpl<T>>(nci); \
	}
	EDGE_DATA_TYPES(X)
	#undef X
	return nullptr;
}

template<template<typename> class Tpl> std::shared_ptr<Node> createNodeATD_RAW(NodeCreationInfo &nci) {
	#define X(T) if (worksWithType<T>(nci)) { \
		logstream << "Detected " << nci.params["type"].get<std::string>() << " as <" << #T << ">"; \
		return createNode<Tpl<T>>(nci); \
	}
	EDGE_RAW_DATA_TYPES(X)
	#undef X
	return nullptr;
}

template<template<typename> class Tpl, typename... Ts>
std::shared_ptr<Node> createNodeATD_Explicit(NodeCreationInfo &nci) {
	std::shared_ptr<Node> r = nullptr;
	(void)std::initializer_list<int>{
		(r ? 0 : (worksWithType<Ts>(nci) ? (r = createNode<Tpl<Ts>>(nci), 0) : 0))...
	};
	return r;
}
template<typename T> std::shared_ptr<Node> createNode(NodeCreationInfo &nci) {
    return std::static_pointer_cast<Node>(T::create(nci));
}



#define DECLNODE(nodetype, classname) \
    template std::shared_ptr<Node> createNode<classname>(NodeCreationInfo &nci);

//#define DECLNODE_ATD(nodetype, tplname) \
//    template std::shared_ptr<Node>  createNodeATD<tplname>(EdgeManager &edges, const Parameters &params);

#define DECLNODE_ATD(nodetype, tplname) \
	std::shared_ptr<Node> createNodeATD_ ## tplname(NodeCreationInfo &nci) { \
		return createNodeATD<tplname>(nci); \
	}

#define DECLNODE_ATD_RAW(nodetype, tplname) \
	std::shared_ptr<Node> createNodeATD_ ## tplname(NodeCreationInfo &nci) { \
		return createNodeATD_RAW<tplname>(nci); \
	}

#define DECLNODE_ATD_TYPES(nodetype, tplname, ...) \
	std::shared_ptr<Node> createNodeATD_ ## tplname(NodeCreationInfo &nci) { \
		return createNodeATD_Explicit<tplname, __VA_ARGS__>(nci); \
	}

#define DECLNODE_ALIAS(nodetype, classname)
#define DECLNODE_ATD_ALIAS(nodetype, tplname)
#define DECLNODE_ATD_RAW_ALIAS(nodetype, tplname)
#define DECLNODE_ATD_TYPES_ALIAS(nodetype, tplname, ...)
