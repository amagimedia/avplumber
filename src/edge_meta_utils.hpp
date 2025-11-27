#pragma once
#include <avcpp/packet.h>
#include <avcpp/frame.h>
#include "edge_types.hpp"

namespace edge_meta_utils {

template<template<typename> class SingleContainer> class MultiContainer:
#define MC_BASE(T) protected SingleContainer<T>
    EDGE_DATA_TYPES_AS_BASES(MC_BASE)
#undef MC_BASE
{
public:
    template<typename Cb> bool some(Cb cb) {
        bool matched = false;
        #define X(T) if (cb(static_cast<SingleContainer<T>*>(this))) { matched = true; } else
        EDGE_DATA_TYPES(X)
        /*fallthrough*/ matched = matched;
        #undef X
        return matched;
    }
    template<typename Cb> void forEach(Cb cb) {
        some([cb](auto arg) -> bool {
            cb(arg);
            return false;
        });
    }
    template<typename T> SingleContainer<T>* get() {
        return static_cast<SingleContainer<T>*>(this);
    }
};

};