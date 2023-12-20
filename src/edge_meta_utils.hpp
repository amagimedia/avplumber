#pragma once
#include <avcpp/packet.h>
#include <avcpp/frame.h>

namespace edge_meta_utils {

template<template<typename> class SingleContainer> class MultiContainer:
    protected SingleContainer<av::Packet>,
    protected SingleContainer<av::VideoFrame>,
    protected SingleContainer<av::AudioSamples>
{
public:
    template<typename Cb> bool some(Cb cb) {
        if (cb(static_cast<SingleContainer<av::Packet>*>(this))) return true;
        if (cb(static_cast<SingleContainer<av::VideoFrame>*>(this))) return true;
        if (cb(static_cast<SingleContainer<av::AudioSamples>*>(this))) return true;
        return false;
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