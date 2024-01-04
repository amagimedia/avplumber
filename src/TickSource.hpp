#pragma once
#include "instance_shared.hpp"

class NonBlockingNodeBase;

class TickSource: public InstanceShared<TickSource> {
protected:
    std::vector<std::weak_ptr<NonBlockingNodeBase>> nodes_;
    std::mutex busy_;
public:
    void tick();
    void add(std::weak_ptr<NonBlockingNodeBase> node);
};
