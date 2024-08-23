#pragma once
#include "instance_shared.hpp"
#include <memory>

class NonBlockingNodeBase;
class EventLoop;

class TickSource: public InstanceShared<TickSource>, public std::enable_shared_from_this<TickSource> {
protected:
    std::vector<std::weak_ptr<NonBlockingNodeBase>> nodes_;
    std::mutex busy_;
    std::shared_ptr<EventLoop> event_loop_;
public:
    void tick(EventLoop &evl);
    void fastTick();
    void add(std::shared_ptr<NonBlockingNodeBase> node);
    TickSource(std::shared_ptr<EventLoop> evl): event_loop_(evl) {
    }
};
