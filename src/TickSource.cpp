#include "TickSource.hpp"
#include "EventLoop.hpp"
#include "graph_core.hpp"
#include <mutex>
#include <algorithm>

void TickSource::tick(EventLoop &evl) {
    std::lock_guard<decltype(busy_)> lock(busy_);
    for (auto &wptr: nodes_) {
        if (wptr.expired()) continue;
        evl.execute([wptr](EventLoop& evloop) {
            std::shared_ptr<NonBlockingNodeBase> node = wptr.lock();
            if (node!=nullptr) {
                node->wrappedProcessNonBlocking(evloop, true);
            }
        });
    }
}

void TickSource::fastTick() {
    std::shared_ptr<TickSource> sthis = this->shared_from_this();
    event_loop_->fastExecute(av::Timestamp(200, av::Rational(1, 1000)), [sthis](EventLoop& evl) { sthis->tick(evl); });
}

void TickSource::add(std::shared_ptr<NonBlockingNodeBase> node) {
    node->setEventLoop(event_loop_, true);
    std::lock_guard<decltype(busy_)> lock(busy_);
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(), [](std::weak_ptr<NonBlockingNodeBase> &p) { return p.expired(); }), nodes_.end());
    nodes_.push_back(std::weak_ptr<NonBlockingNodeBase>(node));
}
