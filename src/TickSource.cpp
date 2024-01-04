#include "TickSource.hpp"
#include "EventLoop.hpp"
#include "graph_core.hpp"
#include <mutex>
#include <algorithm>

void TickSource::tick() {
    std::lock_guard<decltype(busy_)> lock(busy_);
    for (auto &wptr: nodes_) {
        if (wptr.expired()) continue;
        global_event_loop.execute([wptr](EventLoop& evloop) {
            std::shared_ptr<NonBlockingNodeBase> node = wptr.lock();
            if (node!=nullptr) {
                node->processNonBlocking(evloop, true);
            }
        });
    }
}

void TickSource::add(std::weak_ptr<NonBlockingNodeBase> node) {
    std::lock_guard<decltype(busy_)> lock(busy_);
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(), [](std::weak_ptr<NonBlockingNodeBase> &p) { return p.expired(); }), nodes_.end());
    nodes_.push_back(node);
}
