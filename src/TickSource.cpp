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
    int exc_ms = event_loop_->fastExecute(av::Timestamp(fast_tick_ms_, av::Rational(1, 1000)), [sthis](EventLoop& evl) { sthis->tick(evl); });
    if (exc_ms > 2) {
        logstream << "fastTick deadline exceeded by " << exc_ms << "ms";
    }
}

void TickSource::add(std::shared_ptr<NonBlockingNodeBase> node) {
    node->setEventLoop(event_loop_, true);
    std::lock_guard<decltype(busy_)> lock(busy_);
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(), [](std::weak_ptr<NonBlockingNodeBase> &p) { return p.expired(); }), nodes_.end());
    nodes_.push_back(std::weak_ptr<NonBlockingNodeBase>(node));
}

TickSource::TickSource(std::shared_ptr<EventLoop> evl) : event_loop_(evl) {
	const char *envstr = getenv("AVPLUMBER_FAST_TICK_MS");
	if (envstr && envstr[0]) {
		int i = atoi(envstr);
		if (i > 0) {
			fast_tick_ms_ = i;
		} else {
			logstream << "invalid AVPLUMBER_FAST_TICK_MS, must be integer >0";
		}
	}
}
