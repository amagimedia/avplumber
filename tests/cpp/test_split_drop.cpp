// The split node's per-output `drop`, on real edges and the node the factory builds, against the
// whole library (tests/test_split_drop.py).
//
// An output nobody reads stands for a branch whose nodes were stopped: its edge fills and stays
// full. Named in `drop`, it loses its own frames, counted on its edge, and the other output gets
// every frame in order. Without `drop` the split waits for room in it, as it always did, and the
// other output gets nothing more until the full edge is read: what an output that must not lose
// frames relies on, and what a fan-out to optional outputs must not be built with.
#include "graph_mgmt.hpp"
#include <cassert>
#include <memory>

namespace {

constexpr size_t kCapacity = 3;       // of the edge nobody reads
constexpr int kFrames = 20;
constexpr unsigned kLongMs = 5000;    // a wait no test may sit out
constexpr unsigned kStalledMs = 300;  // long enough for the split to have passed a frame on

av::VideoFrame frameAt(int64_t tick) {
    av::VideoFrame frame(AV_PIX_FMT_YUV420P, 16, 16);
    frame.setTimeBase({1, 50});
    frame.setPts(av::Timestamp(tick, {1, 50}));
    return frame;
}

using VideoEdge = std::shared_ptr<Edge<av::VideoFrame>>;

/// A started split from `src` to `first` and `second`, in that order. The edges exist before the
/// node, which takes its data type from them; `unread` gets the small capacity.
void startSplit(const std::shared_ptr<NodeManager> &manager, const std::string &name, const std::string &src,
                const std::string &first, const std::string &second, const std::string &unread,
                const Parameters &drop) {
    manager->edges()->planCapacity(unread, kCapacity);
    for (const std::string &edge: {src, first, second}) manager->edges()->find<av::VideoFrame>(edge);
    Parameters params = {{"type", "split"}, {"name", name}, {"src", src},
                         {"dst", Parameters::array({first, second})}};
    if (!drop.is_null()) params["drop"] = drop;
    manager->createNode(params, true, true);
}

bool takes(const VideoEdge &edge, int64_t tick, unsigned timeout_ms = kLongMs) {
    av::VideoFrame frame;
    return edge->wait_dequeue_timed_ms(frame, timeout_ms) && frame.pts().timestamp() == tick;
}

}

int main() {
    const auto manager = std::make_shared<NodeManager>();
    auto edge = [&](const std::string &name) { return manager->edges()->find<av::VideoFrame>(name); };

    // The unread output is first in dst: the split has dealt with a frame for it by the time
    // the frame reaches the other output, so the counts below are settled when they are read.
    startSplit(manager, "fan", "program", "stopped", "running", "stopped", Parameters::array({"stopped"}));
    const VideoEdge program = edge("program"), stopped = edge("stopped"), running = edge("running");
    for (int tick = 0; tick < kFrames; tick++) {
        program->enqueue(frameAt(tick));
        assert(takes(running, tick));
    }
    assert(stopped->occupied() == static_cast<int>(kCapacity));
    assert(stopped->enqueuedTotal() == kCapacity);
    assert(stopped->droppedTotal() == kFrames - kCapacity);
    assert(running->droppedTotal() == 0);
    // The frames it kept are the oldest: what a restarted branch reads first.
    assert(takes(stopped, 0));

    // No drop: the split waits for room in the full output, and the other output with it.
    startSplit(manager, "waits", "source", "held", "through", "held", Parameters());
    const VideoEdge source = edge("source"), held = edge("held"), through = edge("through");
    for (int tick = 0; tick <= static_cast<int>(kCapacity); tick++) source->enqueue(frameAt(tick));
    for (int tick = 0; tick < static_cast<int>(kCapacity); tick++) assert(takes(through, tick));
    assert(!takes(through, kCapacity, kStalledMs));
    assert(held->droppedTotal() == 0);
    assert(takes(held, 0));                 // room in the full output
    assert(takes(through, kCapacity));      // and the frame that waited goes on

    manager->shutdown();
    return 0;
}
