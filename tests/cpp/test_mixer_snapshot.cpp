#include "mixer/Snapshot.hpp"
#include <cassert>
#include <memory>

int main() {
    using Picture = std::shared_ptr<const int>;
    avp::mixer::Snapshot<Picture> state;
    bool failed = false;
    try { state.capture(0); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    auto blended = std::make_shared<const int>(37);
    state.presented(blended);
    auto first = state.capture(0);
    assert(state.holding() && state.replaces(0) && !state.replaces(1));
    assert(*state.frozen() == blended); // Keep the exact image, not an endpoint.
    assert(!state.canRelease(100, first)); // Routing has not been armed.
    state.arm(100);
    assert(!state.canRelease(99, first));
    assert(!state.canRelease(100, first - 1));
    assert(state.canRelease(100, first));
    state.release();
    auto next_blend = std::make_shared<const int>(52);
    state.presented(next_blend);
    auto second = state.capture(1);
    assert(second != first && *state.frozen() == next_blend);
    state.arm(200);
    assert(!state.canRelease(1000, first)); // Old frames cannot end a new hold.
    assert(state.canRelease(200, second));
    state.arm(210, false); // A ready hard cut replaces the captured picture.
    assert(!state.canRelease(209, 0));
    assert(state.canRelease(210, 0));
    state.finish();
    assert(!state.replaces(0) && !state.replaces(1));
    assert(state.holding()); // Keep output held until the ready frame arrives.
    state.release();
    assert(!state.holding());
}
