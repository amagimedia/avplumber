#pragma once
// A wipe clip that follows the lever: which of its frames a schedule's progress shows, where
// a change of the lever's schedule can start without rewriting a frame already pushed, and the
// background under it. Pure; tests/cpp/test_lever_wipe.cpp.
#include "TakeClock.hpp"
#include "TickGrid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace avp::mixer {

/// The clip's length: its last frame's offset from the first plus one frame of the clip's own
/// rate, from its last two frames (`fallback_period_ns`, one output tick, for a single frame).
inline int64_t clipLengthNs(const std::vector<int64_t>& offsets_ns, int64_t fallback_period_ns) {
    if (offsets_ns.empty()) return 0;
    const size_t n = offsets_ns.size();
    const int64_t period = n > 1 ? offsets_ns[n - 1] - offsets_ns[n - 2] : fallback_period_ns;
    return offsets_ns.back() + std::max<int64_t>(period, 1);
}

/// The frame at progress `p` of a clip of `length_ns` whose frames start at `offsets_ns`
/// (ascending, the first 0): the one a replay from its first frame shows on the output tick
/// nearest p's time, a replay putting each frame on its nearest tick (TickGrid::nearestIndex). The
/// lever run on at the clip's own rate so shows exactly the frames a replay would, at any clip rate.
inline size_t frameAt(const std::vector<int64_t>& offsets_ns, int64_t length_ns, double p, const TickGrid& grid) {
    const int64_t tick = grid.nearestIndex(std::llround(std::clamp(p, 0.0, 1.0) * double(length_ns)));
    const auto after = std::partition_point(offsets_ns.begin(), offsets_ns.end(),
                                            [&](int64_t offset) { return grid.nearestIndex(offset) <= tick; });
    return after == offsets_ns.begin() ? 0 : size_t(after - offsets_ns.begin() - 1);
}

/// A tick the clip player has surely not pushed yet at `now_ns`: it pushes tick T from
/// time(T) + `delay_ns` on, so the next is due within a tick, and this is the one after it.
inline int64_t playerNextNs(const TickGrid& grid, int64_t now_ns, int64_t delay_ns) {
    return grid.time(grid.atOrBefore(now_ns - delay_ns) + 2);
}

/// Where a change of a scrubbed clip's schedule starts: two ticks past the newest program frame
/// (`mix_change_ns`, where a mix's starts), and never before the player's next unpushed tick.
inline int64_t leverAnchorNs(int64_t mix_change_ns, int64_t player_next_ns) {
    return std::max(mix_change_ns, player_next_ns);
}

/// The furthest the lever's wipe clip goes while its overlay is not on air yet, the background still
/// Program: below the switch point by more than rounding a ramp's start to whole milliseconds
/// (rampTo) can carry it past, half a millisecond of a ramp at least a tick long (0.03 at 120 fps).
constexpr double kLeverWipeArmedAt = kWipeSwitchAt - 0.05;

/// The lever wipe's background at the program selector, from the heads of its program and preview
/// slots (pts in ns, 0: none) after the output's last frame at `last_ns`: Preview from where
/// `schedule` reaches the clip's switch point, Program before it, both ways, decided on the newest
/// head no older than `last_ns` (Preview only once `preview_ready`). `preview`: the background so
/// far. Its slot's head goes out if it is a tick not out yet that the schedule has on that side,
/// or Program's while Preview is not ready yet; else nothing does, and both heads are consumed.
struct WipeBackground {
    bool preview;   // the background now
    bool emit;      // its slot's head goes out
};
inline WipeBackground wipeBackground(const RampHistory& schedule, bool preview, int64_t program_ns,
                                     int64_t preview_ns, bool preview_ready, int64_t last_ns) {
    const auto current = [&](int64_t pts_ns) { return pts_ns >= last_ns ? pts_ns : int64_t(0); };
    const int64_t tick_ns = std::max(current(program_ns), current(preview_ns));
    if (tick_ns == 0) return {preview, false};
    const auto past = [&](int64_t pts_ns) { return rampAt(schedule, pts_ns) >= kWipeSwitchAt; };
    const bool wanted = past(tick_ns);
    if (wanted != preview && (!wanted || (preview_ready && preview_ns >= last_ns))) preview = wanted;
    const int64_t head_ns = preview ? preview_ns : program_ns;
    return {preview, head_ns > last_ns && (past(head_ns) == preview || (!preview && wanted))};
}

} // namespace avp::mixer
