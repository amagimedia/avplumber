#pragma once
// When a program change reaches a multiview: the aux tick on which the PVW tile should change,
// and when to set the compositor's composition so that tick is the first drawn with it.
//
// The PGM pad of a pgm_pvw_grid bus carries the program frames with the selector's pts, main
// tick K stamped main.time(K). The aux playout (Playout, presentation mode) draws that frame on
// aux tick aux.nearestIndex(pts) + pgm_delay_frames, at that tick's deadline aux.time(N) +
// latency. cuda_rect_overlay swaps a pending composition at the top of every iteration, before
// it prepares a tick, so a composition set inside (deadline(N-1), deadline(N)] is first drawn on
// tick N; applyAt(N) sits in the middle of that window.
//
// Two alignments of the PVW tile with a take whose first new program frame is K:
//  Program: the first aux tick whose deadline is at or after the program frame's own,
//    main.time(K) + main latency, the instant that frame leaves the main compositor. The PVW
//    tile then changes when the program does, or later by the part of an aux tick between
//    that instant and the next aux deadline: 0 when the frame sits on an aux tick (an even K
//    at 60 -> 30, every K at equal rates and latencies), half an aux tick (16.7 ms at 60 -> 30,
//    20 at 50 -> 25) for the odd ones. The PGM tile of the same multiview trails the program by
//    pgm_delay_frames aux ticks (plus the latency difference), so the PVW tile leads it.
//  PgmTile: the tick whose PGM tile shows frame K, so the PVW and PGM tiles change together,
//    one pgm_delay_frames later than the program (33 ms at 30 aux fps with the default 1).
//
// Error bound, in aux frames, for a cut or fade: 0 nominally. +1 when the follower is woken
// more than half an aux tick late (16 ms at 30 aux fps, 20 ms at 25) or the compositor's render
// thread is that late to tick N; -1 when the render thread draws tick N-1 more than half a tick
// late (the set lands in that iteration). +-1 main tick of the change's own pts when the first
// new program frame is not one main tick after the selector's last output (a missed main
// deadline before the cut, or a frame emitted between the selector switch and the read that
// follows it at once): half an aux tick at 50/60 fps, a whole one at 25/30. In Program mode a
// change lands on a tick whose deadline equals the program frame's: the mixer publishes it
// right after switching the selector, before the take's routing (MixerState::preview_mutex
// lets the follower wake meanwhile), at a random phase between the emission of frame K-1 and
// of frame K, so the follower has what is left of one main tick (0 to 16.7 ms at 60 fps)
// minus the main compositor's render time, its own wake and the composition set (never a read
// of the compositor's status, which waits on the mixer's mutex that the take still holds); a
// set past the deadline lands on the next tick (+1). A fade's change is
// published from a frame already presented, so such a tick has passed by construction
// (`target_unreachable` in the follower's status) and the change lands on the next. The
// timed composition only drops inputs (the previewed scene's are active; the program scene's
// stay warm for the swap), so the compositor applies it at once; when it does add one the bus
// was not receiving (a source that stalled, or takes faster than the warm-up settle, about one
// aux tick) the compositor stages it until that input has a frame for the tick, and past its
// staging deadline (max(250 ms, 2x latency)) keeps the previous layout: the change is dropped,
// not late, until the next preview change. A PGM frame that misses the aux deadline moves the
// PGM tile, not the PVW tile. Wipes and explicit previews are not timed: they draw on the next
// tick.
#include "TickGrid.hpp"
#include <cstdint>

namespace avp::mixer {

enum class PreviewAlign { Program, PgmTile };

struct PreviewFollowTiming {
    TickGrid main;
    TickGrid aux;
    int64_t aux_latency_ns;
    int64_t pgm_delay_ticks;
    int64_t main_latency_ns;
    PreviewAlign align;

    /// The pts of the program frame stamped `effective_ns`, requantized to the main grid: the pts
    /// a frame carries and TickGrid::time may differ by 1 ns (nearest vs floor rounding), and the
    /// aux grid's nearestIndex must see the same tick the compositor sees.
    int64_t programPts(int64_t effective_ns) const { return main.time(main.nearestIndex(effective_ns)); }
    /// When the program frame stamped `effective_ns` leaves the main compositor: its deadline.
    int64_t programDeparture(int64_t effective_ns) const { return programPts(effective_ns) + main_latency_ns; }
    /// Deadlines this close are the same instant: the grids are integer ns of rational ticks
    /// and the latencies come in ms, so a tick meant to leave with the program frame can fall
    /// a few ns short of it, and both compositors take longer than this to render.
    static constexpr int64_t kSameInstantNs = 1000000;
    /// The aux tick the PVW tile changes on for the first frame of a new program stamped
    /// `effective_ns` (see MixerState::PreviewChange), by the alignment above.
    int64_t targetTick(int64_t effective_ns) const {
        const int64_t pts = programPts(effective_ns);
        if (align == PreviewAlign::PgmTile) return aux.nearestIndex(pts) + pgm_delay_ticks;
        // The first tick N with deadline(N) >= the program frame's departure (less the
        // tolerance), i.e. the smallest N with aux.time(N) >= t for t = departure - aux
        // latency; atOrBefore(t - 1) + 1 is the smallest tick past t - 1.
        return aux.atOrBefore(pts + main_latency_ns - aux_latency_ns - kSameInstantNs - 1) + 1;
    }
    /// When the compositor draws `tick`.
    int64_t deadline(int64_t tick) const { return aux.time(tick) + aux_latency_ns; }
    /// When to set a composition so that `tick` is the first drawn with it: half a tick before
    /// its deadline, as far from tick-1's deadline as from its own.
    int64_t applyAt(int64_t tick) const { return deadline(tick) - aux.time(1) / 2; }
    /// The tick a composition set at `now_ns` is first drawn on: the first whose deadline is
    /// still ahead.
    int64_t tickDrawnAfter(int64_t now_ns) const { return aux.atOrBefore(now_ns - aux_latency_ns) + 1; }
};

} // namespace avp::mixer
