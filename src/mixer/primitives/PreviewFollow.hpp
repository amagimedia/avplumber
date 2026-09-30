#pragma once
// When a program change reaches a multiview: the aux tick whose PGM tile first shows it,
// and when to set the compositor's composition so the PVW tile changes on that same tick.
//
// The PGM pad of a pgm_pvw_grid bus carries the program frames with the selector's pts, main
// tick K stamped main.time(K). The aux playout (Playout, presentation mode) draws that frame on
// aux tick aux.nearestIndex(pts) + pgm_delay_frames, at that tick's deadline aux.time(N) +
// latency. cuda_rect_overlay swaps a pending composition at the top of every iteration, before
// it prepares a tick, so a composition set inside (deadline(N-1), deadline(N)] is first drawn on
// tick N; applyAt(N) sits in the middle of that window.
//
// Error bound, in aux frames, for a cut or fade: 0 nominally. +1 when the follower is woken
// more than half an aux tick late (16 ms at 30 aux fps, 20 ms at 25) or the compositor's render
// thread is that late to tick N; -1 when the render thread draws tick N-1 more than half a tick
// late (the set lands in that iteration). +-1 main tick of the change's own pts when the first
// new program frame is not one main tick after the selector's last output (a missed main
// deadline before the cut, or a frame emitted between the selector switch and the read that
// follows it at once): half an aux tick at 50/60 fps, a whole one at 25/30. The timed
// composition only drops inputs (the previewed scene's are active; with swap_preview the program
// scene's stay warm), so the compositor applies it at once; when it does add one the bus was not
// receiving (a source that stalled, or takes faster than the warm-up settle, about one aux tick)
// the compositor stages it until that input has a frame for the tick, and past its staging
// deadline (max(250 ms, 2x latency)) keeps the previous layout: the change is dropped, not late,
// until the next preview change. A PGM frame that misses the aux deadline moves the PGM tile,
// not the PVW tile. Wipes and explicit previews are not timed: they draw on the next tick.
#include "TickGrid.hpp"
#include <cstdint>

namespace avp::mixer {

struct PreviewFollowTiming {
    TickGrid main;
    TickGrid aux;
    int64_t aux_latency_ns;
    int64_t pgm_delay_ticks;

    /// The aux tick whose PGM tile first shows the program frame stamped at `effective_ns`, the
    /// pts of the first frame of the new program (see MixerState::pvw_effective_ns). Requantized
    /// to the main grid first: the pts a frame carries and TickGrid::time may differ by 1 ns
    /// (nearest vs floor rounding), and the aux grid's nearestIndex must see the same tick the
    /// compositor sees.
    int64_t targetTick(int64_t effective_ns) const {
        return aux.nearestIndex(main.time(main.nearestIndex(effective_ns))) + pgm_delay_ticks;
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
