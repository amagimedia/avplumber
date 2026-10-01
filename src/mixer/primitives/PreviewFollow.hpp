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
// How far a change can land from its target tick is in doc/mixer.md.
#include "TickGrid.hpp"
#include <cstdint>

namespace avp::mixer {

enum class PreviewAlign : std::uint8_t { Program, PgmTile };

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
