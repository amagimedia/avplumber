// The multiview tick a program change lands on, and when to set the composition for it,
// against the compositor's own frame-to-tick mapping (Playout, presentation mode + PGM offset)
// and the program frame's departure from the main compositor.
#include "mixer/primitives/PreviewFollow.hpp"
extern "C" {
#include <libavutil/mathematics.h>
}
#include <cassert>
#include <cstdlib>

int main() {
    using namespace avp::mixer;
    // 60 fps program into a 30 fps multiview, two aux ticks of latency, the PGM pad one tick back,
    // the PVW tile on the tick whose PGM tile shows the take.
    {
        const TickGrid main(av::Rational(60, 1)), aux(av::Rational(30, 1));
        const PreviewFollowTiming t{main, aux, aux.time(2), 1, main.time(3), PreviewAlign::PgmTile};
        for (int64_t k = 0; k < 240; ++k) {
            // The pts a program frame carries, converted to ns the way av::Timestamp does (nearest).
            const int64_t pts = av_rescale_q(k, {1, 60}, {1, 1000000000});
            const int64_t compositor = aux.nearestIndex(pts) + 1;
            assert(t.targetTick(pts) == compositor);
            assert(t.targetTick(main.time(k)) == compositor);   // floor rounding sees the same tick
            // Even main ticks sit on an aux tick; odd ones on the half-tick boundary, kept upper.
            assert(compositor == k / 2 + k % 2 + 1);
            // The program frame leaves the main compositor three main ticks after its pts.
            assert(t.programDeparture(pts) == main.time(k) + main.time(3));
        }
        for (int64_t n = 1; n < 100; ++n) {
            // Inside the window that draws first on n, half a tick from either end.
            assert(t.deadline(n - 1) < t.applyAt(n) && t.applyAt(n) < t.deadline(n));
            assert(std::llabs((t.applyAt(n) - t.deadline(n - 1)) - (t.deadline(n) - t.applyAt(n))) <= 2);
            assert(t.tickDrawnAfter(t.applyAt(n)) == n);
            assert(t.tickDrawnAfter(t.deadline(n - 1) + 1) == n);
            assert(t.tickDrawnAfter(t.deadline(n) + 1) == n + 1);
        }
    }
    // The same bus aligned with the program output, at the program's latency (50 ms, three main
    // ticks): the first aux tick whose deadline is at or after the program frame's.
    {
        const TickGrid main(av::Rational(60, 1)), aux(av::Rational(30, 1));
        const PreviewFollowTiming t{main, aux, main.time(3), 1, main.time(3), PreviewAlign::Program};
        for (int64_t k = 0; k < 240; ++k) {
            const int64_t pts = av_rescale_q(k, {1, 60}, {1, 1000000000});
            const int64_t n = t.targetTick(pts);
            assert(n == t.targetTick(main.time(k)));
            assert(t.deadline(n) >= t.programDeparture(pts) - PreviewFollowTiming::kSameInstantNs &&
                   t.deadline(n - 1) < t.programDeparture(pts));
            // An even main tick sits on an aux tick: the same instant. An odd one is half an aux
            // tick before the next: the PVW tile changes 16.7 ms after the program.
            assert(n == (k + 1) / 2);
            const int64_t lag = t.deadline(n) - t.programDeparture(pts);
            assert(k % 2 == 0 ? lag == 0 : std::llabs(lag - main.time(1)) <= 1);
            // The PGM tile of that multiview shows the frame one aux tick later.
            assert(aux.nearestIndex(main.time(k)) + 1 == n + 1);
        }
    }
    // A bus with more latency than the program (the old two-aux-tick default, 66.7 ms against
    // 50): aligned with the program, its tick is the one leaving at or after the program frame.
    {
        const TickGrid main(av::Rational(60, 1)), aux(av::Rational(30, 1));
        const PreviewFollowTiming t{main, aux, aux.time(2), 1, main.time(3), PreviewAlign::Program};
        for (int64_t k = 2; k < 240; ++k) {
            const int64_t pts = main.time(k), n = t.targetTick(pts);
            const int64_t lag = t.deadline(n) - t.programDeparture(pts);
            assert(lag >= -PreviewFollowTiming::kSameInstantNs && t.deadline(n - 1) < t.programDeparture(pts));
            // Even: the aux tick at the pts leaves 16.7 ms after the program; odd: with it, give
            // or take the ns the integer grids round by.
            assert(k % 2 == 0 ? std::llabs(lag - main.time(1)) <= 2 : std::llabs(lag) <= 2);
        }
    }
    // Equal rates and latencies (30 fps program and multiview at 66.7 ms): every take lands on the
    // aux tick leaving with the program frame, one tick before the PGM tile shows it.
    {
        const TickGrid main(av::Rational(30, 1)), aux(av::Rational(30, 1));
        const PreviewFollowTiming t{main, aux, main.time(2), 1, main.time(2), PreviewAlign::Program};
        for (int64_t k = 0; k < 100; ++k) {
            assert(t.targetTick(av_rescale_q(k, {1, 30}, {1, 1000000000})) == k);
            assert(t.deadline(k) == t.programDeparture(main.time(k)));
        }
        const PreviewFollowTiming tile{main, aux, main.time(2), 1, main.time(2), PreviewAlign::PgmTile};
        assert(tile.targetTick(main.time(7)) == 8);
    }
    // A 60 fps bus at the program's rate (full_rate) and latency: the same tick as the program frame.
    {
        const TickGrid main(av::Rational(60, 1));
        const PreviewFollowTiming t{main, main, main.time(3), 1, main.time(3), PreviewAlign::Program};
        for (int64_t k = 0; k < 240; ++k) {
            const int64_t pts = av_rescale_q(k, {1, 60}, {1, 1000000000});
            assert(t.targetTick(pts) == k && t.deadline(k) == t.programDeparture(pts));
        }
    }
    // 25 fps program and multiview, 80 ms latency, the PGM pad one tick back.
    {
        const TickGrid main(av::Rational(25, 1)), aux(av::Rational(25, 1));
        const PreviewFollowTiming t{main, aux, 80000000, 1, 80000000, PreviewAlign::PgmTile};
        for (int64_t k = 0; k < 100; ++k) {
            assert(t.targetTick(av_rescale_q(k, {1, 25}, {1, 1000000000})) == k + 1);
            assert(t.targetTick(main.time(k)) == k + 1);
        }
        assert(t.deadline(10) == aux.time(10) + 80000000);
        assert(t.applyAt(10) == t.deadline(10) - 20000000);
        assert(t.tickDrawnAfter(t.applyAt(10)) == 10);
        // Without the PGM delay the change lands one tick earlier.
        const PreviewFollowTiming undelayed{main, aux, 80000000, 0, 80000000, PreviewAlign::PgmTile};
        assert(undelayed.targetTick(main.time(7)) == 7);
        // Aligned with the program, the PGM delay does not move the PVW tile.
        const PreviewFollowTiming program{main, aux, 80000000, 1, 80000000, PreviewAlign::Program};
        assert(program.targetTick(main.time(7)) == 7);
    }
}
