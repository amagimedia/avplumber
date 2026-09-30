// The multiview tick a program change lands on, and when to set the composition for it,
// against the compositor's own frame-to-tick mapping (Playout, presentation mode + PGM offset).
#include "mixer/primitives/PreviewFollow.hpp"
extern "C" {
#include <libavutil/mathematics.h>
}
#include <cassert>
#include <cstdlib>

int main() {
    using namespace avp::mixer;
    // 60 fps program into a 30 fps multiview, two aux ticks of latency, the PGM pad one tick back.
    {
        const TickGrid main(av::Rational(60, 1)), aux(av::Rational(30, 1));
        const PreviewFollowTiming t{main, aux, aux.time(2), 1};
        for (int64_t k = 0; k < 240; ++k) {
            // The pts a program frame carries, converted to ns the way av::Timestamp does (nearest).
            const int64_t pts = av_rescale_q(k, {1, 60}, {1, 1000000000});
            const int64_t compositor = aux.nearestIndex(pts) + 1;
            assert(t.targetTick(pts) == compositor);
            assert(t.targetTick(main.time(k)) == compositor);   // floor rounding sees the same tick
            // Even main ticks sit on an aux tick; odd ones on the half-tick boundary, kept upper.
            assert(compositor == k / 2 + k % 2 + 1);
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
    // 25 fps program and multiview, 80 ms latency, the PGM pad one tick back.
    {
        const TickGrid main(av::Rational(25, 1)), aux(av::Rational(25, 1));
        const PreviewFollowTiming t{main, aux, 80000000, 1};
        for (int64_t k = 0; k < 100; ++k) {
            assert(t.targetTick(av_rescale_q(k, {1, 25}, {1, 1000000000})) == k + 1);
            assert(t.targetTick(main.time(k)) == k + 1);
        }
        assert(t.deadline(10) == aux.time(10) + 80000000);
        assert(t.applyAt(10) == t.deadline(10) - 20000000);
        assert(t.tickDrawnAfter(t.applyAt(10)) == 10);
        // Without the PGM delay the change lands one tick earlier.
        const PreviewFollowTiming undelayed{main, aux, 80000000, 0};
        assert(undelayed.targetTick(main.time(7)) == 7);
    }
}
