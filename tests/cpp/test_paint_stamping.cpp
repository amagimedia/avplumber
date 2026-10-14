// smooth_timestamps numbers a browser source's frames on the compositor's tick grid by counting and
// corrects the count against the playout deadline: a paint that would reach its slot's tick with
// less than a guard in hand moves the numbering one slot later, once; a numbering that runs more
// than one and a half slots ahead of the clock for a sustained stretch moves back one slot. Driven
// with a fake clock through PaintStamping, the policy the node delegates to, on recorded patterns:
// the probe's window 2 (a paint 27.5 ms past its deadline), the show's 28 fps delivery, Electron's
// 33333 us grid drifting against the 33333.33 us slot, a stall with a burst, and the hold's copies
// ahead of the stamping with a page that slides 2 s in every 6 s.
#include "mixer/primitives/HoldNumbering.hpp"
#include "mixer/primitives/PaintStamping.hpp"
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// A failed check names its case, the expression and the step the harness was at.
std::string g_case, g_step;

template<class... A> void step(const A &...a) {
    std::ostringstream o;
    (o << ... << a);
    g_step = o.str();
}

} // namespace

#define CASE() (g_case = __func__, g_step.clear())
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(g_case + ": " #expr " (line " + \
    std::to_string(__LINE__) + ") at " + g_step); } while (false)

using avp::mixer::HoldNumbering;
using avp::mixer::PaintStamping;

namespace {

const AVRational kFps{30, 1};
const AVRational kUsTb{1, 1000000};        // the DMA-BUF chain's stamps: CLOCK_MONOTONIC microseconds
const int64_t kMs = 1000000;
const int64_t kUs = 1000;
const int64_t P = 33333333;                // one slot at 30 fps: av_rescale_q(1, 1/fps, 1 ns)
const int64_t L = 2 * P;                   // the playout latency at 30 fps (default_latency_ms)
const int64_t GUARD = 10 * kMs;            // the margin a paint must keep to its slot's deadline
const int64_t EARLY = P + P / 2;           // the lead on its slot time a numbering may run ahead
const int EARLY_FRAMES = 30;               // ... for this many frames before it moves back
const int64_t BROWSER = 33333 * kUs;       // Electron's frame interval: 1000000 / 30 whole microseconds
const int64_t D = 4 * kMs;                 // paint stamp (the browser's main process) -> the stamping node
// A phase on the grid: 1000 ms is a whole number of slots, so the first paint sits 25 ms into a slot
// and its round_up slot begins 8.33 ms after it.
const int64_t T0 = 1000 * kMs + 25 * kMs;

int64_t jitter(int64_t n) { return ((n * 7919) % 1500) * kUs; }   // delivery jitter in [0, 1.5 ms)
int64_t slop(int64_t n) { return ((n * 104729) % 1000) * kUs; }   // a node's wake-up slop in [0, 1 ms)

PaintStamping::Config config() {
    PaintStamping::Config c;
    c.fps = kFps;
    c.latency_ns = L;
    c.guard_ns = GUARD;
    c.early_ns = EARLY;
    c.early_frames = EARLY_FRAMES;
    c.discontinuity_ns = 100 * kMs;
    return c;
}

/// What smooth_timestamps does around the policy: every frame, paint or copy, gets the slot the
/// policy gives it for its own stamp and the instant it is stamped.
struct Stream {
    PaintStamping stamping{config()};
    std::vector<int64_t> slots, nows;
    std::vector<bool> copies;

    PaintStamping::Result frame(int64_t now, int64_t stamp_ns, bool copy = false) {
        step(copy ? "copy #" : "paint #", slots.size(), " stamp=", stamp_ns, " now=", now,
             " (last slot ", slots.empty() ? -1 : slots.back(), ")");
        const auto r = stamping.stamp(true, stamp_ns, now);
        slots.push_back(r.slot);
        nows.push_back(now);
        copies.push_back(copy);
        step(g_step, " -> slot ", r.slot, r.shifted ? " shifted " : "", r.shifted ? std::to_string(r.shifted) : "",
             r.realigned ? " realigned" : "");
        return r;
    }
    /// The slot the next frame gets if nothing is late or early.
    int64_t next() const { return stamping.nextSlot(); }
    int64_t margin(size_t i) const { return stamping.marginNs(slots[i], nows[i]); }   // to the slot's deadline
    int64_t lead(size_t i) const { return stamping.timeOf(slots[i]) - nows[i]; }      // on the slot's time
    const PaintStamping::Stats &stats() const { return stamping.stats(); }

    int64_t skipped() const {   // slots between consecutive frames that no frame took
        int64_t n = 0;
        for (size_t i = 1; i < slots.size(); ++i)
            if (slots[i] > slots[i - 1] + 1) n += slots[i] - slots[i - 1] - 1;
        return n;
    }
    int gaps() const {
        int n = 0;
        for (size_t i = 1; i < slots.size(); ++i) n += slots[i] > slots[i - 1] + 1;
        return n;
    }
    int dups() const {          // a frame numbered at or behind the one before it
        int n = 0;
        for (size_t i = 1; i < slots.size(); ++i) n += slots[i] <= slots[i - 1];
        return n;
    }
    void consecutive(size_t from, size_t to) const {
        for (size_t i = from + 1; i <= to; ++i) {
            step("frame ", i, copies[i] ? " (copy)" : "", " slot ", slots[i], " after ", slots[i - 1]);
            CHECK(slots[i] == slots[i - 1] + 1);
        }
    }
    /// No frame leaves with less than `floor` to its slot's deadline.
    void never_late(int64_t floor = 0) const {
        for (size_t i = 0; i < slots.size(); ++i) {
            step("frame ", i, copies[i] ? " (copy)" : "", " slot ", slots[i], " margin ", margin(i) / 1000, " us");
            CHECK(margin(i) >= floor);
        }
    }
    /// Every slot skipped was a shift later or a realignment, every dup a shift back: nothing else
    /// moves the numbering.
    void steps_match_the_counters(int64_t realigned_skips = 0) const {
        step("skipped ", skipped(), " shifts_later ", stats().shifts_later, " realigned skips ", realigned_skips,
             " dups ", dups(), " shifts_back ", stats().shifts_back);
        CHECK(skipped() == int64_t(stats().shifts_later) + realigned_skips);
        CHECK(dups() == int(stats().shifts_back));
    }
};

/// Paints on Electron's grid from `t0`, delivered D plus jitter later.
void paints(Stream &s, int64_t t0, int64_t from, int64_t to, int64_t interval = BROWSER) {
    for (int64_t i = from; i < to; ++i) {
        const int64_t stamp = t0 + i * interval;
        s.frame(stamp + D + jitter(i), stamp);
    }
}

void steady_paints_are_numbered_consecutively_and_keep_the_latency_in_hand() {
    CASE();
    Stream s;
    paints(s, T0, 0, 3000);   // 100 s
    s.consecutive(0, 2999);
    CHECK(s.stats().shifts_later == 0 && s.stats().shifts_back == 0 && s.stats().realignments == 0);
    // round_up: the first paint precedes the slot it is numbered for, by less than a slot
    CHECK(s.stamping.timeOf(s.slots[0]) >= T0 && s.stamping.timeOf(s.slots[0]) < T0 + P);
    // a numbered paint reaches the stamping with the whole latency less its delivery in hand
    s.never_late(L - D - 2 * kMs);
}

void the_probes_27_ms_late_paint_moves_the_numbering_one_slot_later_once() {
    CASE();
    Stream s;
    paints(s, T0, 0, 200);
    // The probe's window 2: a paint reached the mixer 27.51 ms after its slot's deadline
    // (margin_min=-27.51 ms); the next ones came on time again.
    const int64_t slot = s.next();
    const int64_t deadline = s.stamping.timeOf(slot) + L;
    const auto late = s.frame(deadline + 27510 * kUs, T0 + 200 * BROWSER);
    CHECK(late.shifted == 1 && late.slot == slot + 1);
    CHECK(s.margin(200) >= 0);                       // drawn at its tick, not repeated and then superseded
    CHECK(s.stats().shifts_later == 1);
    paints(s, T0, 201, 600);
    s.consecutive(201, 599);                         // once: the shift holds, nothing oscillates
    CHECK(s.stats().shifts_later == 1 && s.stats().shifts_back == 0);
    s.steps_match_the_counters();
    s.never_late();
    // The cost for this window: one slot of latency. Its paints now precede their slot by one more
    // period than before, and stay there (the lead is under the early threshold).
    CHECK(s.lead(599) > s.lead(199) + P - 2 * kMs && s.lead(599) < s.lead(199) + P + 2 * kMs);
    CHECK(s.lead(599) < EARLY);
}

void paints_hovering_at_the_guard_shift_once_and_stay() {
    CASE();
    Stream s;
    s.frame(T0 + D, T0);
    const int64_t g = s.stamping.timeOf(s.slots[0]) - T0;   // the first paint's distance to its slot
    // Delivery such that the margin to the deadline alternates 5 ms under and 5 ms over the guard.
    for (int64_t i = 1; i <= 300; ++i) {
        const int64_t want = GUARD + ((i % 2) ? 5 : -5) * kMs;
        const int64_t counted_slot_time = T0 + g + i * P;   // where plain counting would put it
        s.frame(counted_slot_time + L - want, T0 + i * BROWSER);
    }
    CHECK(s.stats().shifts_later == 1 && s.stats().shifts_back == 0);
    s.steps_match_the_counters();
    s.never_late();
    for (size_t i = 3; i < s.slots.size(); ++i) {
        step("paint ", i, " margin ", s.margin(i) / 1000, " us");
        CHECK(s.margin(i) >= GUARD + P - 5 * kMs);   // a slot over what it would have had
    }
}

void paints_at_28_fps_cost_a_skipped_slot_twice_a_second_and_no_discard() {
    CASE();
    Stream s;
    // The show's windows deliver about 28 fps: one paint every 35.714 ms into 33.333 ms slots.
    paints(s, T0, 0, 1680, 35714 * kUs);   // 60 s
    CHECK(s.stats().shifts_back == 0 && s.stats().realignments == 0 && s.dups() == 0);
    s.steps_match_the_counters();
    // 1680 paints span about 1800 slots: the 120 that get no paint are skipped one at a time
    step("shifts later ", s.stats().shifts_later);
    CHECK(s.stats().shifts_later >= 110 && s.stats().shifts_later <= 125);
    for (size_t i = 1; i < s.slots.size(); ++i) {
        step("paint ", i, " slot ", s.slots[i], " after ", s.slots[i - 1]);
        CHECK(s.slots[i] - s.slots[i - 1] <= 2);
    }
    s.never_late();
}

void the_browser_grid_drifting_the_paints_earlier_costs_one_discard_in_an_hour_at_most() {
    CASE();
    Stream s;
    // 33333 us against 33333.33 us: the paints gain 10 us a second on the grid, 54 ms in 90 minutes.
    const int64_t n = 90 * 60 * 30;
    paints(s, T0 - 5 * kMs, 0, n);   // first paint 13.3 ms before its slot: lead 9 ms, 1.5 slots is 50 ms
    CHECK(s.stats().shifts_later == 0 && s.stats().realignments == 0);
    CHECK(s.stats().shifts_back == 1);   // once past 1.5 slots early for 30 frames: one slot back
    CHECK(s.dups() == 1 && s.gaps() == 0);
    s.steps_match_the_counters();
    s.never_late(L - D - 2 * kMs);
    CHECK(s.lead(n - 1) < EARLY);
    // the shift back happened after the lead had grown past the threshold, not before
    size_t at = 1;
    while (s.slots[at] > s.slots[at - 1]) ++at;
    step("shift back at paint ", at, " lead before ", s.lead(at - 1) / 1000, " us");
    CHECK(s.lead(at - 1) > EARLY && at > 60 * 60 * 30);
}

void a_stall_followed_by_a_burst_plays_the_backlog_out_and_settles_within_a_few_seconds() {
    CASE();
    Stream s;
    paints(s, T0, 0, 300);
    // 150 ms without delivery, then the five paints made meanwhile arrive together.
    const int64_t release = T0 + 300 * BROWSER + D + 150 * kMs;
    for (int64_t i = 300; i < 305; ++i) s.frame(release, T0 + i * BROWSER);
    CHECK(s.margin(300) >= 0);                        // the first is numbered for a tick still to come
    const uint64_t shifted = s.stats().shifts_later;
    CHECK(shifted >= 2 && shifted <= 4);              // the slots the stall left without a paint
    s.consecutive(300, 304);                          // the backlog keeps its order and spacing
    paints(s, T0, 305, 605);                          // 10 s of normal delivery
    CHECK(s.stats().shifts_later == shifted);         // nothing late after the burst
    CHECK(s.stats().shifts_back >= 1 && s.stats().shifts_back <= 3);   // the lead the burst left is worked off
    s.steps_match_the_counters();
    s.consecutive(455, 604);                          // settled within 5 s
    CHECK(s.lead(604) < EARLY);
    s.never_late();
}

void hold_copies_and_paints_share_one_numbering_without_a_dup_or_a_skip_at_the_handoffs() {
    CASE();
    Stream s;
    HoldNumbering hold{kFps};
    int64_t copies = 0;
    // The slide page: 60 paints on the browser's grid, then 4 s without a paint, every 6 s. The hold
    // (repeat_last_frame, ahead of the stamping) fills the pause with copies stamped on its train.
    for (int64_t cycle = 0; cycle < 20; ++cycle) {
        const int64_t start = T0 + cycle * 180 * BROWSER;
        for (int64_t i = 0; i < 60; ++i) {
            const int64_t stamp = start + i * BROWSER;
            const int64_t arrival = stamp + D + jitter(cycle * 60 + i);
            hold.live(arrival, stamp / kUs, kUsTb);
            s.frame(arrival, stamp);
        }
        const int64_t next_arrival = start + 180 * BROWSER + D + jitter((cycle + 1) * 60);
        int64_t n = 0;
        while (hold.nextRepeatNs() + slop(n) < next_arrival) {
            const int64_t now = hold.nextRepeatNs() + slop(n++);
            CHECK(hold.due(now));
            const int64_t pts_us = hold.copy(now);
            s.frame(now + 50 * kUs, pts_us * kUs, true);   // the copy reaches the stamping at once
            ++copies;
        }
        step("cycle ", cycle, " copies ", n);
        CHECK(n >= 119 && n <= 121);
    }
    s.consecutive(0, s.slots.size() - 1);   // every slot once: paints, copies, and both handoffs
    CHECK(s.stats().shifts_later == 0 && s.stats().shifts_back == 0 && s.stats().realignments == 0);
    CHECK(hold.realignments() == 0);
    CHECK(copies >= 20 * 119);
    s.never_late(GUARD);
}

void a_paint_after_a_pause_without_a_hold_is_realigned_to_its_own_stamp() {
    CASE();
    Stream s;
    paints(s, T0, 0, 60);
    const int64_t resume = T0 + 180 * BROWSER;   // 4 s later
    const auto r = s.frame(resume + D, resume);
    CHECK(r.realigned && r.shifted == 0);
    CHECK(s.stamping.timeOf(r.slot) >= resume && s.stamping.timeOf(r.slot) < resume + P);   // round_up
    CHECK(s.margin(60) >= L - D);
    paints(s, resume, 1, 60);
    s.consecutive(60, 119);
    CHECK(s.stats().realignments == 1 && s.gaps() == 1 && s.dups() == 0);
}

void a_reset_starts_the_numbering_again_from_the_next_paint() {
    CASE();
    Stream s;
    paints(s, T0, 0, 10);
    s.stamping.reset();
    const int64_t t1 = T0 + 1000 * kMs;
    const auto r = s.frame(t1 + D, t1);
    CHECK(!r.realigned && !r.shifted);
    CHECK(s.stamping.timeOf(r.slot) >= t1 && s.stamping.timeOf(r.slot) < t1 + P);
    CHECK(s.stats().realignments == 0);
}

} // namespace

int main() {
    steady_paints_are_numbered_consecutively_and_keep_the_latency_in_hand();
    the_probes_27_ms_late_paint_moves_the_numbering_one_slot_later_once();
    paints_hovering_at_the_guard_shift_once_and_stay();
    paints_at_28_fps_cost_a_skipped_slot_twice_a_second_and_no_discard();
    the_browser_grid_drifting_the_paints_earlier_costs_one_discard_in_an_hour_at_most();
    a_stall_followed_by_a_burst_plays_the_backlog_out_and_settles_within_a_few_seconds();
    hold_copies_and_paints_share_one_numbering_without_a_dup_or_a_skip_at_the_handoffs();
    a_paint_after_a_pause_without_a_hold_is_realigned_to_its_own_stamp();
    a_reset_starts_the_numbering_again_from_the_next_paint();
    std::cout << "paint stamping OK\n";
}
