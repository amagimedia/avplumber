// The copies repeat_last_frame emits while a page is idle are numbered one slot after the
// last stamp on its output; the emission time never enters a stamp. Driven with a fake
// clock through HoldNumbering, the policy the node delegates to.
#include "mixer/primitives/HoldNumbering.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

using avp::mixer::HoldNumbering;

namespace {

const AVRational kFps{30, 1};
const AVRational kTb{1, 30};               // smooth_timestamps stamps video in 1/fps: pts is the slot
const AVRational kNs{1, 1000000000};
const int64_t P = 33333333;                // the node's period for 30 fps: av_rescale_q(1, 1/fps, 1 ns)
const int64_t kMs = 1000000;
const int64_t kUs = 1000;

int64_t time_of(int64_t slot) { return av_rescale_q(slot, kTb, kNs); }
int64_t round_up(int64_t ns) { return av_rescale_q_rnd(ns, kNs, kTb, AV_ROUND_UP); }   // smooth's alignment
int64_t round_near(int64_t ns) { return av_rescale_q(ns, kNs, kTb); }                  // the old copy stamp
// Wake-up slop of the node in [0, 1 ms): it waits whole milliseconds and emits 0-1 ms after the target.
int64_t slop(int64_t n) { return ((n * 7919) % 1000) * kUs; }

/// What repeat_last_frame::process does around the policy: forward a live frame with its own
/// stamp, or, woken with nothing queued, emit a copy when one is due.
struct Node {
    HoldNumbering hold{kFps};
    std::vector<int64_t> out;              // every stamp put on the output, in 1/30

    void paint(int64_t now, int64_t pts) {
        hold.live(now, pts, kTb);
        out.push_back(pts);                // live frames pass untouched
    }
    bool idle(int64_t now) {
        if (!hold.due(now)) return false;
        out.push_back(hold.copy(now));
        return true;
    }
    int64_t last() const { return out.back(); }
};

// A slide's last paint: stamped 333 us before tick L, which smooth rounds up to L; it reaches the
// hold node a little later, so the copy train a + 1.5 P + n P sits at phase 0.49 of slot L+1+n.
const int64_t L = 1000;
const int64_t kArrival = time_of(L) - P / 100;

void every_stamp_follows_the_previous_one(const std::vector<int64_t> &out) {
    for (size_t i = 1; i < out.size(); ++i) CHECK(out[i] > out[i - 1]);
}

} // namespace

void a_still_page_at_the_half_slot_phase_gets_one_copy_per_slot_where_rounding_gave_pairs() {
    Node node;
    node.paint(kArrival, L);
    CHECK(!node.idle(kArrival + P));                     // nothing between live frames
    int old_dup = 0, old_skip = 0;
    int64_t old_prev = -1;
    for (int64_t n = 0; n < 10000; ++n) {
        const int64_t due = kArrival + P + P / 2 + n * P;   // the train is anchored to the arrival
        CHECK(node.hold.nextRepeatNs() == due);
        CHECK(!node.idle(due - 1));
        const int64_t now = due + slop(n);               // 0.49 P plus up to 1 ms into the slot
        CHECK(node.idle(now));
        CHECK(node.last() == L + 1 + n);                 // counted, whatever the slop
        const int64_t old = round_near(now);             // what the node stamped before
        if (old_prev >= 0) {
            if (old == old_prev) ++old_dup;
            else if (old > old_prev + 1) ++old_skip;
        }
        old_prev = old;
    }
    CHECK(node.out.size() == 10001);
    every_stamp_follows_the_previous_one(node.out);
    CHECK(node.hold.realignments() == 0);
    // The same wake-ups rounded to the nearest slot flip around the half slot: pairs.
    CHECK(old_dup > 100 && old_skip > 100);
}

void a_paint_between_copies_takes_its_own_slot_and_the_next_copy_follows_it() {
    Node node;
    node.paint(kArrival, L);
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(0)) && node.last() == L + 1);
    const int64_t e2 = node.hold.nextRepeatNs() + slop(1);
    CHECK(node.idle(e2) && node.last() == L + 2);
    // The page paints again 10 ms after that copy; smooth rounds the stamp up to the next slot
    // and the frame reaches the hold node 2 ms later, before the next copy would be due.
    const int64_t stamp = e2 + 10 * kMs;
    const int64_t pts = round_up(stamp);
    CHECK(pts == L + 3);
    const int64_t arrival = stamp + 2 * kMs;
    CHECK(!node.idle(arrival - 1));
    node.paint(arrival, pts);
    CHECK(node.last() == L + 3);                         // its own stamp, untouched
    CHECK(node.hold.nextRepeatNs() == arrival + P + P / 2);   // the train re-anchors to the paint
    CHECK(!node.idle(arrival + P + P / 2 - 1));
    CHECK(node.idle(arrival + P + P / 2 + slop(2)) && node.last() == L + 4);   // follows the paint
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(3)) && node.last() == L + 5);
    CHECK(node.hold.realignments() == 0);
    // A late paint: stamped in the slot of the last copy but reaching the node 29 ms after that
    // slot began. It still passes first, with its own stamp; the copy after it would be 1.4
    // periods behind the clock, so the band re-aligns it to the next slot, and the ones after
    // count from there.
    const int64_t late_arrival = node.hold.nextRepeatNs() - P / 2;
    CHECK(late_arrival - time_of(L + 5) > 29 * kMs);
    node.paint(late_arrival, L + 5);
    CHECK(node.last() == L + 5);
    const int64_t after = late_arrival + P + P / 2 + slop(4);
    CHECK(node.idle(after));
    CHECK(node.last() > L + 5 && node.last() == round_up(after) && node.last() == L + 8);
    CHECK(node.hold.realignments() == 1);
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(5)) && node.last() == L + 9);
    CHECK(node.hold.realignments() == 1);
    every_stamp_follows_the_previous_one(node.out);
}

void a_paint_that_reached_the_node_long_after_its_stamp_is_re_aligned_once_then_counted() {
    Node node;
    const int64_t arrival = time_of(L) + 10 * P;         // the import path held the frame for ten periods
    node.paint(arrival, L);
    CHECK(node.last() == L);                             // the live frame still passes with its own stamp
    const int64_t first = node.hold.nextRepeatNs() + slop(0);   // = time(L) + 11.5 P
    CHECK(node.idle(first));
    // Counting from L would put every copy ten periods behind the clock, late at the compositor
    // forever on a page that never paints again; the band takes the next slot instead.
    CHECK(node.last() == round_up(first) && node.last() == L + 12);
    CHECK(node.hold.realignments() == 1);
    for (int64_t n = 1; n <= 100; ++n) {
        const int64_t now = node.hold.nextRepeatNs() + slop(n);
        CHECK(node.idle(now));
        CHECK(node.last() == L + 12 + n);                // counted again from the re-aligned slot
        CHECK(time_of(node.last()) >= now);              // each copy precedes its slot
    }
    CHECK(node.hold.realignments() == 1);                // the band fires once and goes quiet
    every_stamp_follows_the_previous_one(node.out);
}

void a_long_stall_emits_no_copy_at_or_before_the_last_emitted_slot() {
    Node node;
    node.paint(kArrival, L);
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(0)) && node.last() == L + 1);
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(1)) && node.last() == L + 2);
    // The node is woken three periods late for the copy due at time(L+3) + 0.49 P.
    const int64_t due = node.hold.nextRepeatNs();
    const int64_t now = due + 3 * P + 700 * kUs;         // = time(L+6) + 0.51 P
    CHECK(node.idle(now));
    const int64_t c = node.last();
    CHECK(c > L + 2);                                    // never at or before a slot already emitted
    CHECK(c == round_up(now) && c == L + 7);             // re-aligned to the next slot, like smooth does
    CHECK(time_of(c) >= now);                            // so it precedes its slot: in time at the compositor
    CHECK(node.hold.nextRepeatNs() == now + P);          // the train restarts from the wake-up: no burst
    CHECK(node.hold.realignments() == 1);
    CHECK(node.idle(now + P + slop(2)) && node.last() == c + 1);
    CHECK(node.idle(node.hold.nextRepeatNs() + slop(3)) && node.last() == c + 2);
    // A short stall (half a period) is no re-alignment: the copy keeps its number and the train its grid.
    const int64_t due2 = node.hold.nextRepeatNs();
    CHECK(node.idle(due2 + P / 2) && node.last() == c + 3);
    CHECK(node.hold.nextRepeatNs() == due2 + P);
    CHECK(node.hold.realignments() == 1);
    every_stamp_follows_the_previous_one(node.out);
}

void live_paints_at_the_frame_rate_pass_through_unchanged() {
    Node node;
    for (int64_t n = 0; n < 300; ++n) {
        // Each paint is stamped a few ms before its tick (round_up gives slot L+n) and arrives 3 ms later.
        const int64_t stamp = time_of(L + n) - 4 * kMs + ((n * 37) % 20) * 100 * kUs;
        const int64_t pts = round_up(stamp);
        CHECK(pts == L + n);
        const int64_t arrival = stamp + 3 * kMs;
        CHECK(!node.idle(arrival));                      // the 1.5 P schedule never fires between paints
        node.paint(arrival, pts);
        CHECK(node.last() == pts);
    }
    CHECK(node.out.size() == 300);
    for (size_t i = 0; i < node.out.size(); ++i) CHECK(node.out[i] == L + int64_t(i));
}

void copies_in_a_finer_time_base_count_whole_frames_without_accumulating_rounding() {
    // A stream stamped in microseconds at 30 fps: a frame is 33333.33 us, so the copies must be
    // anchor + n frames, not anchor + n * 33333.
    const AVRational us{1, 1000000};
    HoldNumbering hold(kFps);
    const int64_t anchor = 5000000;                      // 5 s on the same clock
    hold.live(anchor * kUs + 3 * kMs, anchor, us);
    int64_t previous = anchor;
    for (int64_t n = 1; n <= 3000; ++n) {
        const int64_t pts = hold.copy(hold.nextRepeatNs() + 200 * kUs);
        CHECK(pts == anchor + av_rescale_q(n, av_inv_q(kFps), us));
        CHECK(pts - previous == 33333 || pts - previous == 33334);
        previous = pts;
    }
    CHECK(previous == anchor + 100000000);               // 3000 frames are exactly 100 s
    CHECK(hold.realignments() == 0);
}

int main() {
    a_still_page_at_the_half_slot_phase_gets_one_copy_per_slot_where_rounding_gave_pairs();
    a_paint_between_copies_takes_its_own_slot_and_the_next_copy_follows_it();
    a_long_stall_emits_no_copy_at_or_before_the_last_emitted_slot();
    a_paint_that_reached_the_node_long_after_its_stamp_is_re_aligned_once_then_counted();
    live_paints_at_the_frame_rate_pass_through_unchanged();
    copies_in_a_finer_time_base_count_whole_frames_without_accumulating_rounding();
    std::cout << "repeat_last_frame numbering OK\n";
}
