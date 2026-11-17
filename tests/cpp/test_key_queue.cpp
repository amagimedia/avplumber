#include "mixer/primitives/KeyQueue.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

namespace {

using Queue = avp::mixer::KeyQueue<int, int64_t>;
constexpr size_t kBound = 6;   // the keyer's

// Stamps count half ticks, so tick k is stamped 2k and ends (nearest tick) at 2k + 1.
int64_t stamp(int tick) { return 2 * tick; }
int shownAt(Queue &queue, int tick) {
    const int *frame = queue.at(stamp(tick) + 1);
    return frame ? *frame : -1;
}
// A paint is its own number; a held copy of paint n is -n.
int copyOf(int paint) { return -paint; }
int picture(int frame) { return frame < 0 ? -frame : frame; }

void frames_in_stamp_order_show_on_their_ticks() {
    Queue queue(kBound);
    CHECK(queue.at(stamp(0) + 1) == nullptr);   // nothing to show before the first frame
    for (int tick = 0; tick < 100; ++tick) {
        // Four ticks ahead of the program, as with a four-frame playout latency at 60 fps.
        for (int ahead = tick ? tick + 4 : 0; ahead <= tick + 4; ++ahead) queue.push(100 + ahead, stamp(ahead));
        CHECK(queue.queued() == 5);
        CHECK(shownAt(queue, tick) == 100 + tick);
        CHECK(queue.queued() == 4);
    }
    CHECK(queue.stats().superseded == 0 && queue.stats().replaced == 0 && queue.stats().early == 0);
}

void a_tick_without_a_frame_shows_the_last_one_again() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    queue.push(12, stamp(2));
    CHECK(shownAt(queue, 0) == 10);
    CHECK(shownAt(queue, 1) == 10);
    CHECK(shownAt(queue, 2) == 12);
    queue.push(13, stamp(3));
    queue.push(14, stamp(4));
    CHECK(shownAt(queue, 4) == 14);   // a program frame that skipped tick 3 shows the newest due
    CHECK(queue.stats().superseded == 0 && queue.stats().replaced == 0 && queue.stats().early == 0);
}

void a_paint_replaces_the_copy_held_for_its_tick() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    queue.push(copyOf(10), stamp(1));   // the page was quiet for a period and a half
    queue.push(11, stamp(1));           // its paint, before the program reached tick 1
    CHECK(queue.queued() == 2 && queue.stats().replaced == 1 && queue.stats().superseded == 0);
    CHECK(shownAt(queue, 0) == 10);
    CHECK(shownAt(queue, 1) == 11);
}

// A delivery stall of three periods at 60 fps: the hold emitted two copies, numbered on from
// the last paint, before the stalled paints came through with the stamps they always had.
// First in, first out showed 10, 10, 12, 13: the copy for tick 2 stood before paint 11, so tick 1
// repeated and paint 11 was never shown.
void paints_delivered_late_are_not_kept_behind_the_copies_numbered_past_them() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    queue.push(copyOf(10), stamp(1));
    queue.push(copyOf(10), stamp(2));
    CHECK(shownAt(queue, 0) == 10);
    for (int paint = 11; paint <= 13; ++paint) queue.push(paint, stamp(paint - 10));
    // Both copies are older pictures than paint 11: the one for tick 2 stood before it, the one
    // for tick 1 shares its stamp.
    CHECK(queue.stats().superseded == 1 && queue.stats().replaced == 1);
    CHECK(queue.queued() == 3);
    for (int tick = 1; tick <= 3; ++tick) CHECK(shownAt(queue, tick) == 10 + tick);
    CHECK(queue.stats().early == 0);
}

// A copy re-aligned to the slot of its emission is numbered up to three slots past the
// paint it copies. First in, first out showed 10, 10, 10, 13: two repeats, two paints lost.
void a_copy_numbered_three_slots_ahead_does_not_hold_back_the_paints_before_it() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    queue.push(copyOf(10), stamp(3));
    for (int paint = 11; paint <= 13; ++paint) queue.push(paint, stamp(paint - 10));
    CHECK(queue.stats().superseded == 1 && queue.stats().replaced == 0);
    for (int tick = 0; tick <= 3; ++tick) CHECK(shownAt(queue, tick) == 10 + tick);
}

void a_copy_already_shown_stays_shown_until_a_paint_is_due() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    queue.push(copyOf(10), stamp(1));
    CHECK(shownAt(queue, 0) == 10);
    CHECK(picture(shownAt(queue, 1)) == 10);   // the paint for tick 1 missed its program frame
    queue.push(11, stamp(1));
    queue.push(12, stamp(2));
    CHECK(queue.stats().superseded == 0 && queue.stats().replaced == 0);   // nothing queued was stamped at or after them
    CHECK(shownAt(queue, 2) == 12);             // too late for its tick: the next one shows the newest due
}

// A page that stays quiet: the copies are all there is, each on its own tick.
void copies_of_a_quiet_page_show_in_turn() {
    Queue queue(kBound);
    queue.push(10, stamp(0));
    for (int tick = 0; tick < 50; ++tick) {
        queue.push(copyOf(10), stamp(tick + 1));
        CHECK(picture(shownAt(queue, tick)) == 10);
        CHECK(queue.queued() == 1);
    }
    CHECK(queue.stats().superseded == 0 && queue.stats().replaced == 0 && queue.stats().early == 0);
}

void a_source_that_restarts_its_clock_drops_what_was_queued_for_the_old_one() {
    Queue queue(kBound);
    queue.push(10, stamp(1000));
    queue.push(11, stamp(1001));
    queue.push(20, stamp(0));
    CHECK(queue.queued() == 1 && queue.stats().superseded == 2 && queue.stats().replaced == 0);
    CHECK(shownAt(queue, 0) == 20);
}

void a_full_queue_shows_its_oldest_frame_early_and_counts_it() {
    Queue queue(kBound);
    for (int tick = 0; tick <= 6; ++tick) queue.push(100 + tick, stamp(tick));   // a seventh frame
    CHECK(queue.queued() == kBound && queue.stats().early == 1);
    CHECK(queue.shown() && *queue.shown() == 100);
    CHECK(shownAt(queue, 0) == 100);   // the one early frame is the next to show: nothing lost
    queue.push(107, stamp(7));          // again the next to show (101) ...
    queue.push(108, stamp(8));          // ... and one more before the next program frame
    CHECK(queue.stats().early == 3);
    CHECK(shownAt(queue, 1) == 102);    // paint 101 is skipped,
    CHECK(shownAt(queue, 2) == 102);    // and the tick after repeats
}

// With the queue as full as a four-frame latency makes it, a copy and the paint for the same
// tick used to take two places: the second pushed the frame of the tick not yet drawn out.
void a_copy_and_its_paint_take_one_place_in_a_full_queue() {
    Queue queue(kBound);
    for (int tick = 0; tick <= 5; ++tick) queue.push(100 + tick, stamp(tick));
    queue.push(copyOf(105), stamp(6));
    CHECK(queue.stats().early == 1);    // frame 100, the next to show
    queue.push(106, stamp(6));
    CHECK(queue.stats().early == 1 && queue.stats().replaced == 1 && queue.queued() == kBound);
    for (int tick = 0; tick <= 6; ++tick) CHECK(shownAt(queue, tick) == 100 + tick);
}

void a_frame_set_to_show_is_shown_until_a_queued_one_is_due() {
    Queue queue(kBound);
    queue.show(7);
    CHECK(shownAt(queue, 0) == 7 && shownAt(queue, 500) == 7);
    queue.push(8, stamp(501));
    CHECK(shownAt(queue, 500) == 7);
    CHECK(shownAt(queue, 501) == 8);
}

} // namespace

int main(int argc, char **argv) {
    const std::vector<std::pair<std::string, void (*)()>> cases = {
        {"frames_in_stamp_order_show_on_their_ticks", frames_in_stamp_order_show_on_their_ticks},
        {"a_tick_without_a_frame_shows_the_last_one_again", a_tick_without_a_frame_shows_the_last_one_again},
        {"a_paint_replaces_the_copy_held_for_its_tick", a_paint_replaces_the_copy_held_for_its_tick},
        {"paints_delivered_late_are_not_kept_behind_the_copies_numbered_past_them",
         paints_delivered_late_are_not_kept_behind_the_copies_numbered_past_them},
        {"a_copy_numbered_three_slots_ahead_does_not_hold_back_the_paints_before_it",
         a_copy_numbered_three_slots_ahead_does_not_hold_back_the_paints_before_it},
        {"a_copy_already_shown_stays_shown_until_a_paint_is_due", a_copy_already_shown_stays_shown_until_a_paint_is_due},
        {"copies_of_a_quiet_page_show_in_turn", copies_of_a_quiet_page_show_in_turn},
        {"a_source_that_restarts_its_clock_drops_what_was_queued_for_the_old_one",
         a_source_that_restarts_its_clock_drops_what_was_queued_for_the_old_one},
        {"a_full_queue_shows_its_oldest_frame_early_and_counts_it", a_full_queue_shows_its_oldest_frame_early_and_counts_it},
        {"a_copy_and_its_paint_take_one_place_in_a_full_queue", a_copy_and_its_paint_take_one_place_in_a_full_queue},
        {"a_frame_set_to_show_is_shown_until_a_queued_one_is_due", a_frame_set_to_show_is_shown_until_a_queued_one_is_due},
    };
    try {
        for (const auto &test : cases) {
            if (argc > 1 && std::string(argv[1]) != test.first) continue;
            test.second();
            std::cout << "OK: " << test.first << "\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
