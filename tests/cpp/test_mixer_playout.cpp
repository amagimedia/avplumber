#include "mixer/Playout.hpp"
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>

#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(#expr); } while (false)

void burst_keeps_every_frame() {
    // Two equally paced sources arrive in a burst after a delayed receiver wake.
    // Neither the early source's future frame nor the late source's first frame
    // may be discarded just because both are visible at the first deadline.
    avp::mixer::Playout<int> mix(2, {60, 1});
    mix.push(0, 100, 0);
    mix.push(1, 200, 4000000);
    mix.push(0, 101, 18000000);
    mix.push(1, 201, 23000000);
    CHECK(!mix.prepare(33000000));
    auto first = mix.prepare(34000000);
    CHECK(first && first->index == 0);
    CHECK(*first->frames[0] == 100 && *first->frames[1] == 200);
    mix.commit();
    mix.push(0, 102, 34000000);
    mix.push(1, 202, 38000000);
    auto second = mix.prepare(51000000);
    CHECK(second && second->index == 1);
    CHECK(*second->frames[0] == 101 && *second->frames[1] == 201);
    mix.commit();
    CHECK(mix.stats(0).repeats == 0 && mix.stats(1).discarded == 0);

}

void sixteen_independent_phases() {
    avp::mixer::Playout<int> mix(16, {60, 1});
    int next[16] = {};
    int previous[16] = {};
    for (int tick = 0; tick < 3600; ++tick) {
        const int64_t wake = tick * 1000000000LL / 60 + 35333333;
        for (int source = 0; source < 16; ++source) {
            while (true) {
                const int id = next[source];
                const int64_t arrival = id * 1000000000LL / 60 + source * 900000 +
                    ((id * 7 + source * 3) % 13) * 1000000;
                if (arrival > wake) break;
                mix.push(source, id, arrival);
                ++next[source];
            }
        }
        const auto *decision = mix.prepare(wake);
        CHECK(decision);
        for (int source = 0; source < 16; ++source) {
            if (tick > 20) CHECK(*decision->frames[source] == previous[source] + 1);
            if (decision->frames[source]) previous[source] = *decision->frames[source];
        }
        mix.commit();
    }
}

void bounded_queue_counts_overflow() {
    avp::mixer::Playout<int> mix(1, {60, 1});
    for (int id = 0; id < 100; ++id) mix.push(0, id, id * 1000000000LL / 60);
    CHECK(mix.queued(0) <= 8);
    CHECK(mix.stats(0).overflow == 92);
}

void latency_cannot_exceed_retained_frames() {
    bool rejected = false;
    try { avp::mixer::Playout<int> unsupported(1, {60, 1}, 200.0); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected);
    avp::mixer::Playout<int> mix(1, {60, 1}, 100.0,
                                avp::mixer::TimestampMode::Presentation);
    int next = 0;
    for (int output = 0; output < 120; ++output) {
        const int64_t now = output * 1000000000LL / 60 + 100000000;
        while (next * 1000000000LL / 60 <= now) {
            mix.push(0, next, next * 1000000000LL / 60);
            ++next;
        }
        const auto *decision = mix.prepare(now);
        CHECK(decision && *decision->frames[0] == output);
        mix.commit();
    }
    CHECK(mix.stats(0).overflow == 0 && mix.stats(0).discarded == 0);
}

void missed_deadlines_do_not_catch_up_in_bursts() {
    avp::mixer::Playout<int> mix(1, {60, 1});
    mix.push(0, 0, 0);
    CHECK(mix.prepare(34000000));
    mix.commit();
    mix.push(0, 1, 16666667);
    mix.push(0, 2, 33333333);
    mix.push(0, 3, 50000000);
    auto decision = mix.prepare(85000000);
    CHECK(decision && decision->index == 3);
    CHECK(*decision->frames[0] == 3);
    mix.commit();
    CHECK(mix.missedDeadlines() == 2);
    CHECK(mix.stats(0).discarded == 2);
    CHECK(!mix.prepare(85000000));
}

void latency_and_backpressure() {
    avp::mixer::Playout<int> mix(1, {60, 1}, 50.0);
    mix.push(0, 42, 0);
    CHECK(!mix.prepare(49999999));
    const auto first = mix.prepare(50000000);
    CHECK(first && *first->frames[0] == 42);
    CHECK(mix.prepare(200000000) == first);
    CHECK(mix.stats(0).repeats == 0);
    mix.commit();
    CHECK(mix.stats(0).repeats == 0);
    CHECK(mix.latencyNs() == 50000000);
}

void source_clock_gap_recovers_bounded_delay() {
    avp::mixer::Playout<int> mix(1, {60, 1});
    mix.push(0, 10, 0);
    CHECK(mix.prepare(34000000));
    mix.commit();
    // Producer restarts after half a second. Do not assign the new frame to
    // the old missing slot and keep half a second of latency indefinitely.
    mix.push(0, 11, 500000000);
    auto early = mix.prepare(500000000);
    CHECK(early && *early->frames[0] == 10);
    mix.commit();
    auto decision = mix.prepare(534000000);
    CHECK(decision && *decision->frames[0] == 11);
    mix.commit();
    mix.push(0, 12, 516666667);
    decision = mix.prepare(551000000);
    CHECK(decision && *decision->frames[0] == 12);
    mix.commit();
    CHECK(mix.stats(0).discontinuities == 1);
}

void preserves_presentation_timestamps_for_rate_conversion() {
    avp::mixer::Playout<int> mix(1, {60, 1}, {}, avp::mixer::TimestampMode::Presentation);
    mix.push(0, 10, 0);
    mix.push(0, 11, 33333333);
    CHECK(*mix.prepare(34000000)->frames[0] == 10);
    mix.commit();
    CHECK(*mix.prepare(51000000)->frames[0] == 10);
    mix.commit();
    CHECK(*mix.prepare(68000000)->frames[0] == 11);
    mix.commit();
    CHECK(mix.stats(0).repeats == 1);
    CHECK(mix.stats(0).discarded == 0);
}

void rational_clock_and_reference_lifetime() {
    const avp::mixer::FrameRate ntsc(30000, 1001);
    CHECK(ntsc.time(30000) == 1001000000000LL);
    CHECK(ntsc.time(1800000) == 60060000000000LL);
    CHECK(ntsc.atOrBefore(33366666) == 1);
    CHECK(ntsc.atOrBefore(33366665) == 0);
    CHECK(ntsc.time(-1) == -33366667);
    avp::mixer::Playout<std::shared_ptr<int>> mix(1, {60, 1});
    auto frame = std::make_shared<int>(42);
    std::weak_ptr<int> owner = frame;
    mix.push(0, frame, 0);
    frame.reset();
    CHECK(!owner.expired());
    CHECK(mix.prepare(34000000));
    mix.commit();
    CHECK(!owner.expired());
    mix.push(0, std::make_shared<int>(43), 16666667);
    CHECK(mix.prepare(51000000));
    CHECK(!owner.expired());
    mix.commit();
    CHECK(owner.expired());
}

void invalid_parameters_fail_early() {
    for (double latency : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try { avp::mixer::Playout<int> mix(1, {60, 1}, latency); }
        catch (const std::invalid_argument &) { rejected = true; }
        CHECK(rejected);
    }
    avp::mixer::Playout<int> zero(1, {60, 1}, 0.0);
    zero.push(0, 7, 0);
    CHECK(*zero.prepare(0)->frames[0] == 7);
}

void eof_drains_future_frames_before_finishing() {
    avp::mixer::Playout<int> mix(2, {60, 1});
    mix.push(0, 10, 0);
    mix.push(0, 11, 16666667);
    mix.push(1, 20, 0);
    mix.endInput(0);
    mix.endInput(1);
    CHECK(!mix.finished());
    CHECK(*mix.prepare(34000000)->frames[0] == 10);
    mix.commit();
    CHECK(!mix.finished());
    auto last = mix.prepare(51000000);
    CHECK(*last->frames[0] == 11 && *last->frames[1] == 20);
    mix.commit();
    CHECK(mix.finished());
}

void inactive_slot_reactivation_does_not_reuse_old_scene() {
    avp::mixer::Playout<int> mix(2, {60, 1});
    mix.push(0, 10, 0);
    mix.push(1, 20, 0);
    CHECK(mix.prepare(34000000));
    mix.commit();
    mix.setActive(1, false);
    mix.push(0, 11, 16666667);
    auto decision = mix.prepare(51000000);
    CHECK(decision->index == 1 && !decision->frames[1]);
    mix.commit();
    mix.setActive(1, true);
    mix.push(0, 12, 33333333);
    mix.push(1, 99, 33333333);
    decision = mix.prepare(68000000);
    CHECK(decision->index == 2 && *decision->frames[1] == 99);
    mix.commit();
}

void prewarm_waits_for_every_active_slot() {
    avp::mixer::Playout<int> mix(2, {60, 1}, {}, avp::mixer::TimestampMode::Presentation);
    mix.push(0, 10, 0);
    CHECK(!mix.prepare(34000000, true));
    mix.push(1, 20, 16666667);
    CHECK(!mix.prepare(34000000, true));
    CHECK(mix.nextDeadline() == 49999999);
    const auto *decision = mix.prepare(51000000, true);
    CHECK(decision && *decision->frames[0] == 10 && *decision->frames[1] == 20);
    mix.commit();
    CHECK(mix.stats(0).repeats == 0 && mix.stats(1).repeats == 0);
    CHECK(mix.nextDeadline() == 66666666);
}

void irregular_first_paints_do_not_shorten_the_playout_delay() {
    avp::mixer::Playout<int> mix(1, {60, 1});
    mix.push(0, 0, 0);
    int next = 1;
    for (int tick = 0; tick <= 100; ++tick) {
        const int64_t now = tick * 1000000000LL / 60 + 33333334;
        while ((next + 2) * 1000000000LL / 60 <= now) {
            mix.push(0, next, (next + 2) * 1000000000LL / 60);
            ++next;
        }
        auto decision = mix.prepare(now);
        CHECK(decision);
        // Following the first sporadic paint, frame 98 is presented at tick
        // 100. Frame 100 is already available, but is two output ticks early.
        if (tick == 100) CHECK(*decision->frames[0] == 98);
        mix.commit();
    }
}

void prewarmed_slots_share_frame_ids_and_output_ticks() {
    // Hidden and visible slots start at different times. A late-starting
    // preview must use the same content grid; it must not restart PTS at zero.
    using avp::mixer::TimestampMode;
    avp::mixer::Playout<int> program(2, {60, 1}, {}, TimestampMode::Presentation);
    avp::mixer::Playout<int> preview(2, {60, 1}, {}, TimestampMode::Presentation);
    const avp::mixer::FrameRate rate(60, 1);
    for (int tick = 0; tick < 180; ++tick) {
        // Unequal arrival jitter, while presentation timestamps remain exact.
        for (int source = 0; source < 2; ++source) {
            program.push(source, tick + source * 1000, rate.time(tick));
            if (tick >= 37) preview.push(source, tick + source * 1000, rate.time(tick));
        }
        auto pgm = program.prepare(rate.time(tick) + 33333334, true);
        CHECK(pgm && pgm->index == tick);
        if (tick >= 37) {
            auto pvw = preview.prepare(rate.time(tick) + 33333334, true);
            CHECK(pvw && pvw->index == pgm->index);
            CHECK(pvw->frames == pgm->frames);
            preview.commit();
        }
        program.commit();
    }
    CHECK(program.stats(0).repeats == 0 && preview.stats(1).discarded == 0);
}

void scene_reload_discards_prewarm_frames_before_first_visible_frame() {
    using avp::mixer::TimestampMode;
    avp::mixer::Playout<int> mix(2, {60, 1}, {}, TimestampMode::Presentation);
    mix.push(0, 10, 0);
    mix.push(1, 20, 0);
    CHECK(mix.prepare(34000000, true));
    mix.commit();
    mix.push(1, 21, 16666667); // queued from the previous hidden scene
    mix.push(1, 22, 33333333);
    mix.resetInput(1);
    mix.push(0, 11, 16666667);
    CHECK(!mix.prepare(51000000, true)); // no partial/stale new scene
    mix.push(0, 12, 33333333);
    mix.push(1, 99, 33333333);
    CHECK(!mix.prepare(51000000, true)); // new frame still in the future
    auto decision = mix.prepare(68000000, true);
    CHECK(decision && decision->index == 2);
    CHECK(*decision->frames[0] == 12 && *decision->frames[1] == 99);
    mix.commit();
    CHECK(mix.stats(1).discarded == 2);
    CHECK(mix.stats(1).repeats == 0);
}

void stalled_input_does_not_block_healthy_inputs_or_replay_late_burst() {
    using avp::mixer::TimestampMode;
    avp::mixer::Playout<int> mix(2, {60, 1}, {}, TimestampMode::Presentation);
    const avp::mixer::FrameRate rate(60, 1);
    for (int tick = 0; tick < 15; ++tick) {
        mix.push(0, tick, rate.time(tick));
        if (tick < 5 || tick > 10) mix.push(1, 100 + tick, rate.time(tick));
        if (tick == 10) {
            for (int late = 5; late <= 10; ++late)
                mix.push(1, 100 + late, rate.time(late));
        }
        auto decision = mix.prepare(rate.time(tick) + 33333334, true);
        CHECK(decision && decision->index == tick);
        CHECK(*decision->frames[0] == tick);
        CHECK(*decision->frames[1] == (tick >= 5 && tick < 10 ? 104 : 100 + tick));
        mix.commit();
    }
    CHECK(mix.stats(0).repeats == 0 && mix.stats(0).discarded == 0);
    CHECK(mix.stats(1).repeats == 5 && mix.stats(1).discarded == 5);
    CHECK(mix.missedDeadlines() == 0);
}

void route_reset_rejects_old_frames_still_in_upstream_edges() {
    avp::mixer::Playout<int> mix(1, {60, 1}, {}, avp::mixer::TimestampMode::Presentation);
    mix.push(0, 10, 0);
    CHECK(mix.prepare(34000000, true));
    mix.commit();
    mix.resetInput(0, 33333333);
    mix.push(0, 11, 16666667); // old route, delivered after the reset request
    CHECK(!mix.prepare(68000000, true));
    mix.push(0, 99, 33333333);
    auto decision = mix.prepare(68000000, true);
    CHECK(decision && *decision->frames[0] == 99);
    mix.commit();
    CHECK(mix.stats(0).discarded == 1);
}

void presentation_phase_on_half_tick_keeps_every_frame() {
    avp::mixer::Playout<int> mix(1, {60, 1}, {}, avp::mixer::TimestampMode::Presentation);
    const avp::mixer::FrameRate rate(60, 1);
    // A valid 60 Hz source starts at 25 ms, exactly halfway between output
    // ticks. Converting 1/60 to integer ns alternates rounding errors every
    // three frames; those sub-ns errors must not become repeats/skips.
    int next = 0;
    for (int tick = 0; tick < 300; ++tick) {
        if (tick == 100) {
            mix.resetInput(0);
            next = tick;
        }
        while (next <= tick + 1) {
            mix.push(0, next, 25000000 + rate.time(next));
            ++next;
        }
        auto frame = mix.prepare(rate.time(tick + 2) + 33333334, true);
        CHECK(frame && frame->index == tick + 2);
        CHECK(*frame->frames[0] == tick);
        mix.commit();
    }
    CHECK(mix.stats(0).repeats == 0 && mix.stats(0).discarded == 1); // reset's one future frame
}

void resumed_source_is_live_after_an_eof_marker() {
    avp::mixer::Playout<int> mix(2, {60, 1});
    mix.push(0, 10, 0);
    mix.push(1, 20, 0);
    mix.endInput(0);
    CHECK(mix.prepare(34000000));
    mix.commit();
    mix.push(0, 11, 16666667);
    mix.push(1, 21, 16666667);
    mix.endInput(1);
    CHECK(mix.prepare(51000000));
    mix.commit();
    CHECK(!mix.finished());
}

int main(int argc, char **argv) {
    const std::pair<const char *, void (*)()> cases[] = {
        {"latency_cannot_exceed_retained_frames", latency_cannot_exceed_retained_frames},
        {"resumed_source_is_live_after_an_eof_marker", resumed_source_is_live_after_an_eof_marker},
        {"presentation_phase_on_half_tick_keeps_every_frame", presentation_phase_on_half_tick_keeps_every_frame},
        {"route_reset_rejects_old_frames_still_in_upstream_edges", route_reset_rejects_old_frames_still_in_upstream_edges},
        {"prewarmed_slots_share_frame_ids_and_output_ticks", prewarmed_slots_share_frame_ids_and_output_ticks},
        {"scene_reload_discards_prewarm_frames_before_first_visible_frame", scene_reload_discards_prewarm_frames_before_first_visible_frame},
        {"stalled_input_does_not_block_healthy_inputs_or_replay_late_burst", stalled_input_does_not_block_healthy_inputs_or_replay_late_burst},
        {"irregular_first_paints_do_not_shorten_the_playout_delay", irregular_first_paints_do_not_shorten_the_playout_delay},
        {"prewarm_waits_for_every_active_slot", prewarm_waits_for_every_active_slot},
        {"inactive_slot_reactivation_does_not_reuse_old_scene", inactive_slot_reactivation_does_not_reuse_old_scene},
        {"eof_drains_future_frames_before_finishing", eof_drains_future_frames_before_finishing},
        {"rational_clock_and_reference_lifetime", rational_clock_and_reference_lifetime},
        {"invalid_parameters_fail_early", invalid_parameters_fail_early},
        {"preserves_presentation_timestamps_for_rate_conversion", preserves_presentation_timestamps_for_rate_conversion},
        {"source_clock_gap_recovers_bounded_delay", source_clock_gap_recovers_bounded_delay},
        {"latency_and_backpressure", latency_and_backpressure},
        {"burst_keeps_every_frame", burst_keeps_every_frame},
        {"sixteen_independent_phases", sixteen_independent_phases},
        {"bounded_queue_counts_overflow", bounded_queue_counts_overflow},
        {"missed_deadlines_do_not_catch_up_in_bursts", missed_deadlines_do_not_catch_up_in_bursts},
    };
    try {
        for (const auto &test : cases) {
            if (argc > 1 && std::string(argv[1]) != test.first) continue;
            test.second();
            std::cout << "OK: " << test.first << "\n";
            if (argc > 1) return 0;
        }
        return argc > 1 ? 2 : 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
