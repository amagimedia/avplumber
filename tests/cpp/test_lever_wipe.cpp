// The lever's wipe: which clip frame a schedule shows, where the lever's ramps start, the
// background the program selector picks, and the take as the orchestrator runs it from these
// pieces (src/mixer/orchestrator/fade.cpp lever(), selectProgram, wipe.cpp selectWipe), each
// stage reading a tick at its own time on a 60 fps program.
#include "mixer/primitives/ClipScrub.hpp"
#include "mixer/primitives/Lever.hpp"
#include "mixer/primitives/TakeClock.hpp"
#include "mixer/transition_control.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <optional>
#include <utility>
#include <vector>

void print_stack_trace() {}

using namespace avp::mixer;

namespace {

// Offsets of a clip decoded at `fps`, as clip_cache's elapsedNs rescales its pts to ns.
std::vector<int64_t> clipOffsets(int fps, int frames) {
    std::vector<int64_t> offsets;
    for (int k = 0; k < frames; ++k) offsets.push_back(av_rescale(k, 1000000000, fps));
    return offsets;
}

void frameMath() {
    const TickGrid grid{av::Rational(60, 1)};
    const auto offsets = clipOffsets(30, 30);
    const int64_t length = clipLengthNs(offsets, grid.time(1));
    assert(std::llabs(length - 1000000000) <= 1);   // the last frame's own duration, from ns-rounded offsets
    assert(clipLengthNs({0}, grid.time(1)) == grid.time(1) && clipLengthNs({}, 1) == 0);
    // The ends, and every frame in order on the way, both ways.
    assert(frameAt(offsets, length, 0, grid) == 0 && frameAt(offsets, length, 1, grid) == 29);
    assert(frameAt(offsets, length, -0.5, grid) == 0 && frameAt(offsets, length, 1.5, grid) == 29);
    size_t previous = 0;
    for (int i = 0; i <= 1000; ++i) {
        const size_t k = frameAt(offsets, length, i / 1000.0, grid);
        assert(k >= previous && k <= previous + 1);
        previous = k;
    }
    for (int i = 1000; i >= 0; --i) {   // back the same way, one frame at a time
        const size_t k = frameAt(offsets, length, i / 1000.0, grid);
        assert(k <= previous && k + 1 >= previous);
        previous = k;
    }
    assert(previous == 0);
    // A variable rate clip: each frame on the tick nearest its own timestamp.
    const std::vector<int64_t> variable{0, 10000000, 50000000, 60000000};
    assert(frameAt(variable, 100000000, 0.3, grid) == 1 && frameAt(variable, 100000000, 0.55, grid) == 2 &&
           frameAt(variable, 100000000, 0.65, grid) == 3);

    // Run on at the clip's own rate, the lever's clip shows on every tick the frame a replay from
    // its first frame shows (clip_cache play: frame k from tick start + nearestIndex(offset k)),
    // whatever the clip's rate against the program's.
    for (const auto& [clip_fps, out_fps] : {std::pair{30, 60}, {25, 50}, {24, 60}, {60, 60}, {50, 50}, {25, 60}}) {
        const TickGrid out{av::Rational(out_fps, 1)};
        const auto clip = clipOffsets(clip_fps, 2 * clip_fps);
        const int64_t clip_length = clipLengthNs(clip, out.time(1));
        const int64_t start = 123456;
        const RampNs run{out.time(start), clip_length};
        for (int64_t j = 0; j < 2 * out_fps; ++j) {
            size_t played = 0;
            for (size_t k = 0; k < clip.size(); ++k)
                if (out.nearestIndex(clip[k]) <= j) played = k;
            assert(frameAt(clip, clip_length, rampAt(run, out.time(start + j)), out) == played);
        }
    }
}

void anchors() {
    const TickGrid grid{av::Rational(60, 1)};
    for (const int64_t delay : {int64_t(0), grid.time(2)}) {
        for (int64_t now = 1000000000; now < 1100000000; now += 1234567) {
            // The player pushed tick T once time(T) + delay passed: the anchor is past all of them.
            const int64_t pushed = grid.atOrBefore(now - delay);
            assert(grid.time(pushed) + delay <= now);
            const int64_t next = playerNextNs(grid, now, delay);
            assert(next == grid.time(pushed + 2) && next > grid.time(pushed + 1));
            assert(leverAnchorNs(now - 50000000, next) == next && leverAnchorNs(next + 5, next) == next + 5);
        }
    }
}

// The program selector's background under the lever's wipe, whatever order the two slots' frames
// of a tick reach it in: each tick goes out at most once, in order, on the side the schedule has
// there, except Program's while Preview is not there yet for a switch to it.
void background() {
    const TickGrid grid{av::Rational(60, 1)};
    const auto t = [&](int64_t tick) { return grid.time(tick); };
    // Past the switch point from tick 10 to tick 20: forward over 19 ticks, back from tick 15.
    RampHistory schedule(RampNs{t(0), t(19)});
    schedule.change({t(30) + t(1) / 2, -t(20), t(15), t(30)});
    const auto past = [&](int64_t tick) { return rampAt(schedule, t(tick)) >= kWipeSwitchAt; };
    assert(!past(9) && past(10) && past(20) && !past(21));

    // Back below the half with Program's frame of the last tick out late, behind Preview's of the
    // next: nothing goes out until Program's own frame of that tick, never that tick twice.
    auto bg = wipeBackground(schedule, true, t(20), t(21), true, t(20));
    assert(!bg.preview && !bg.emit);
    bg = wipeBackground(schedule, bg.preview, t(21), 0, true, t(20));
    assert(!bg.preview && bg.emit);
    // Back, with Program's frame of a tick still past the half beside Preview's of the next.
    bg = wipeBackground(schedule, true, t(20), t(21), true, t(19));
    assert(!bg.preview && !bg.emit);
    // Forward with Preview's frame of the last tick late: the switch, then Preview's own frame.
    bg = wipeBackground(schedule, false, t(10), t(9), true, t(9));
    assert(bg.preview && !bg.emit);
    bg = wipeBackground(schedule, bg.preview, 0, t(10), true, t(9));
    assert(bg.preview && bg.emit);
    // Forward, with Preview's frame of a tick before the switch point beside Program's past it.
    bg = wipeBackground(schedule, false, t(10), t(9), true, t(8));
    assert(bg.preview && !bg.emit);
    // Program goes on while Preview is not ready or not there for the switch; nothing already out.
    assert(wipeBackground(schedule, false, t(12), t(12), false, t(11)).emit);
    bg = wipeBackground(schedule, false, t(10), 0, false, t(9));
    assert(!bg.preview && bg.emit);
    assert(!wipeBackground(schedule, true, t(12), t(12), true, t(12)).emit);
    assert(!wipeBackground(schedule, true, 0, 0, true, t(12)).emit);

    // Every order: each slot's frame of tick T arrives up to 1.5 ticks late (in order per slot),
    // one selector call per arrival with the heads that are there, and the call consumes them.
    for (uint32_t seed = 1; seed <= 500; ++seed) {
        uint32_t random = seed;
        const auto jitter = [&] {
            random = random * 1664525u + 1013904223u;
            return int64_t(random >> 8) % (3 * t(1) / 2);
        };
        std::vector<std::pair<int64_t, int>> arrivals;   // (time, slot: 0 Program, 1 Preview)
        for (int slot = 0; slot < 2; ++slot) {
            int64_t previous = 0;
            for (int64_t tick = 1; tick <= 40; ++tick) {
                previous = std::max(previous, t(tick) + jitter());
                arrivals.push_back({previous, slot});
            }
        }
        std::stable_sort(arrivals.begin(), arrivals.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        int64_t arrived[2] = {0, 0}, taken[2] = {0, 0};
        bool preview = false;
        int64_t last = 0, previewed = 0;
        for (const auto& [at, slot] : arrivals) {
            ++arrived[slot];
            int64_t heads[2];
            for (int s = 0; s < 2; ++s) heads[s] = taken[s] < arrived[s] ? t(taken[s] + 1) : 0;
            bg = wipeBackground(schedule, preview, heads[0], heads[1], true, last);
            preview = bg.preview;
            if (bg.emit) {
                const int64_t tick = grid.atOrBefore(heads[preview]);
                assert(heads[preview] > last);
                if (preview) previewed = tick;
                assert(preview == past(tick) || (past(tick) && !previewed && tick <= 12));
                last = heads[preview];
            }
            for (int s = 0; s < 2; ++s) if (heads[s]) ++taken[s];
        }
        assert(previewed >= 10 && previewed <= 20 && last == t(40));
    }
}

// One lever wipe as the graph runs it, from the orchestrator's pieces in its order: a lever
// position (leverStep, then rampTo from the anchor over the span, to the clip player and the
// keys' clock), and each program tick read by every stage at its own time: the clip player as
// the tick begins or `scrub_delay_ns` after it (clip_cache scrub_delay_ms), the program selector
// once the slots drew it (latency_ms 50), the wipe selector a tick later, the keyer half a tick
// after that.
struct LeverWipe {
    enum class Mode { Idle, Wipe, Cut };
    enum class Ended { No, Completed, Cancelled, Dropped };
    struct Frame {
        double clip = -1;     // the clip player's progress on this tick
        int background = -1;  // 1: Preview (the take's destination), 0: Program
        int keys = -1;        // the keys going with the take: 1 switched
        bool overlay = false; // the wipe selector put the clip's overlay of this tick on air
    };
    const TickGrid grid;
    const int64_t period = grid.time(1);
    const std::vector<int64_t> offsets = clipOffsets(30, 30);
    const int64_t length_ns = clipLengthNs(offsets, period);
    int64_t ready_ticks = 1;   // from the player's first tick to the preset slot and overlay being ready
    const int64_t lag[4];

    Lever lever;
    Mode mode = Mode::Idle;
    bool manual = false, on_air = false, switched = false, scrubbing = false;
    LeverEnd end = LeverEnd::None;
    double take_lever = 0;
    std::optional<FadeRequest> request;
    RampHistory schedule, player;   // the orchestrator's take_schedule, the clip player's own
    TakeClock clock;
    int64_t take_start_ns = 0, lever_at_ns = 0, last_ns = 0, on_air_tick = 0, ended_ns = 0;
    Ended ended = Ended::No;
    int takes = 0;
    std::map<int64_t, Frame> frames;
    int64_t now;
    int64_t next[4];   // the next tick each stage reads: player, program selector, wipe selector, keyer

    explicit LeverWipe(int64_t phase_ns = 3000000, int fps = 60, int64_t scrub_delay_ns = 0)
        : grid(av::Rational(fps, 1)), lag{scrub_delay_ns, 52000000, 52000000 + period, 52000000 + period * 3 / 2},
          now(grid.time(1000) + phase_ns) {
        for (int i = 0; i < 4; ++i) next[i] = grid.atOrBefore(now - lag[i]) + 1;
        last_ns = grid.time(next[1] - 1);   // the program is on air: its selector put out a tick before
    }

    // Each stage reads its ticks up to wall time `to`, in the order their times come.
    void run(int64_t to) {
        while (true) {
            int i = 0;
            for (int k = 1; k < 4; ++k)
                if (grid.time(next[k]) + lag[k] < grid.time(next[i]) + lag[i]) i = k;
            if (grid.time(next[i]) + lag[i] > to) break;
            now = grid.time(next[i]) + lag[i];
            const int64_t tick = next[i]++;
            switch (i) {
                case 0: if (scrubbing) frames[tick].clip = rampAt(player, grid.time(tick)); break;
                case 1: selectProgram(tick); break;
                case 2: selectWipe(tick); break;
                case 3: keyer(tick); break;
            }
        }
        now = to;
    }
    void wait(double ticks) { run(now + int64_t(ticks * double(period))); }

    int64_t anchorNs() const {   // takeScheduleFromNs()
        return leverAnchorNs(last_ns + 2 * period, playerNextNs(grid, now, lag[0]));
    }
    void scheduleTake(const FadeRequest& changed) {   // MixerState::scheduleTake, applyTakeSchedule
        request = changed;
        schedule.change(rampTimesNs(changed));
        player.change(rampTimesNs(changed));
        if (clock.generation && clock.end == TakeClock::End::Running) clock.schedule = schedule;   // clampTakeClock()
    }
    void moveLever(double progress, std::optional<double> span_ms) {
        if (!request) return;
        const int64_t span_ns = leverSpanNs(span_ms, now, lever_at_ns, period);
        lever_at_ns = now;
        const int64_t from_ns = anchorNs();
        scheduleTake(rampTo(*request, progress, from_ns / 1000000, (from_ns + span_ns) / 1000000));
    }
    void finish(Ended how, int64_t t) {   // completeTake, or interruptTransition
        ended = how;
        ended_ns = t;
        if (clock.generation && clock.end == TakeClock::End::Running) {
            clock.end = how == Ended::Completed ? TakeClock::End::Completed : TakeClock::End::Cancelled;
            clock.end_ns = t;
        }
        if (end == LeverEnd::None) lever.release();
        mode = Mode::Idle;
        manual = on_air = switched = scrubbing = false;
        end = LeverEnd::None;
        request.reset();
    }
    void position(double position, std::optional<double> span_ms = 16) {   // leverTo()
        const bool lever_take = manual && mode == Mode::Wipe;
        const auto step = leverStep(lever, position, mode == Mode::Idle, lever_take,
                                    lever_take && end != LeverEnd::None, on_air);
        switch (step.kind) {
            case LeverStep::Kind::Ignore: return;
            case LeverStep::Kind::Drop: finish(Ended::Dropped, 0); return;
            case LeverStep::Kind::Start: {   // armWipe(): the clip held at its first frame
                const int64_t now_ms = now / 1000000;
                FadeRequest held{now_ms, 1.0, true};
                held.time_floor_ms = held.time_ceiling_ms = now_ms;
                request = held;
                schedule = player = RampHistory(rampTimesNs(held));
                scrubbing = true;
                mode = Mode::Wipe;
                manual = true;
                ended = Ended::No;
                lever_at_ns = 0;
                on_air_tick = grid.atOrBefore(now) + 1 + ready_ticks;
                ++takes;
                break;
            }
            case LeverStep::Kind::Move: break;
        }
        take_lever = step.progress;
        end = step.end;
        if (on_air) moveLever(step.progress, span_ms);
        else if (mode == Mode::Wipe) moveLever(std::min(step.progress, kLeverWipeArmedAt), span_ms);
    }
    void resumeLever() { position(lever.position(), std::nullopt); }
    LeverEnd landLeverEnd(int64_t t) {
        if (end == LeverEnd::None || !request || !leverEndLanded(*request, t)) return LeverEnd::None;
        const LeverEnd landed = end;
        if (landed == LeverEnd::Cancel) {
            finish(Ended::Cancelled, t);
            resumeLever();
        }
        return landed;
    }
    void autoTake() {   // runOnFromLever()
        manual = false;
        end = LeverEnd::None;
        lever.release();
        const int64_t at_ms = anchorNs() / 1000000;
        scheduleTake(runOn(*request, at_ms, double(length_ns) / 1e9));
        take_start_ns = request->start_ms * 1000000;
    }
    void autoPressed() {   // actOnTransitionOnAir(), AUTO for the take's scene
        if (on_air && manual) autoTake();
    }
    void cut() {   // actOnTransitionOnAir()'s cut: it owns the take until it lands
        mode = Mode::Cut;
        manual = false;
        end = LeverEnd::None;
    }
    void interrupt() {   // mixer.interrupt, or a take aborted on its deadline
        end = LeverEnd::None;
        finish(Ended::Dropped, 0);
    }

    void selectProgram(int64_t tick) {
        const int64_t t = grid.time(tick);
        Frame& frame = frames[tick];
        if (mode == Mode::Cut) {   // the cut lands on this frame: Preview, the clock completed
            finish(Ended::Completed, t);
            frame.background = 1;
        } else if (mode == Mode::Wipe && on_air) {
            const auto bg = wipeBackground(schedule, switched, t, t, true, last_ns);
            switched = bg.preview;
            assert(bg.emit);
            frame.background = switched;
        } else {
            frame.background = mode == Mode::Idle && ended == Ended::Completed;
        }
        last_ns = t;
    }
    void selectWipe(int64_t tick) {
        const int64_t t = grid.time(tick);
        if (mode == Mode::Wipe && !on_air) {
            if (tick < on_air_tick) return;
            on_air = true;   // the clip, below the switch point until now, goes on to the lever
            take_start_ns = t;
            clock = clock.next(t, length_ns, FadeCurve::Linear, kWipeSwitchAt);
            if (take_lever > kLeverWipeArmedAt) moveLever(take_lever, std::nullopt);
            else clock.schedule = schedule;   // clampTakeClock()
            frames[tick].overlay = true;
            return;
        }
        if (!on_air) return;
        if (mode == Mode::Wipe) {
            const LeverEnd landed = landLeverEnd(t);
            if (landed == LeverEnd::Cancel) return;
            if ((landed == LeverEnd::Complete || (!manual && t >= take_start_ns + length_ns)) && switched) {
                finish(Ended::Completed, t);   // the overlay retired: this tick is the new program
                if (landed == LeverEnd::Complete) resumeLever();
                return;
            }
        }
        frames[tick].overlay = true;
    }
    void keyer(int64_t tick) {   // mixer_keyer followTake()
        const int64_t t = grid.time(tick);
        if (clock.generation == 0 || t < clock.on_air_ns) return;
        frames[tick].keys = clock.end == TakeClock::End::Cancelled ? 0
            : clock.end == TakeClock::End::Completed && t >= clock.end_ns ? 1 : clock.progress(t) == 1;
    }

    // On every tick of the overlay on air the clip, the background and the keys are on the same
    // side of the clip's switch point (up to the ticks the keyer has drawn); returns how many
    // times they crossed it together.
    int together() const {
        int crossings = 0;
        int previous = -1;
        for (const auto& [tick, frame] : frames) {
            if (!frame.overlay || tick >= next[3]) continue;
            const int clip = frame.clip >= kWipeSwitchAt;
            assert(frame.clip >= 0 && frame.background == clip && frame.keys == clip);
            if (previous >= 0 && clip != previous) ++crossings;
            previous = clip;
        }
        return crossings;
    }
    size_t clipFrame(int64_t tick) const { return frameAt(offsets, length_ns, frames.at(tick).clip, grid); }
};

// A streamed T-bar, forward past the switch point, back below it and on to the far end, at
// different speeds and with positions arriving anywhere within a tick: the clip, the background
// and the keys switch on the same tick both ways, and the take completes on the clip's last frame.
// The graph's scrub_delay_ms at `fps`: the overlay's latency_ms (50 here) less a frame.
int64_t graphScrubDelayNs(int fps) { return 50000000 - av_rescale(1, 1000000000, fps); }

void throwBackAndComplete() {
    for (const double per_second : {0.5, 1.0, 1.6, 3.0}) {
        for (int64_t phase_ms = 0; phase_ms < 17; phase_ms += 2) {
            for (const int64_t delay : {int64_t(0), graphScrubDelayNs(60)}) {
                LeverWipe show(phase_ms * 1000000, 60, delay);
                const auto stroke = [&](double from, double to) {
                    const int steps = std::max(1, int(std::fabs(to - from) / per_second * 60));
                    for (int k = 1; k <= steps; ++k) {
                        show.position(from + (to - from) * k / steps);
                        show.wait(1);
                    }
                };
                show.position(0.1);
                stroke(0.1, 0.8);
                stroke(0.8, 0.3);
                show.wait(10);
                assert(show.together() == 2 && show.ended == LeverWipe::Ended::No);
                stroke(0.3, 1.0);
                show.wait(10);
                assert(show.ended == LeverWipe::Ended::Completed && show.clock.end == TakeClock::End::Completed);
                assert(show.together() == 3 && show.clipFrame(show.grid.atOrBefore(show.ended_ns)) == 29);
                // Engaged at the far end: moving it back starts the next lever take.
                assert(show.lever.engaged() && show.lever.origin() == 1);
                show.position(0.9);
                assert(show.mode == LeverWipe::Mode::Wipe && show.manual && show.takes == 2);
            }
        }
    }
}

void cancel() {
    LeverWipe show;
    show.position(0.3);
    show.wait(4);
    show.position(0.7);
    show.wait(8);
    assert(show.frames.rbegin()->second.clip >= kWipeSwitchAt);
    // Back at its start: the background returns at the half, then the take ends with Program as it was.
    show.position(0.0, 50);
    show.wait(12);
    assert(show.ended == LeverWipe::Ended::Cancelled && show.clock.end == TakeClock::End::Cancelled);
    assert(show.together() == 2);
    assert(show.lever.engaged() && show.lever.origin() == 0 && show.mode == LeverWipe::Mode::Idle);
    // Before anything is on air the start drops the take at once.
    LeverWipe early;
    early.position(0.2);
    early.position(0.0);
    assert(early.ended == LeverWipe::Ended::Dropped && early.mode == LeverWipe::Mode::Idle);
}

void autoTakesOver() {
    LeverWipe show;
    show.position(0.3);
    show.wait(10);   // on air
    assert(show.on_air);
    show.autoTake();
    const int64_t from = show.next[0];
    show.wait(60);
    // From 0.3 the clip runs on a frame of its own rate per two program ticks, to its end, completing there.
    assert(show.ended == LeverWipe::Ended::Completed && show.together() == 1);
    size_t clip = show.clipFrame(from), ticks = 0;
    for (const auto& [tick, frame] : show.frames) {
        if (tick < from || !frame.overlay) continue;
        assert(show.clipFrame(tick) >= clip && show.clipFrame(tick) <= clip + 1);
        clip = show.clipFrame(tick);
        ++ticks;
    }
    assert(clip == 29 && std::abs(int(ticks) - 42) <= 4);
    // AUTO released the lever away from its ends: it does nothing until brought to one.
    assert(!show.lever.engaged());
    show.position(0.6);
    assert(show.mode == LeverWipe::Mode::Idle);
}

void cutAndNewTake() {
    LeverWipe show;
    show.position(0.4);
    show.wait(10);
    assert(show.on_air);
    show.cut();
    show.position(0.9);   // the cut owns the program now
    assert(show.request && show.end == LeverEnd::None);
    show.wait(4);
    assert(show.ended == LeverWipe::Ended::Completed && show.clock.end == TakeClock::End::Completed);
    assert(!show.lever.engaged());   // a cut leaves the lever away from its ends inactive
    // A new lever take after the cut starts only from an end.
    show.position(1.0);
    show.position(0.8);
    assert(show.mode == LeverWipe::Mode::Wipe && show.manual && show.request);
}

// A cut pressed on the lever's take owns it until it lands: AUTO under it is ignored, and the
// lever is released there, also when its own end was landing and it moved off that end meanwhile;
// so is a take dropped (mixer.interrupt) or aborted. The next position then does nothing.
void cutOwnsTheTake() {
    LeverWipe show;
    show.position(0.4);
    show.wait(10);
    show.cut();
    const auto request = *show.request;
    show.autoPressed();
    assert(show.request->start_ms == request.start_ms && show.request->time_ceiling_ms == request.time_ceiling_ms);
    assert(show.mode == LeverWipe::Mode::Cut && show.lever.engaged());
    for (const bool cut : {true, false}) {
        LeverWipe landing;
        landing.position(0.6);
        landing.wait(8);
        landing.position(1.0, 100);
        landing.wait(0.5);
        landing.position(0.5);   // kept while the far end lands
        if (cut) landing.cut(); else landing.interrupt();
        landing.wait(4);
        assert(landing.ended == (cut ? LeverWipe::Ended::Completed : LeverWipe::Ended::Dropped));
        assert(!landing.lever.engaged());
        landing.position(0.48);
        assert(landing.mode == LeverWipe::Mode::Idle && landing.takes == 1);
    }
}

// A brisk throw while the preset slot still loads: the clip follows the lever from the arm, below
// its switch point until the overlay is on air (Preview never shows without it), then goes on to
// the lever where the background and the keys go with it.
void throwBeforeOnAir() {
    for (const int64_t ready : {2, 4, 6, 10}) {
        LeverWipe show;
        show.ready_ticks = ready;
        for (int k = 0; k <= 8; ++k) {
            show.position(0.1 + 0.1 * k);
            show.wait(1);
        }
        show.wait(30);
        assert(show.together() == 1 && show.ended == LeverWipe::Ended::No);
        for (const auto& [tick, frame] : show.frames) {
            if (frame.clip >= 0) assert(frame.overlay || (frame.clip < kWipeSwitchAt && frame.background != 1));
            if (frame.overlay) {
                assert(frame.clip > 0 && frame.clip < kWipeSwitchAt);   // not the clip's first frame
                break;
            }
        }
    }
}

// A steady stroke, one position per tick as the page streams them, at the program's rate: from
// its first on-air tick the clip moves on every tick at about the stroke's own rate, never
// a still first frame and then a jump.
void steadyStroke() {
    for (const int fps : {25, 50, 60}) {
        for (const int64_t ready : {0, 1, 3}) {
            for (const int64_t delay : {int64_t(0), graphScrubDelayNs(fps)}) {
                LeverWipe show(3000000, fps, delay);
                show.ready_ticks = ready;
                for (int k = 1; k <= fps; ++k) {
                    show.position(double(k) / fps, 1000 / fps);
                    show.wait(1);
                }
                show.wait(10);
                assert(show.ended == LeverWipe::Ended::Completed && show.together() == 1);
                int still = 0;   // the anchor's two ticks at most, as for the lever's mix
                double previous = -1, step = 0;
                for (const auto& [tick, frame] : show.frames) {
                    if (!frame.overlay) continue;
                    if (frame.clip == 0) ++still;
                    if (previous >= 0) step = std::max(step, frame.clip - previous);
                    previous = frame.clip;
                }
                assert(still <= 2 && step <= 1.2 / fps);
            }
        }
    }
}

// A position off the far end while that end lands is kept; once the take completed it starts
// the next take from there, as moving a T-bar back does. One still at the end starts nothing.
void movedWhileLanding() {
    for (const double after : {0.8, 1.0}) {
        LeverWipe show;
        show.position(0.6);
        show.wait(8);
        show.position(1.0, 100);
        show.wait(0.5);
        show.position(after);
        assert(show.mode == LeverWipe::Mode::Wipe && show.takes == 1);
        show.wait(20);
        assert(show.lever.origin() == 1 && show.lever.engaged());
        if (after < 1) {
            assert(show.takes == 2 && show.mode == LeverWipe::Mode::Wipe && show.manual &&
                   std::fabs(show.take_lever - 0.2) < 1e-9);
        } else {
            assert(show.takes == 1 && show.mode == LeverWipe::Mode::Idle && show.ended == LeverWipe::Ended::Completed);
        }
    }
}

}  // namespace

int main() {
    frameMath();
    anchors();
    background();
    throwBackAndComplete();
    cancel();
    autoTakesOver();
    cutAndNewTake();
    cutOwnsTheTake();
    throwBeforeOnAir();
    steadyStroke();
    movedWhileLanding();
}
