// A pause whose synchronizing seek fails leaves the team as it was and wakes what went to sleep
// in the meantime: a pause node sleeps until resetInput(), as on a resume (tests/test_pause_team.py).
#include "PauseControlTeam.hpp"
#include <cassert>
#include <memory>
#include <stdexcept>

namespace {

struct FailingSeek: IFlushAndSeek {
    void flushAndSeek(StreamTarget) override { throw std::runtime_error("Can not determine current time"); }
};

struct CountingSeek: IFlushAndSeek {
    int seeks = 0;
    void flushAndSeek(StreamTarget) override { ++seeks; }
};

struct CountingInput: IInputReset {
    int resets = 0;
    void resetInput() override { ++resets; }
};

void aFailedPauseWakesTheInputsAndLeavesTheTeamRunning() {
    auto team = std::make_shared<PauseControlTeam>();
    auto seek = std::make_shared<FailingSeek>();
    auto input = std::make_shared<CountingInput>();
    team->addSyncObj(seek);
    team->addNode(input);
    team->resumeAt(12345);

    bool threw = false;
    try {
        team->pause();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(!team->isPaused());
    assert(input->resets == 1);       // a pause node asleep on the failed pause is woken
    assert(team->waitForResume(0));
}

void aFailedPauseOfAPausedTeamLeavesItPausedAndAsleep() {
    auto team = std::make_shared<PauseControlTeam>();
    auto input = std::make_shared<CountingInput>();
    team->addNode(input);
    team->pause(false);
    auto seek = std::make_shared<FailingSeek>();
    team->addSyncObj(seek);

    bool threw = false;
    try {
        team->pause();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(team->isPaused());
    assert(input->resets == 0);       // it was paused before: nothing to wake
}

void aPauseThatSucceedsWakesNothing() {
    auto team = std::make_shared<PauseControlTeam>();
    auto seek = std::make_shared<CountingSeek>();
    auto input = std::make_shared<CountingInput>();
    team->addSyncObj(seek);
    team->addNode(input);
    team->pause();
    assert(team->isPaused());
    assert(seek->seeks == 1);
    assert(input->resets == 0);
    team->resume();
    assert(!team->isPaused());
    assert(input->resets == 1);
}

}  // namespace

int main() {
    aFailedPauseWakesTheInputsAndLeavesTheTeamRunning();
    aFailedPauseOfAPausedTeamLeavesItPausedAndAsleep();
    aPauseThatSucceedsWakesNothing();
    return 0;
}
