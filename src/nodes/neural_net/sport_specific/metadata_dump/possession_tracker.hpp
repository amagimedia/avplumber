#pragma once

#include "types.hpp"

#include <string>

namespace metadata_dump {

// Possession-level accumulation state. The active possession (`poss_acc`),
// the deferred possession (held back when the last shot is awaiting a
// scoreboard verdict), and the team-control change debouncer that gates
// when a new possession starts.
//
// Lifecycle methods (updatePossessionAcc, writePossessionRecord,
// writePbpRecord, flushPossession, etc.) currently remain on MetadataDump
// because they touch file writers, score-tracker state, and clip-summary
// counters — they migrate into the possessions sink node in phase 6.
struct PossessionTrackerState {
    bool poss_active = false;
    PossessionAcc poss_acc;
    bool have_deferred_poss = false;
    PossessionAcc deferred_poss;
    std::string prev_possessing_team;

    // Team-control change debouncer (separate from possession_tracker
    // upstream's debounce; this filters event-line spam).
    int pending_new_handler_id = -1;
    std::string pending_new_handler_team;
    int pending_new_handler_frames = 0;

    void reset() { *this = PossessionTrackerState{}; }
};

}  // namespace metadata_dump
