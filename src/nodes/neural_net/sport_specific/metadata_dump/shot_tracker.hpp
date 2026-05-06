#pragma once

#include "types.hpp"

#include <cstdint>
#include <string>

namespace metadata_dump {

// Per-frame shot lifecycle state: the last release seen, the currently
// pending shot waiting for a scoreboard verdict, and the cumulative-counter
// edge-detection state used to spot release/arrival/result events from the
// upstream `shot_events` metadata.
//
// Lifecycle methods (onShotRelease/onShotArrival/onShotResult) currently
// remain on MetadataDump because they mutate the active possession's
// shots[] vector — they migrate to PossessionTracker in phase 5.
struct ShotTrackerState {
    bool have_last_release = false;
    PendingRelease last_release;

    bool have_pending_shot = false;
    PendingShot pending_shot;

    int prev_total_releases = 0;
    int prev_total_arrivals = 0;

    int last_known_handler_id = -1;
    std::string last_known_handler_team;

    void reset() { *this = ShotTrackerState{}; }
};

}  // namespace metadata_dump
