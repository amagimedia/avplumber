# Playlist Control Correctness Design

## Scope

Correct two defects in the playlist regression demo without changing AVPlumber
framework or node behavior:

1. Whole-playlist Play is incorrectly redirected by the highlighted element.
2. A superseded asynchronous activation can visibly select its stale source.

The existing permanent Janus output, source-only Stop behavior, playback modes,
fixture media, and TUI layout remain unchanged.

## Playlist and element selection

The selected element is an edit and item-action target. The active element is
the whole playlist's playback position. Highlighting a row changes only the
selected element; it must not redirect a whole-playlist transport action.

Whole-playlist Play uses these rules:

- after Pause, resume the active element;
- after Stop, restart the active element from its configured cue-in;
- before any element has been active, start the first enabled element.

Restarting the active element must preserve a different highlighted selection.
ITEM PLAY remains the explicit operation for playing the selected element.

## Superseded activation

Source preparation remains asynchronous and serialized on the existing backend
worker. After the target produces its ready frame, the backend resumes it for
the cut and performs a final request-validity check immediately before the
visible switch is committed.

If a newer request supersedes the activation at that boundary, the backend
stops the prepared target and returns without changing the active item, active
slot, probe item, or source-switcher selection. Submission of the switch command
is the activation commit point; a request accepted after that point is the next
activation rather than a cancellation of the committed cut.

This preserves non-blocking public backend methods. The fix does not hold a
state lock across an AVPlumber control command and does not introduce new graph
or framework behavior.

## Verification seams

Tests exercise public behavior at two boundaries:

1. `PlaylistController` actions and status plus calls observed by a fake backend.
   LIST STOP, highlighting another row, and LIST PLAY must restart the prior
   active item while preserving the highlighted selection. The equivalent
   Pause-to-Play case must resume the prior active item.
2. `AsyncPlaylistBackend` commands and events through a fake AVPlumber API. A
   newer request injected during the stale target's final resume must prevent
   that target's switch command and ready event; only the newer target may cut.

Each defect is implemented as a separate red-green TDD slice. Completion also
requires the complete playlist test suite and the remote live Janus regression,
including output liveness across Stop and Stop-to-Play.
