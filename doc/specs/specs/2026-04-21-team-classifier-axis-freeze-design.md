## Goal

Keep realtime team classification inside the existing C++/TensorRT pipeline while preventing late global team-label swaps after bootstrap and minimizing `unknown` players.

## Problem

The current classifier bootstraps two team centroids from per-track jersey colors and then keeps assigning by nearest centroid. This allows centroid drift to silently exchange the meaning of `team=0` and `team=1`, especially when both teams occupy nearby UV regions or lighting changes over time. Individual track locking prevents per-track flips after assignment, but unassigned or newly seen players can still inherit globally swapped labels.

At the same time, the system should keep coverage high. A fix that only raises assignment thresholds would reduce churn but would also create too many `unknown` players.

## Design

### Frozen identity axis

After bootstrap succeeds, compute a team-separation axis from the two bootstrap centroids:

- `axis = normalize(centroid_1 - centroid_0)` in the classifier's weighted YUV space
- `midpoint = 0.5 * (centroid_0 + centroid_1)` in the same space

This axis becomes the identity reference for the rest of the run. Team meaning is no longer defined by whichever centroid is numerically closer on a later frame.

### Stable side ordering

The bootstrap centroids define a canonical ordering:

- negative projection side -> `team=0`
- positive projection side -> `team=1`

Later centroid adaptation must preserve this ordering. Centroids may move to follow appearance drift, but only within their own team side.

### Assignment by projection

Per-track assignment uses the track EMA color:

1. Project the track color onto the frozen axis relative to the frozen midpoint.
2. Use projection sign to choose the candidate team.
3. Use absolute projection magnitude as the primary assignment margin.

This replaces late-stage identity decisions based on nearest-centroid symmetry.

### Coverage-first lock policy

To keep `unknown` counts low, use two lock paths:

- strong projection margin: assign immediately
- weak projection margin: assign after repeated same-side evidence across a small number of frames

This preserves the current goal of fast team coverage while making identity drift much harder.

### Online centroid updates

Centroids remain useful as appearance references, but they no longer define team identity.

- Update centroid `k` only from tracks already assigned to `k`
- Do not allow unassigned tracks to redefine the global split

This lets the classifier adapt to lighting and camera changes without relabeling the teams.

## Data Flow

1. `jersey_color_extract` produces jersey YUV + confidence.
2. `team_classifier` updates per-track EMA colors.
3. Once bootstrap conditions are met, classifier computes and stores frozen axis + midpoint.
4. New and unassigned tracks are classified by side-of-axis evidence.
5. Assigned tracks update only their own team centroid.

## Error Handling

- If bootstrap centroids are too close, do not freeze axis yet; remain in bootstrap mode until sufficient separation exists.
- If a track has low jersey confidence or too few pixels, keep it unassigned until more evidence arrives.
- If projection margin is weak but consistent, allow delayed assignment through repeated same-side confirmation.

## Testing

- Realtime verification on the current live NBA example.
- Confirm that:
  - all or nearly all players still get classified
  - no late global team inversion occurs after bootstrap
  - newly appearing players inherit the stable global team identity
  - individual `track_id`s do not flip
- Inspect debug logs for bootstrap timing, frozen-axis state, assignment counts, and same-side confirmations.
