# Shared-Baseline Residual Histogram Team Classifier

## Goal

Fix the histogram team classifier by removing the appearance signal that players from both teams share before bootstrap, assignment, and handoff. Keep the current CUDA extractor unchanged for this pass.

## Problem

The current branch classifies on raw `UV` and `L` torso histograms. In practice those histograms still contain a large shared component:

- skin that survives the current filter
- shadows and neutral dark bins from the full player mask
- floor reflections and compression noise
- generic clothing structure that is common to both teams

That means the classifier in `team_classifier.cpp` is often clustering "player-like torso appearance" instead of "team-distinct jersey appearance". The result is weak prototype separation and unstable or wrong assignments, especially when one team is dark and the other is light or neutral.

## Scope

Only `src/nodes/neural_net/sport_specific/team_classifier.cpp` changes in this pass.

The following remain unchanged:

- `jersey_color_extract.cu`
- `jersey_color_extract.cpp`
- GPU mask side data flow
- tracker and draw nodes

Example params may be adjusted only if the residual distance scale makes the current defaults obviously wrong.

## Design

### Core idea

Learn a frozen per-shot shared baseline histogram from all eligible tracks, subtract it from each sample, clamp to positive residual, renormalize, and do all team separation work in that residual space.

Conceptually:

`residual[i] = max(0, raw[i] - shared[i])`

Then:

- if residual mass is near zero, treat the sample as weak evidence
- otherwise normalize the residual histogram and use it for clustering and assignment

This is histogram-space common-mode rejection. Skin tone diversity stops being a blocker because the shared component is modeled from the actual observed players in the shot instead of from a hardcoded skin rule.

### Shared baseline estimation

During bootstrap, collect all eligible track EMAs exactly as the current code already does.

For each UV bin and each L bin:

- gather the value from every eligible track histogram
- compute the median across tracks
- store that as the frozen shared baseline for the current shot

Use separate frozen arrays:

- `shared_uv_[256]`
- `shared_l_[16]`

These are reset only on shot transition or EOF, alongside the rest of classifier state.

Median is preferred over mean because a median baseline is resistant to one team temporarily dominating the frame early in bootstrap.

### Residualization points

Residualization happens only inside `team_classifier.cpp`.

1. Bootstrap:
   Build team prototypes from residualized track EMAs, not raw EMAs.

2. Assignment:
   Score the current detection histogram against residualized prototypes using the residualized current sample when available, else the residualized track EMA.

3. Online prototype update:
   Update prototypes using residualized samples only.

4. Recent-appearance handoff:
   Store and compare residualized histograms, not raw histograms.

Raw EMAs still remain in per-track state because they are the stable underlying observation history. Residual histograms are derived views used for team discrimination.

### Weak evidence handling

Residual subtraction can legitimately collapse a sample when most of its mass is shared.

Add a residual mass gate after subtraction and before normalization:

- sum the residual histogram mass
- if mass is below epsilon, mark the sample as weak / unusable

Behavior for weak residuals:

- skip bootstrap sample contribution
- skip assignment lock decisions from that sample
- skip prototype update
- skip handoff insertion

The classifier should fall back to existing sticky track state when available, but it must not force a new assignment from a nearly empty residual.

### Prototype representation

Team prototypes remain histogram pairs:

- `proto_uv_[2][256]`
- `proto_l_[2][16]`

But they now live in residual space. Prototype separation checks also operate in residual space.

### Parameter impact

No new node parameters are required for the first pass.

The existing params remain valid:

- `uv_weight`
- `l_weight`
- `soft_assignment_margin`
- `initial_assignment_margin`
- `assignment_margin`
- `bootstrap_min_prototype_distance`
- `handoff_max_hist_distance`

Those values may need tuning after the fix because residualization will usually increase inter-team contrast and decrease the common low-information mass.

## Data Flow

1. `jersey_color_extract` emits raw normalized `jersey_uv_hist` and `jersey_l_hist`.
2. `team_classifier` updates raw per-track EMAs from matched detections.
3. Once bootstrap conditions are met, classifier computes frozen shared baseline histograms from the track EMA pool.
4. Classifier residualizes track EMAs against the shared baseline.
5. K-means bootstrap builds residual-space team prototypes.
6. Later frames residualize current samples or track EMAs before assignment, centroid update, and handoff.

## Error Handling

- If there are too few eligible tracks, bootstrap does nothing.
- If residualized samples collapse to zero mass for too many tracks, bootstrap remains pending.
- If prototype separation after residualization is still below threshold, remain unbootstrapped.
- Shot transitions reset raw track state, shared baseline, prototypes, and recent appearances.

## Why this pass first

This is the safest fix because it addresses the failure mode without moving the CUDA or segmentation boundary:

- no kernel changes
- no metadata contract changes
- no need to invent skin heuristics for varied skin tones
- easy to reason about from the current branch

If residualization is still not enough, the next step would be extractor tightening. That should be a follow-up, not bundled into this fix.

## Acceptance Criteria

- bootstrap succeeds with stronger prototype separation than the raw-histogram branch on the same clip
- white-team and dark-team players cluster by team more consistently across a wide shot
- new tracks are less likely to inherit the wrong team from shared non-jersey torso signal
- no regressions in metadata write-back or debug overlay behavior
