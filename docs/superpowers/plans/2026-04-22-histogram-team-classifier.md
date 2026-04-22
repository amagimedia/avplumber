# Histogram Team Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single dominant YUV color with 16x16 UV + 16-bin L histograms for team classification using chi-square distance.

**Architecture:** CUDA kernel outputs normalized histograms per player detection. CPU-side `team_classifier` builds team prototypes as averaged histograms and classifies players by chi-square distance. No fallback to old single-color path.

**Tech Stack:** C++17, CUDA PTX, nlohmann/json metadata transport, NV12 frame access.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/nodes/neural_net/sport_specific/jersey_color_extract.cu` | Modify | Add L histogram accumulation, output both UV and L histograms normalized |
| `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp` | Modify | Add GPU buffers for histograms, DtoH copy, write histogram arrays to JSON metadata |
| `src/nodes/neural_net/sport_specific/team_classifier.cpp` | Modify | Replace single-color state/distance/bootstrap with histogram-based equivalents |
| `examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber` | Modify | Update team_classifier params for histogram distance scale |

---

### Task 1: CUDA kernel — add L histogram and histogram output

**Files:**
- Modify: `src/nodes/neural_net/sport_specific/jersey_color_extract.cu`

- [ ] **Step 1: Add L histogram constants and shared memory**

At the top of the file, after the existing histogram constants (line 3-5), add:

```cuda
constexpr int kHistBinsL = 16;
```

Inside `kJerseyUVMean`, after `__shared__ int s_skin_count;` (line 70), add:

```cuda
__shared__ int s_l_hist[kHistBinsL];
```

In the shared memory init block (after line 73), add L histogram zeroing:

```cuda
if (tid < kHistBinsL) {
    s_l_hist[tid] = 0;
}
```

- [ ] **Step 2: Add new output parameters to kernel signature**

After the existing `float* __restrict__ out_confidence` parameter (line 60), add:

```cuda
float* __restrict__ out_uv_hist,
float* __restrict__ out_l_hist
```

- [ ] **Step 3: Accumulate L histogram in the pixel loop**

Inside the cloth pixel accumulation block (after line 192, where `atomicAdd(&s_cloth_count, 1)` is), add:

```cuda
const int bl = min(kHistBinsL - 1, max(0, y * kHistBinsL / 256));
atomicAdd(&s_l_hist[bl], 1);
```

- [ ] **Step 4: Output normalized histograms in thread 0 block**

At the end of the thread 0 block (after line 228, before the closing `}`), add histogram output:

```cuda
const float inv_cloth = (s_cloth_count > 0) ? (1.0f / (float)s_cloth_count) : 0.0f;
for (int i = 0; i < kHistBins; ++i) {
    out_uv_hist[det * kHistBins + i] = (float)s_hist_count[i] * inv_cloth;
}
for (int i = 0; i < kHistBinsL; ++i) {
    out_l_hist[det * kHistBinsL + i] = (float)s_l_hist[i] * inv_cloth;
}
```

Also handle the early-exit zero-output paths (lines 93-101 and lines 123-131) — add zeroing of the histogram outputs:

```cuda
for (int i = 0; i < kHistBins; ++i) out_uv_hist[det * kHistBins + i] = 0.0f;
for (int i = 0; i < kHistBinsL; ++i) out_l_hist[det * kHistBinsL + i] = 0.0f;
```

- [ ] **Step 5: Commit**

```bash
git add src/nodes/neural_net/sport_specific/jersey_color_extract.cu
git commit -m "feat(jersey_color_extract): output normalized UV+L histograms from CUDA kernel"
```

---

### Task 2: jersey_color_extract.cpp — GPU buffers and histogram metadata output

**Files:**
- Modify: `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp`

- [ ] **Step 1: Add histogram size constants**

At the top of the class, after the existing member variables (around line 136), add:

```cpp
static constexpr int kUVHistSize = 256;
static constexpr int kLHistSize = 16;
```

- [ ] **Step 2: Add device buffer members**

After `CUdeviceptr d_out_confidence_` (line 149), add:

```cpp
CUdeviceptr d_out_uv_hist_ = 0;
CUdeviceptr d_out_l_hist_ = 0;
```

- [ ] **Step 3: Update releaseBuffers()**

In `releaseBuffers()` (around line 170), add cleanup for the two new buffers before `capacity_ = 0`:

```cpp
if (d_out_uv_hist_) { JERSEY_CHECK_CU(cuMemFree(d_out_uv_hist_)); d_out_uv_hist_ = 0; }
if (d_out_l_hist_) { JERSEY_CHECK_CU(cuMemFree(d_out_l_hist_)); d_out_l_hist_ = 0; }
```

- [ ] **Step 4: Update ensureCapacity()**

In `ensureCapacity()` (around line 205-219), add allocations after `d_out_confidence_`:

```cpp
if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_uv_hist_, (size_t)next * kUVHistSize * sizeof(float)))) return false;
if (JERSEY_CHECK_CU(cuMemAlloc(&d_out_l_hist_, (size_t)next * kLHistSize * sizeof(float)))) return false;
```

- [ ] **Step 5: Update kernel args array**

In `process()`, the `args[]` array (lines 413-449) currently ends with `&d_out_confidence_`. Add the two new pointers:

```cpp
(void*)&d_out_uv_hist_,
(void*)&d_out_l_hist_
```

- [ ] **Step 6: Add DtoH copy for histograms**

After the existing DtoH copies for `host_confidence` (around line 493), add:

```cpp
std::vector<float> host_uv_hist(packed.size() * kUVHistSize, 0.0f);
std::vector<float> host_l_hist(packed.size() * kLHistSize, 0.0f);
if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_uv_hist.data(), d_out_uv_hist_, host_uv_hist.size() * sizeof(float)))) {
    if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
    this->sink_->put(frm);
    return;
}
if (JERSEY_CHECK_CU(cuMemcpyDtoH(host_l_hist.data(), d_out_l_hist_, host_l_hist.size() * sizeof(float)))) {
    if (d_debug_masks) JERSEY_CHECK_CU(cuMemFree(d_debug_masks));
    this->sink_->put(frm);
    return;
}
```

- [ ] **Step 7: Replace jersey_y/jersey_uv with histogram arrays in JSON output**

In the per-detection output loop (around lines 501-527), replace the block that writes `jersey_uv` and `jersey_y` (lines 512-517):

```cpp
// Old code to remove:
//   Parameters uv = Parameters::array();
//   uv.push_back(host_best_yuv[i * 3u + 1u]);
//   uv.push_back(host_best_yuv[i * 3u + 2u]);
//   det["jersey_uv"] = uv;
//   det["jersey_y"] = host_best_yuv[i * 3u];
```

Replace with:

```cpp
Parameters uv_hist = Parameters::array();
for (int b = 0; b < kUVHistSize; ++b) {
    uv_hist.push_back(host_uv_hist[i * kUVHistSize + b]);
}
det["jersey_uv_hist"] = uv_hist;

Parameters l_hist = Parameters::array();
for (int b = 0; b < kLHistSize; ++b) {
    l_hist.push_back(host_l_hist[i * kLHistSize + b]);
}
det["jersey_l_hist"] = l_hist;
```

Keep the `if (host_cloth_count[i] >= min_pixels_ && host_best_count[i] > 0)` gate — only emit histograms when there are enough cloth pixels.

- [ ] **Step 8: Commit**

```bash
git add src/nodes/neural_net/sport_specific/jersey_color_extract.cpp
git commit -m "feat(jersey_color_extract): transport UV+L histograms in metadata JSON"
```

---

### Task 3: team_classifier.cpp — histogram-based classification

This is the largest change. Replace all single-color state, distance functions, bootstrap, assignment, and handoff with histogram equivalents.

**Files:**
- Modify: `src/nodes/neural_net/sport_specific/team_classifier.cpp`

- [ ] **Step 1: Add histogram constants and chi-square distance function**

At the top of the anonymous namespace (after the `iequals` function, around line 24), add:

```cpp
constexpr int kUVHistBins = 256;
constexpr int kLHistBins = 16;

float chiSquareDistance(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        const float s = a[i] + b[i];
        if (s > 1e-7f) sum += (d * d) / s;
    }
    return sum;
}

void normalizeHistogram(float* hist, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += hist[i];
    if (sum > 1e-7f) {
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) hist[i] *= inv;
    }
}

void emaUpdateHistogram(float* ema, const float* sample, int n, float alpha) {
    for (int i = 0; i < n; ++i) {
        ema[i] = (1.0f - alpha) * ema[i] + alpha * sample[i];
    }
    normalizeHistogram(ema, n);
}
```

- [ ] **Step 2: Replace TrackColor struct**

Replace the `TrackColor` struct (lines 50-60) with:

```cpp
struct TrackColor {
    float uv_hist_ema[kUVHistBins] = {};
    float l_hist_ema[kLHistBins] = {};
    float last_confidence = 0.0f;
    int assigned_team = -1;
    int initial_candidate_team = -1;
    int initial_candidate_frames = 0;
    uint32_t hits = 0;
    uint64_t last_frame = 0;
};
```

- [ ] **Step 3: Replace ParsedSeg struct**

Replace the `ParsedSeg` struct (lines 62-75) to carry histograms instead of single color:

```cpp
struct ParsedSeg {
    int det_index = -1;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float uv_hist[kUVHistBins] = {};
    float l_hist[kLHistBins] = {};
    int pixels = 0;
    float confidence = 0.0f;
};
```

- [ ] **Step 4: Replace CurrentEvidence struct**

Replace the `CurrentEvidence` struct (lines 77-84) with:

```cpp
struct CurrentEvidence {
    bool has_sample = false;
    float uv_hist[kUVHistBins] = {};
    float l_hist[kLHistBins] = {};
    float confidence = 0.0f;
    int pixels = 0;
};
```

- [ ] **Step 5: Replace RecentAppearance struct**

Replace the `RecentAppearance` struct (lines 86-97) with:

```cpp
struct RecentAppearance {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    float uv_hist[kUVHistBins] = {};
    float l_hist[kLHistBins] = {};
    float confidence = 0.0f;
    int team = -1;
    uint64_t frame = 0;
};
```

- [ ] **Step 6: Replace class member variables**

In the `TeamClassifier` class, replace the single-color parameters and state (lines 110-152):

Remove these members:
- `luma_weight_`
- `centroids_[2][3]`
- `identity_axis_[3]`
- `identity_midpoint_[3]`
- `identity_axis_separation_`

Add these members:

```cpp
float uv_weight_ = 1.0f;
float l_weight_ = 0.5f;
float bootstrap_min_prototype_distance_ = 0.1f;
float handoff_max_hist_distance_ = 0.15f;

float proto_uv_[2][kUVHistBins] = {};
float proto_l_[2][kLHistBins] = {};
```

Keep all other members: `tracks_`, `bootstrapped_`, `frame_counter_`, `recent_appearances_`, and all the margin/threshold/handoff members except `handoff_max_color_distance_` (replaced by `handoff_max_hist_distance_`).

- [ ] **Step 7: Update resetState()**

Replace `resetState()` (lines 154-164) with:

```cpp
void resetState() {
    tracks_.clear();
    std::memset(proto_uv_, 0, sizeof(proto_uv_));
    std::memset(proto_l_, 0, sizeof(proto_l_));
    bootstrapped_ = false;
    frame_counter_ = 0;
    recent_appearances_.clear();
}
```

Add `#include <cstring>` at the top of the file if not present.

- [ ] **Step 8: Replace distance functions**

Remove `toWeightedColor()`, `teamDistance()`, `axisProjection()`, `weightedColorDistance()` (lines 166-221).

Replace with:

```cpp
float combinedHistDistance(const float* uv_a, const float* l_a,
                           const float* uv_b, const float* l_b) const {
    return uv_weight_ * chiSquareDistance(uv_a, uv_b, kUVHistBins) +
           l_weight_ * chiSquareDistance(l_a, l_b, kLHistBins);
}

float teamDistance(const float* uv_hist, const float* l_hist, int team) const {
    return combinedHistDistance(uv_hist, l_hist, proto_uv_[team], proto_l_[team]);
}
```

- [ ] **Step 9: Replace parseJerseyColor with parseJerseyHistogram**

Remove the static `parseJerseyColor()` function (lines 191-199). Replace with:

```cpp
static bool parseJerseyHistogram(const Parameters& det,
                                  float* uv_hist, float* l_hist,
                                  float& confidence) {
    if (!det.contains("jersey_uv_hist") || !det["jersey_uv_hist"].is_array() ||
        (int)det["jersey_uv_hist"].size() < kUVHistBins) return false;
    if (!det.contains("jersey_l_hist") || !det["jersey_l_hist"].is_array() ||
        (int)det["jersey_l_hist"].size() < kLHistBins) return false;
    for (int i = 0; i < kUVHistBins; ++i) {
        uv_hist[i] = det["jersey_uv_hist"][i].get<float>();
    }
    for (int i = 0; i < kLHistBins; ++i) {
        l_hist[i] = det["jersey_l_hist"][i].get<float>();
    }
    confidence = det.value("jersey_confidence", 0.0f);
    return true;
}
```

- [ ] **Step 10: Update seg detection parsing**

In `process()`, the seg parsing loop (lines 461-469) calls `parseJerseyColor`. Replace:

```cpp
for (int i = 0; i < (int)seg_md["detections"].size(); ++i) {
    auto& det = seg_md["detections"][i];
    if (!det.is_object() || !matchesLabel(det, seg_labels_)) continue;
    ParsedSeg s;
    if (!parseBBox(det, s.x1, s.y1, s.x2, s.y2)) continue;
    if (!parseJerseyHistogram(det, s.uv_hist, s.l_hist, s.confidence)) continue;
    s.pixels = det.value("jersey_cloth_pixels", det.value("jersey_pixels", 0));
    s.det_index = i;
    segs.push_back(s);
}
```

- [ ] **Step 11: Update track EMA update**

In the matched-pair update block (lines 530-555), replace the single-color EMA with histogram EMA:

```cpp
for (size_t ti = 0; ti < tracked.size(); ++ti) {
    const int si = track_to_seg[ti];
    if (si < 0) continue;
    const ParsedSeg& seg = segs[(size_t)si];
    CurrentEvidence& evidence = current_evidence[ti];
    evidence.has_sample = true;
    std::memcpy(evidence.uv_hist, seg.uv_hist, sizeof(evidence.uv_hist));
    std::memcpy(evidence.l_hist, seg.l_hist, sizeof(evidence.l_hist));
    evidence.confidence = seg.confidence;
    evidence.pixels = seg.pixels;
    if (seg.pixels < min_jersey_pixels_ || seg.confidence < min_jersey_confidence_) continue;
    TrackColor& tc = tracks_[tracked[ti].track_id];
    if (tc.hits == 0) {
        std::memcpy(tc.uv_hist_ema, seg.uv_hist, sizeof(tc.uv_hist_ema));
        std::memcpy(tc.l_hist_ema, seg.l_hist, sizeof(tc.l_hist_ema));
    } else {
        emaUpdateHistogram(tc.uv_hist_ema, seg.uv_hist, kUVHistBins, ema_alpha_track_);
        emaUpdateHistogram(tc.l_hist_ema, seg.l_hist, kLHistBins, ema_alpha_track_);
    }
    tc.last_confidence = seg.confidence;
    tc.hits += 1;
    tc.last_frame = frame_counter_;
    matched += 1;
}
```

- [ ] **Step 12: Replace bootstrapCentroids() with bootstrapPrototypes()**

Replace the entire `bootstrapCentroids()` method (lines 288-368) with:

```cpp
void bootstrapPrototypes() {
    struct Sample {
        float uv_hist[kUVHistBins];
        float l_hist[kLHistBins];
    };
    std::vector<Sample> samples;
    samples.reserve(tracks_.size());
    for (const auto& kv : tracks_) {
        if (kv.second.hits == 0 || kv.second.last_confidence < min_jersey_confidence_) continue;
        Sample s;
        std::memcpy(s.uv_hist, kv.second.uv_hist_ema, sizeof(s.uv_hist));
        std::memcpy(s.l_hist, kv.second.l_hist_ema, sizeof(s.l_hist));
        samples.push_back(s);
    }
    if ((int)samples.size() < bootstrap_min_tracks_) return;

    // Init: first sample as seed 0
    std::memcpy(proto_uv_[0], samples[0].uv_hist, sizeof(proto_uv_[0]));
    std::memcpy(proto_l_[0], samples[0].l_hist, sizeof(proto_l_[0]));

    // Farthest sample as seed 1
    float best_d = -1.0f;
    size_t best_i = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const float d = combinedHistDistance(samples[i].uv_hist, samples[i].l_hist,
                                             proto_uv_[0], proto_l_[0]);
        if (d > best_d) { best_d = d; best_i = i; }
    }
    std::memcpy(proto_uv_[1], samples[best_i].uv_hist, sizeof(proto_uv_[1]));
    std::memcpy(proto_l_[1], samples[best_i].l_hist, sizeof(proto_l_[1]));

    // K-means iterations
    for (int iter = 0; iter < 10; ++iter) {
        float sum_uv[2][kUVHistBins] = {};
        float sum_l[2][kLHistBins] = {};
        int count[2] = {0, 0};
        for (const auto& s : samples) {
            const float d0 = combinedHistDistance(s.uv_hist, s.l_hist, proto_uv_[0], proto_l_[0]);
            const float d1 = combinedHistDistance(s.uv_hist, s.l_hist, proto_uv_[1], proto_l_[1]);
            const int k = (d0 <= d1) ? 0 : 1;
            for (int b = 0; b < kUVHistBins; ++b) sum_uv[k][b] += s.uv_hist[b];
            for (int b = 0; b < kLHistBins; ++b) sum_l[k][b] += s.l_hist[b];
            count[k] += 1;
        }
        for (int k = 0; k < 2; ++k) {
            if (count[k] > 0) {
                const float inv = 1.0f / (float)count[k];
                for (int b = 0; b < kUVHistBins; ++b) proto_uv_[k][b] = sum_uv[k][b] * inv;
                for (int b = 0; b < kLHistBins; ++b) proto_l_[k][b] = sum_l[k][b] * inv;
                normalizeHistogram(proto_uv_[k], kUVHistBins);
                normalizeHistogram(proto_l_[k], kLHistBins);
            }
        }
    }

    const float sep = combinedHistDistance(proto_uv_[0], proto_l_[0], proto_uv_[1], proto_l_[1]);
    if (sep < bootstrap_min_prototype_distance_) return;

    bootstrapped_ = true;

    logstream << "team_classifier: bootstrap"
              << " frame=" << frame_counter_
              << " tracks=" << samples.size()
              << " prototype_sep=" << sep;
}
```

- [ ] **Step 13: Update the bootstrap call site**

In `process()`, replace `bootstrapCentroids()` call (line 559) with `bootstrapPrototypes()`.

- [ ] **Step 14: Replace per-track assignment logic**

The assignment block (lines 578-696) uses axis projection. Replace the inner logic for each tracked player. The structure stays the same (strong/weak/handoff lock) but distance computation changes:

For each tracked player with bootstrapped state and hits > 0:

```cpp
const bool use_current_sample = evidence.has_sample &&
                                evidence.pixels >= min_jersey_pixels_ &&
                                evidence.confidence >= min_jersey_confidence_;
const float* sample_uv = use_current_sample ? evidence.uv_hist : tc.uv_hist_ema;
const float* sample_l = use_current_sample ? evidence.l_hist : tc.l_hist_ema;
const float sample_confidence = use_current_sample ? evidence.confidence : tc.last_confidence;

const float dist0 = teamDistance(sample_uv, sample_l, 0);
const float dist1 = teamDistance(sample_uv, sample_l, 1);
const int candidate_team = (dist0 <= dist1) ? 0 : 1;
const float margin = std::fabs(dist0 - dist1);
```

All subsequent checks that used `margin >= initial_assignment_margin_` etc. stay the same — they now operate on chi-square margin instead of axis projection magnitude.

Remove the centroid EMA update that used `can_update_centroid`. Replace with prototype EMA update:

```cpp
if (team >= 0 && can_update_centroid && candidate_team == team) {
    emaUpdateHistogram(proto_uv_[team], sample_uv, kUVHistBins, ema_alpha_centroid_);
    emaUpdateHistogram(proto_l_[team], sample_l, kLHistBins, ema_alpha_centroid_);
}
```

- [ ] **Step 15: Update handoff to use histogram distance**

Replace `findRecentHandoff()` to use histogram comparison instead of `weightedColorDistance`. Change the color distance check:

```cpp
const float hist_distance = combinedHistDistance(
    sample.uv_hist, sample.l_hist, it->uv_hist, it->l_hist);
if (hist_distance > handoff_max_hist_distance_) continue;
```

Update the score computation:

```cpp
const float score = center_rel * 16.0f + hist_distance * 100.0f + (float)age * 1.5f;
```

The `HandoffMatch` struct's `color_distance` field becomes `hist_distance`.

- [ ] **Step 16: Update RecentAppearance population**

In the block that populates `recent_appearances_` (lines 736-753), replace the single y/u/v copies with histogram copies:

```cpp
std::memcpy(recent.uv_hist, evidence.uv_hist, sizeof(recent.uv_hist));
std::memcpy(recent.l_hist, evidence.l_hist, sizeof(recent.l_hist));
```

- [ ] **Step 17: Update write-back of jersey fields to player metadata**

In the block that copies jersey fields from seg to player (lines 698-731), remove the copy of `jersey_y` and `jersey_uv`. Add copy of `jersey_uv_hist` and `jersey_l_hist`:

```cpp
if (seg_det.contains("jersey_uv_hist")) player_det["jersey_uv_hist"] = seg_det["jersey_uv_hist"];
if (seg_det.contains("jersey_l_hist")) player_det["jersey_l_hist"] = seg_det["jersey_l_hist"];
```

Remove the lines that copy `jersey_y` and `jersey_uv`.

- [ ] **Step 18: Update debug log line**

Update the per-frame debug log (lines 761-786) to show prototype separation instead of axis separation:

```cpp
const float proto_sep = bootstrapped_
    ? combinedHistDistance(proto_uv_[0], proto_l_[0], proto_uv_[1], proto_l_[1])
    : 0.0f;
```

Replace `axis_sep` with `proto_sep` in the log. Remove references to `identity_axis_separation_`, `soft_margin`, `init_margin`, `centroid_margin` labels — replace with `proto_sep`, `uv_weight`, `l_weight`.

- [ ] **Step 19: Update create() parameter parsing**

In the `create()` static method (lines 791-839):

Remove parsing for: `luma_weight`, `bootstrap_axis_min_separation`, `handoff_max_color_distance`.

Add parsing for:

```cpp
if (params.count("uv_weight")) r->uv_weight_ = params["uv_weight"].get<float>();
if (params.count("l_weight")) r->l_weight_ = params["l_weight"].get<float>();
if (params.count("bootstrap_min_prototype_distance")) r->bootstrap_min_prototype_distance_ = params["bootstrap_min_prototype_distance"].get<float>();
if (params.count("handoff_max_hist_distance")) r->handoff_max_hist_distance_ = params["handoff_max_hist_distance"].get<float>();
```

- [ ] **Step 20: Commit**

```bash
git add src/nodes/neural_net/sport_specific/team_classifier.cpp
git commit -m "feat(team_classifier): histogram-based classification with chi-square distance"
```

---

### Task 4: Update example script params

**Files:**
- Modify: `examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber`

- [ ] **Step 1: Update team_classifier node params**

On line 30, in the `team_classifier` node JSON, make these param changes:

Remove:
- `"luma_weight": 1.0`

Add:
- `"uv_weight": 1.0`
- `"l_weight": 0.5`
- `"bootstrap_min_prototype_distance": 0.1`

Replace:
- `"soft_assignment_margin": 2.0` → `"soft_assignment_margin": 0.02`
- `"initial_assignment_margin": 4.0` → `"initial_assignment_margin": 0.05`
- `"assignment_margin": 8.0` → `"assignment_margin": 0.08`
- `"bootstrap_axis_min_separation": 6.0` → remove (replaced by `bootstrap_min_prototype_distance`)
- `"handoff_max_color_distance": 18.0` → `"handoff_max_hist_distance": 0.15`

- [ ] **Step 2: Commit**

```bash
git add examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber
git commit -m "feat(example): update team classifier params for histogram distance scale"
```

---

### Task 5: Build verification and remote sync

- [ ] **Step 1: Local build check**

Verify the code compiles locally (will fail to link without CUDA but confirms C++ syntax):

```bash
make clean && make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1 HAVE_NVCC=1 2>&1 | tail -20
```

Fix any compile errors.

- [ ] **Step 2: Rsync changed files to remote**

```bash
rsync -avz --relative -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" \
  /home/jp/git/avplumber/./src/nodes/neural_net/sport_specific/jersey_color_extract.cu \
  /home/jp/git/avplumber/./src/nodes/neural_net/sport_specific/jersey_color_extract.cpp \
  /home/jp/git/avplumber/./src/nodes/neural_net/sport_specific/team_classifier.cpp \
  /home/jp/git/avplumber/./examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber \
  fedora@172.17.36.132:/home/fedora/avplumber/
```

- [ ] **Step 3: Remote build**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  "cd /home/fedora/avplumber && make clean && make -j8 \
  NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1 HAVE_NVOF_FRUC=1 HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'"
```

- [ ] **Step 4: Remote run and verify logs**

```bash
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132 \
  "cd /home/fedora/avplumber && LD_LIBRARY_PATH=/usr/local/lib:/opt/tensorrt/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  ./avplumber -p 20200 -s examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber"
```

Check logs for:
- `jersey_color_extract` shows non-zero histogram data and valid cloth counts
- `team_classifier: bootstrap` line appears with reasonable `prototype_sep` value
- Team assignments stabilize (consistent `assigned_t0` / `assigned_t1` counts)
- No crashes, no CUDA errors

- [ ] **Step 5: Commit any fixes from build/run**

```bash
git add -u
git commit -m "fix: address build/runtime issues from histogram team classifier"
```
