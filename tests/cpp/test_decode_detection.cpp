// CPU unit tests for yolo_base::DetectionDecoder, exercising the real decode
// logic (including the boxes_normalized rescale added for DeepStream/Triton-style
// YOLO exports). Built against a minimal NvInfer.h stub so no GPU toolchain is
// required. Returns non-zero on any failed check.

#include "nodes/neural_net/yolo/decode_detection.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace yolo_base;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        double _a = (a), _b = (b);                                             \
        if (std::fabs(_a - _b) > (eps)) {                                      \
            std::printf("FAIL %s:%d: %s (%.6f) != %s (%.6f)\n", __FILE__,      \
                        __LINE__, #a, _a, #b, _b);                             \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Build a [1, N, 6] end2end output tensor from rows of {x1,y1,x2,y2,conf,cls}.
static nvinfer1::Dims dims3(int a, int b, int c) {
    nvinfer1::Dims d;
    d.nbDims = 3;
    d.d[0] = a; d.d[1] = b; d.d[2] = c;
    return d;
}

// --- Test 1: pixel-space passthrough (L1 scoreboard style), no rescale ---
static void test_pixel_passthrough() {
    std::vector<float> out = {
        10, 20, 110, 220, 0.9f, 0,   // kept
         5,  5,  15,  15, 0.8f, 0,   // kept
         0,  0,   0,   0, 0.0f, 0,   // dropped (conf < thresh, zero pad)
    };
    std::vector<nvinfer1::Dims> odims = { dims3(1, 3, 6) };
    std::vector<const float*> houts = { out.data() };
    std::vector<int> no_remap;

    DecodeParams dp{0, 0.25f, OutputBoxFormat::EndToEndXYXY, no_remap};
    DetectionDecoder dec;
    DetectionResult r = dec.decode(houts, odims, dp);

    CHECK(r.detections.size() == 2);
    if (r.detections.size() >= 1) {
        CHECK_NEAR(r.detections[0].x1, 10.0, 1e-4);
        CHECK_NEAR(r.detections[0].y2, 220.0, 1e-4);
        CHECK_NEAR(r.detections[0].conf, 0.9, 1e-6);
        CHECK(r.detections[0].cls == 0);
    }
}

// --- Test 2: normalized coords rescaled to model pixels (L2 components style) ---
static void test_normalized_rescale() {
    std::vector<float> out = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.95f, 4,   // kept -> *1024
        0.0f, 0.0f, 0.0f, 0.0f, 0.00f, 0,   // dropped
    };
    std::vector<nvinfer1::Dims> odims = { dims3(1, 2, 6) };
    std::vector<const float*> houts = { out.data() };
    std::vector<int> no_remap;

    DecodeParams dp{0, 0.25f, OutputBoxFormat::EndToEndXYXY, no_remap};
    dp.boxes_normalized = true;
    dp.model_w = 1024;
    dp.model_h = 1024;

    DetectionDecoder dec;
    DetectionResult r = dec.decode(houts, odims, dp);

    CHECK(r.detections.size() == 1);
    if (!r.detections.empty()) {
        CHECK_NEAR(r.detections[0].x1, 0.1 * 1024, 1e-3);
        CHECK_NEAR(r.detections[0].y1, 0.2 * 1024, 1e-3);
        CHECK_NEAR(r.detections[0].x2, 0.3 * 1024, 1e-3);
        CHECK_NEAR(r.detections[0].y2, 0.4 * 1024, 1e-3);
        CHECK(r.detections[0].cls == 4);
    }
}

// --- Test 3: boxes_normalized flag off -> no rescale even if model_w/h set ---
static void test_normalized_flag_gates_rescale() {
    std::vector<float> out = { 0.1f, 0.2f, 0.3f, 0.4f, 0.95f, 0 };
    std::vector<nvinfer1::Dims> odims = { dims3(1, 1, 6) };
    std::vector<const float*> houts = { out.data() };
    std::vector<int> no_remap;

    DecodeParams dp{0, 0.25f, OutputBoxFormat::EndToEndXYXY, no_remap};
    dp.boxes_normalized = false;  // gate off
    dp.model_w = 1024;
    dp.model_h = 1024;

    DetectionDecoder dec;
    DetectionResult r = dec.decode(houts, odims, dp);
    CHECK(r.detections.size() == 1);
    if (!r.detections.empty()) {
        CHECK_NEAR(r.detections[0].x1, 0.1, 1e-6);  // unchanged
        CHECK_NEAR(r.detections[0].x2, 0.3, 1e-6);
    }
}

// --- Test 4: non-square rescale uses model_w for x and model_h for y ---
static void test_nonsquare_rescale() {
    std::vector<float> out = { 0.5f, 0.5f, 1.0f, 1.0f, 0.9f, 0 };
    std::vector<nvinfer1::Dims> odims = { dims3(1, 1, 6) };
    std::vector<const float*> houts = { out.data() };
    std::vector<int> no_remap;

    DecodeParams dp{0, 0.25f, OutputBoxFormat::EndToEndXYXY, no_remap};
    dp.boxes_normalized = true;
    dp.model_w = 640;
    dp.model_h = 384;

    DetectionDecoder dec;
    DetectionResult r = dec.decode(houts, odims, dp);
    CHECK(r.detections.size() == 1);
    if (!r.detections.empty()) {
        CHECK_NEAR(r.detections[0].x1, 320.0, 1e-3);  // 0.5 * 640
        CHECK_NEAR(r.detections[0].y1, 192.0, 1e-3);  // 0.5 * 384
        CHECK_NEAR(r.detections[0].x2, 640.0, 1e-3);
        CHECK_NEAR(r.detections[0].y2, 384.0, 1e-3);
    }
}

// --- Test 5: class_index_remap applied after (independent of rescale) ---
static void test_class_index_remap() {
    std::vector<float> out = {
        0.1f, 0.1f, 0.2f, 0.2f, 0.9f, 0,   // cls 0 -> remap to 1
        0.1f, 0.1f, 0.2f, 0.2f, 0.9f, 1,   // cls 1 -> remap to 0
    };
    std::vector<nvinfer1::Dims> odims = { dims3(1, 2, 6) };
    std::vector<const float*> houts = { out.data() };
    std::vector<int> remap = {1, 0};  // swap

    DecodeParams dp{0, 0.25f, OutputBoxFormat::EndToEndXYXY, remap};
    DetectionDecoder dec;
    DetectionResult r = dec.decode(houts, odims, dp);
    CHECK(r.detections.size() == 2);
    if (r.detections.size() == 2) {
        CHECK(r.detections[0].cls == 1);
        CHECK(r.detections[1].cls == 0);
    }
}

int main() {
    test_pixel_passthrough();
    test_normalized_rescale();
    test_normalized_flag_gates_rescale();
    test_nonsquare_rescale();
    test_class_index_remap();

    if (g_failures == 0) {
        std::printf("OK: all decode_detection tests passed\n");
        return 0;
    }
    std::printf("FAILED: %d check(s)\n", g_failures);
    return 1;
}
