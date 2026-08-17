#include "doctr_recognizer.hpp"

#include <avcpp/av.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kRecH = 32;
constexpr int kRecW = 128;
constexpr int kRecSteps = 33;

const std::string kFrenchVocab =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    R"(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)"
    "°£€¥¢฿"
    "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";

std::vector<std::string> splitUtf8Codepoints(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) break;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

const std::vector<std::string> kFrenchTokens = splitUtf8Codepoints(kFrenchVocab);

ocr::Recognition decodeParseq(const float* logits, int steps, int classes) {
    ocr::Recognition result;
    float confidence_sum = 0.0f;
    int confidence_count = 0;
    const int eos = (int)kFrenchTokens.size();
    if (classes != eos + 1) {
        throw Error("doctr recognizer class count does not match french vocabulary");
    }
    for (int step = 0; step < steps; ++step) {
        const float* row = logits + step * classes;
        int best = 0;
        float maximum = row[0];
        for (int cls = 1; cls < classes; ++cls) {
            if (row[cls] > maximum) { maximum = row[cls]; best = cls; }
        }
        float denominator = 0.0f;
        for (int cls = 0; cls < classes; ++cls) denominator += std::exp(row[cls] - maximum);
        const float probability = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        if (best == eos) break;
        if (best >= 0 && best < eos) {
            result.text += kFrenchTokens[(size_t)best];
            confidence_sum += probability;
            ++confidence_count;
        }
    }
    result.confidence = confidence_count ? confidence_sum / (float)confidence_count : 0.0f;
    return result;
}

void requireCuda(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    throw Error(std::string("doctr CUDA failure in ") + operation);
}

} // namespace

namespace ocr {

DoctrRecognizer::~DoctrRecognizer() {
    cleanup();
}

void DoctrRecognizer::cleanup() noexcept {
    runner_.cleanup();
    if (d_boxes_) {
        cuMemFree(d_boxes_);
        d_boxes_ = 0;
    }
    preprocess_kernel_ = nullptr;
    batch_size_ = 0;
}

void DoctrRecognizer::init(TrtLogger& logger, CUmodule preprocess_module,
                           const std::string& engine_path, int batch_size) {
    cleanup();
    if (batch_size <= 0) throw Error("doctr recognizer batch size must be positive");
    requireCuda(cuModuleGetFunction(&preprocess_kernel_, preprocess_module,
                                    "kNV12_doctr_crop_resize_pad_f32"),
                "cuModuleGetFunction(kNV12_doctr_crop_resize_pad_f32)");
    runner_.init(logger, engine_path,
                 TensorContract{"", nvinfer1::DataType::kFLOAT, {batch_size, 3, kRecH, kRecW}},
                 TensorContract{"", nvinfer1::DataType::kFLOAT,
                                {batch_size, kRecSteps, (int64_t)kFrenchTokens.size() + 1}});
    requireCuda(cuMemAlloc(&d_boxes_, (size_t)batch_size * 4 * sizeof(int)), "cuMemAlloc(boxes)");
    batch_size_ = batch_size;
}

std::vector<Recognition> DoctrRecognizer::recognize(const av::VideoFrame& frame,
                                                     const std::vector<PixelBox>& boxes) {
    if ((int)boxes.size() > batch_size_) throw Error("doctr recognizer batch overflow");
    if (boxes.empty()) return {};
    std::vector<int> packed((size_t)batch_size_ * 4, 0);
    for (size_t i = 0; i < boxes.size(); ++i) {
        const PixelBox& box = boxes[i];
        packed[i * 4 + 0] = box.x1;
        packed[i * 4 + 1] = box.y1;
        packed[i * 4 + 2] = std::max(1, box.x2 - box.x1);
        packed[i * 4 + 3] = std::max(1, box.y2 - box.y1);
    }
    for (size_t i = boxes.size(); i < (size_t)batch_size_; ++i) {
        packed[i * 4 + 2] = 1;
        packed[i * 4 + 3] = 1;
    }
    requireCuda(cuMemcpyHtoDAsync(d_boxes_, packed.data(), packed.size() * sizeof(int),
                                  runner_.stream()), "cuMemcpyHtoDAsync(boxes)");

    CUdeviceptr y_plane = (CUdeviceptr)(uintptr_t)frame.raw()->data[0];
    CUdeviceptr uv_plane = (CUdeviceptr)(uintptr_t)frame.raw()->data[1];
    int y_pitch = frame.raw()->linesize[0];
    int uv_pitch = frame.raw()->linesize[1];
    CUdeviceptr output = runner_.inputPtr();
    int batch = batch_size_;
    int height = kRecH;
    int width = kRecW;
    float mean_r = 0.694f, mean_g = 0.695f, mean_b = 0.693f;
    float std_r = 0.299f, std_g = 0.296f, std_b = 0.301f;
    void* args[] = {
        &y_plane, &y_pitch, &uv_plane, &uv_pitch, &output, &d_boxes_,
        &batch, &height, &width,
        &mean_r, &mean_g, &mean_b, &std_r, &std_g, &std_b,
    };
    constexpr unsigned int block_x = 16;
    constexpr unsigned int block_y = 16;
    const unsigned int grid_x = (unsigned int)((width + (int)block_x - 1) / (int)block_x);
    const unsigned int grid_y = (unsigned int)((height + (int)block_y - 1) / (int)block_y);
    requireCuda(cuLaunchKernel(preprocess_kernel_, grid_x, grid_y, (unsigned int)batch,
                               block_x, block_y, 1, 0, runner_.stream(), args, nullptr),
                "cuLaunchKernel(doctr preprocess)");
    runner_.infer();

    std::vector<Recognition> results;
    results.reserve(boxes.size());
    const int classes = (int)kFrenchTokens.size() + 1;
    for (size_t i = 0; i < boxes.size(); ++i) {
        results.push_back(decodeParseq(
            runner_.output().data() + i * kRecSteps * classes, kRecSteps, classes));
    }
    return results;
}

} // namespace ocr
