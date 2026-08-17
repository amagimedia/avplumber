#pragma once

#include "ocr_trt_runner.hpp"

#include <string>
#include <vector>

namespace av { class VideoFrame; }

namespace ocr {

struct PixelBox {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

struct Recognition {
    std::string text;
    float confidence = 0.0f;
};

class DoctrRecognizer {
public:
    DoctrRecognizer() = default;
    ~DoctrRecognizer();
    DoctrRecognizer(const DoctrRecognizer&) = delete;
    DoctrRecognizer& operator=(const DoctrRecognizer&) = delete;

    void init(TrtLogger& logger, CUmodule preprocess_module,
              const std::string& engine_path, int batch_size);
    void cleanup() noexcept;
    std::vector<Recognition> recognize(const av::VideoFrame& frame,
                                       const std::vector<PixelBox>& boxes);
    int batchSize() const { return batch_size_; }

private:
    TrtRunner runner_;
    CUfunction preprocess_kernel_ = nullptr;
    CUdeviceptr d_boxes_ = 0;
    int batch_size_ = 0;
};

} // namespace ocr
