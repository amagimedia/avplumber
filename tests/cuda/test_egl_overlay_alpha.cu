// Exercise the production kernels: alpha lanes, opaque compatibility, clear
// pixels, offset rectangles, bilinear alpha, and pitched-output boundaries.
// nvcc -std=c++17 tests/cuda/test_egl_overlay_alpha.cu -o /tmp/test_egl_overlay_alpha
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../../src/nodes/hwaccel/egl_image_cuda_overlay.cu"

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Fixture {
    cudaArray_t array = nullptr;
    cudaTextureObject_t texture = 0;
    unsigned char *output = nullptr;
    static constexpr int width = 6, height = 5, pitch = 32;
    Fixture() {
        check(cudaMalloc(&output, pitch * height));
        auto format = cudaCreateChannelDesc<uchar4>();
        check(cudaMallocArray(&array, &format, 2, 2));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;
        cudaTextureDesc sampling{};
        sampling.normalizedCoords = 1;
        sampling.filterMode = cudaFilterModeLinear;
        sampling.addressMode[0] = sampling.addressMode[1] = cudaAddressModeClamp;
        sampling.readMode = cudaReadModeNormalizedFloat;
        check(cudaCreateTextureObject(&texture, &resource, &sampling, nullptr));
    }
    ~Fixture() {
        cudaDestroyTextureObject(texture);
        cudaFreeArray(array);
        cudaFree(output);
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

void runCase(Fixture &f, std::array<int, 4> lanes, bool alpha) {
    const std::array<unsigned char, 4> alpha_values{0, 64, 128, 255};
    std::array<std::array<unsigned char, 4>, 4> input{};
    for (int i = 0; i < 4; ++i) {
        input[i][lanes[0]] = 32;
        input[i][lanes[1]] = 96;
        input[i][lanes[2]] = 160;
        if (lanes[3] >= 0) input[i][lanes[3]] = alpha_values[i];
    }
    check(cudaMemcpy2DToArray(f.array, 0, 0, input.data(), 8, 8, 2, cudaMemcpyHostToDevice));
    check(cudaMemset(f.output, 0xa5, f.pitch * f.height));
    clear_rgb0<<<1, dim3(8, 8)>>>(f.output, f.pitch, f.width, f.height, alpha ? 0 : 255);
    check(cudaGetLastError());
    // A 3x3 destination samples the 2x2 source's corners and bilinear midpoint.
    composite_rgba_texture<<<1, dim3(8, 8)>>>(f.texture, f.output, f.pitch,
        1, 1, 3, 3, lanes[0], lanes[1], lanes[2], lanes[3], alpha);
    check(cudaGetLastError());
    std::vector<unsigned char> result(f.pitch * f.height);
    check(cudaMemcpy(result.data(), f.output, result.size(), cudaMemcpyDeviceToHost));
    const std::array<int, 9> sampled_alpha{0, 32, 64, 64, 112, 160, 128, 192, 255};
    for (int y = 0; y < f.height; ++y) {
        for (int x = 0; x < f.pitch; ++x) {
            int expected = 0xa5;
            if (x < f.width * 4) {
                const int px = x / 4, channel = x % 4;
                const bool covered = px >= 1 && px <= 3 && y >= 1 && y <= 3;
                if (channel == 3) {
                    expected = covered ? (alpha && lanes[3] >= 0 ?
                        sampled_alpha[(y - 1) * 3 + px - 1] : 255) : (alpha ? 0 : 255);
                } else {
                    expected = covered ? 32 + channel * 64 : 0;
                }
            }
            if (result[y * f.pitch + x] != expected)
                throw std::runtime_error("overlay alpha, RGB or pitch mismatch at " +
                    std::to_string(x) + "," + std::to_string(y));
        }
    }
}

int main() {
    try {
        Fixture f;
        // RGB/BGR, ARGB/ABGR, RGBA/BGRA CUDA EGL lane layouts.
        const std::array<std::array<int, 4>, 6> layouts{{
            {2, 1, 0, -1}, {0, 1, 2, -1}, {2, 1, 0, 3},
            {0, 1, 2, 3}, {3, 2, 1, 0}, {1, 2, 3, 0}}};
        for (auto lanes : layouts)
            for (bool alpha : {false, true}) runCase(f, lanes, alpha);
        std::cout << "PASS: 12 EGL overlay alpha/opaque kernel cases\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
