// Run on an NVIDIA host:
// nvcc -std=c++17 tests/cuda/test_v210_unpack.cu -o <path>/test_v210_unpack
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../../src/nodes/hwaccel/v210_unpack.cu"

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct DeviceBytes {
    uint8_t* data = nullptr;
    explicit DeviceBytes(size_t size) { check(cudaMalloc(&data, size)); }
    ~DeviceBytes() { cudaFree(data); }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

static uint16_t sample(int component, int x, int row) {
    // Traverses all 1024 codes and gives each component a distinct pattern.
    return (component * 341 + x * 31 + row * 17) & 1023;
}

static void run(int width, int height, int padding, bool semiplanar) {
    const int source_pitch = ((width * 2 + 2) / 3) * 4 + padding;
    const int y_pitch = width * 2 + 32;
    const int c_pitch = (semiplanar ? width * 2 : width) + 16;
    std::vector<uint8_t> source(size_t(source_pitch) * height, 0xa5);
    std::vector<uint8_t> expected_y(size_t(y_pitch) * height, 0xa5);
    std::vector<uint8_t> expected_u(size_t(c_pitch) * height, 0xa5);
    std::vector<uint8_t> expected_v(size_t(c_pitch) * height, 0xa5);
    auto put = [](std::vector<uint8_t>& dst, size_t offset, uint16_t value) {
        std::memcpy(dst.data() + offset, &value, sizeof(value));
    };
    for (int row = 0; row < height; ++row) {
        std::vector<uint16_t> samples;
        for (int pair = 0; pair < width / 2; ++pair) {
            const auto u = sample(1, pair, row), v = sample(2, pair, row);
            const auto y0 = sample(0, pair * 2, row), y1 = sample(0, pair * 2 + 1, row);
            samples.insert(samples.end(), {u, y0, v, y1});
            const int shift = semiplanar ? 6 : 0;
            put(expected_y, size_t(row) * y_pitch + pair * 4, y0 << shift);
            put(expected_y, size_t(row) * y_pitch + pair * 4 + 2, y1 << shift);
            put(expected_u, size_t(row) * c_pitch + pair * (semiplanar ? 4 : 2), u << shift);
            put(semiplanar ? expected_u : expected_v,
                size_t(row) * c_pitch + pair * (semiplanar ? 4 : 2) + (semiplanar ? 2 : 0), v << shift);
        }
        for (size_t i = 0; i < samples.size(); i += 3) {
            uint32_t word = 0xc0000000; // The unused top two bits must be ignored.
            for (size_t j = 0; j < 3 && i + j < samples.size(); ++j)
                word |= uint32_t(samples[i + j]) << (j * 10);
            std::memcpy(source.data() + size_t(row) * source_pitch + (i / 3) * 4, &word, 4);
        }
    }
    DeviceBytes packed(source.size()), y(expected_y.size()), u(expected_u.size()), v(expected_v.size());
    check(cudaMemcpy(packed.data, source.data(), source.size(), cudaMemcpyHostToDevice));
    check(cudaMemset(y.data, 0xa5, expected_y.size()));
    check(cudaMemset(u.data, 0xa5, expected_u.size()));
    check(cudaMemset(v.data, 0xa5, expected_v.size()));
    unpack_v210<<<dim3((width / 2 + 31) / 32, (height + 7) / 8), dim3(32, 8)>>>(
        packed.data, source_pitch, width, height, y.data, y_pitch, u.data, c_pitch,
        semiplanar ? nullptr : v.data, c_pitch, semiplanar);
    check(cudaGetLastError());
    auto compare = [](const DeviceBytes& device, const std::vector<uint8_t>& expected) {
        std::vector<uint8_t> actual(expected.size());
        check(cudaMemcpy(actual.data(), device.data, actual.size(), cudaMemcpyDeviceToHost));
        if (actual != expected) throw std::runtime_error("v210 pixels or output padding differ");
    };
    compare(y, expected_y);
    compare(u, expected_u);
    compare(v, expected_v);
}

int main() {
    try {
        for (bool semiplanar : {false, true}) {
            for (int width : {2, 4, 6, 46, 48, 50, 62, 64, 66, 1920})
                for (int padding : {0, 4, 128}) run(width, 9, padding, semiplanar);
            run(1920, 1080, 0, semiplanar);
        }
        std::cout << "PASS: 62 v210 CUDA cases, both layouts, 10-bit codes, tails and pitches\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
