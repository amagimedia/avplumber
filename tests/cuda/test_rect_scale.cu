// NVIDIA integration test of the production scaler, including UV lane isolation.
// nvcc -std=c++17 tests/cuda/test_rect_scale.cu -o /tmp/test_rect_scale
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../../src/nodes/hwaccel/cuda_rect_scale.cu"

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceBytes {
    unsigned char* data = nullptr;
    explicit DeviceBytes(size_t size) { check(cudaMalloc(&data, size)); }
    ~DeviceBytes() { cudaFree(data); }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

void checkUpscale(int lanes, int origin_x) {
    // A 2x2 linear ramp inside a larger cropped/padded source. Quarter-pixel
    // values distinguish bilinear interpolation from cubic smoothstep weights.
    constexpr std::array<int,16> expected = {
        0,16,48,64, 32,48,80,96, 96,112,144,160, 128,144,176,192
    };
    constexpr int source_pitch = 16, target_pitch = 20;
    std::vector<unsigned char> source(source_pitch*4, 0xee);
    for (int y=0; y<2; ++y)
        for (int x=0; x<2; ++x)
            for (int lane=0; lane<lanes; ++lane)
                source[(y+1)*source_pitch+(x+1)*lanes+lane] = lane ? 255-(y*128+x*64) : y*128+x*64;
    std::vector<unsigned char> target(target_pitch*8, 0xa5);
    DeviceBytes input(source.size()), output(target.size());
    check(cudaMemcpy(input.data, source.data(), source.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output.data, target.data(), target.size(), cudaMemcpyHostToDevice));
    scale_plane<<<dim3(1,1),dim3(8,8)>>>(input.data,source_pitch,1,1,2,2,
        output.data,target_pitch,origin_x,2,4,4,6,8,lanes,1,0);
    check(cudaGetLastError());
    check(cudaMemcpy(target.data(),output.data,target.size(),cudaMemcpyDeviceToHost));
    for (int y=0; y<8; ++y) {
        for (int byte=0; byte<target_pitch; ++byte) {
            const int x=byte/lanes, lane=byte%lanes, ox=x-origin_x, oy=y-2;
            const bool written=x<6 && ox>=0 && ox<4 && oy>=0 && oy<4;
            const int value=written ? (lane ? 255-expected[oy*4+ox] : expected[oy*4+ox]) : 0xa5;
            if (target[y*target_pitch+byte]!=value)
                throw std::runtime_error("bilinear pixel, crop, lane or clipping mismatch");
        }
    }
}
void checkUpscaleWord(int lanes, int origin_x, int shift) {
    // The byte ramp times four lands on genuine 10-bit codes; bilinear results
    // scale linearly, so the expected table is the byte table times four.
    constexpr std::array<int,16> expected = {
        0,16,48,64, 32,48,80,96, 96,112,144,160, 128,144,176,192
    };
    constexpr int source_pitch_e = 8, target_pitch_e = 16;
    std::vector<unsigned short> source(source_pitch_e*4, 0xeeee);
    for (int y=0; y<2; ++y)
        for (int x=0; x<2; ++x)
            for (int lane=0; lane<lanes; ++lane) {
                const int code = lane ? 1023-(y*512+x*256) : y*512+x*256;
                source[(y+1)*source_pitch_e+(x+1)*lanes+lane] = (unsigned short)(code << shift);
            }
    std::vector<unsigned short> target(target_pitch_e*8, 0xa5a5);
    DeviceBytes input(source.size()*2), output(target.size()*2);
    check(cudaMemcpy(input.data, source.data(), source.size()*2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(output.data, target.data(), target.size()*2, cudaMemcpyHostToDevice));
    scale_plane<<<dim3(1,1),dim3(8,8)>>>(input.data,source_pitch_e*2,1,1,2,2,
        output.data,target_pitch_e*2,origin_x,2,4,4,6,8,lanes,2,shift);
    check(cudaGetLastError());
    check(cudaMemcpy(target.data(),output.data,target.size()*2,cudaMemcpyDeviceToHost));
    for (int y=0; y<8; ++y) {
        for (int e=0; e<target_pitch_e; ++e) {
            const int x=e/lanes, lane=e%lanes, ox=x-origin_x, oy=y-2;
            const bool written=x<6 && ox>=0 && ox<4 && oy>=0 && oy<4;
            const int logical=lane ? 1023-4*expected[oy*4+ox] : 4*expected[oy*4+ox];
            const int value=written ? logical<<shift : 0xa5a5;
            if (target[y*target_pitch_e+e]!=value)
                throw std::runtime_error("word bilinear pixel, shift, lane or clipping mismatch");
        }
    }
}
void checkPromote(int lanes, int dst_shift) {
    // 8-bit source -> 10-bit destination, promotion multiplier 4 (16->64 style).
    // Bilinear result then times four, stored at the P210 shift; the byte table
    // is the same expected upscale.
    constexpr std::array<int,16> expected = {
        0,16,48,64, 32,48,80,96, 96,112,144,160, 128,144,176,192
    };
    constexpr int source_pitch = 8, target_pitch_e = 16;
    std::vector<unsigned char> source(source_pitch*4, 0xee);
    for (int y=0; y<2; ++y)
        for (int x=0; x<2; ++x)
            for (int lane=0; lane<lanes; ++lane)
                source[(y+1)*source_pitch+(x+1)*lanes+lane] = lane ? 255-(y*128+x*64) : y*128+x*64;
    std::vector<unsigned short> target(target_pitch_e*8, 0xa5a5);
    DeviceBytes input(source.size()), output(target.size()*2);
    check(cudaMemcpy(input.data, source.data(), source.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output.data, target.data(), target.size()*2, cudaMemcpyHostToDevice));
    convert_scale_plane<<<dim3(1,1),dim3(8,8)>>>(input.data,source_pitch,1,1,2,2,1,0,
        output.data,target_pitch_e*2,2,2,4,4,6,8,lanes,2,dst_shift,4.f);
    check(cudaGetLastError());
    check(cudaMemcpy(target.data(),output.data,target.size()*2,cudaMemcpyDeviceToHost));
    for (int y=0; y<8; ++y) {
        for (int e=0; e<target_pitch_e; ++e) {
            const int x=e/lanes, lane=e%lanes, ox=x-2, oy=y-2;
            const bool written=x<6 && ox>=0 && ox<4 && oy>=0 && oy<4;
            const int base=lane ? 255-expected[oy*4+ox] : expected[oy*4+ox];
            const int value=written ? (base*4)<<dst_shift : 0xa5a5;
            if (target[y*target_pitch_e+e]!=value)
                throw std::runtime_error("promote pixel, multiplier, shift or lane mismatch");
        }
    }
}
int main() {
    try {
        for (int lanes : {1,2}) for (int origin : {-1,1,4}) checkUpscale(lanes,origin);
        for (int lanes : {1,2}) for (int origin : {-1,1,4}) for (int shift : {0,6})
            checkUpscaleWord(lanes,origin,shift);
        for (int lanes : {1,2}) for (int shift : {0,6}) checkPromote(lanes,shift);
        std::cout << "PASS: bilinear pixels, UV lanes, word samples, storage shift, "
                     "8->10-bit promotion, crop, pitch and destination clipping\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
