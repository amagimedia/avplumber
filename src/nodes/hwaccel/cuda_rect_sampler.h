#pragma once
// How a packed 8-bit RGBA CUDA array is exposed to the compositor's *_TEX layers.
// The kernel fetches tex2D<uchar4> at integer texel centres, so the texture must
// return elements (CU_TRSF_READ_AS_INTEGER) with point filtering and unnormalized
// coordinates. Without the flag the unit promotes texels to unit-range floats and
// the uchar4 fetch yields the float's bit pattern: 255 reads as 0, 127 as 255.
// The parity test builds its textures through this same function.
// Include after the CUDA driver API declarations (cuda.h, or the dynlink mirror
// the modules use).
#if !defined(__cuda_cuda_h__)
#error "cuda_rect_sampler.h needs the CUDA driver API declarations included first"
#endif

namespace avp::mixer {

inline void rectTextureDesc(CUarray array, CUDA_RESOURCE_DESC &res, CUDA_TEXTURE_DESC &td) {
    res = CUDA_RESOURCE_DESC{};
    res.resType = CU_RESOURCE_TYPE_ARRAY;
    res.res.array.hArray = array;
    td = CUDA_TEXTURE_DESC{};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    td.filterMode = CU_TR_FILTER_MODE_POINT;
    td.flags = CU_TRSF_READ_AS_INTEGER;
}

}
