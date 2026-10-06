#pragma once
#include "cuda_array_textures.hpp"

namespace avp::mixer {

class RectArrayTextures : public avp::cuda::ArrayTextures<2, checkCu> {};

} // namespace avp::mixer
