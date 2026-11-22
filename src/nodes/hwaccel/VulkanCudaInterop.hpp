#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Zero-copy sharing between CUDA device memory and Vulkan compute.
//
// Both directions work over VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD /
// CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: CUDA VMM memory imported by Vulkan (SharedImage,
// SharedBuffer), and Vulkan-exported memory mapped into CUDA (CudaView). The dma-buf handle type
// is not the one to reach for here - CUDA imports CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD only on
// Tegra/Thor, and returns an unnamed error for a Vulkan dma-buf on a discrete GPU.
//
// What stays impossible is multi-plane YUV: the NVIDIA vulkan driver exposes no external memory
// for G8_B8_R8_3PLANE_444, G8_B8R8_2PLANE_420 and friends - everything a Vulkan yuv444p/yuv420p
// frame is made of - in either tiling, so a Vulkan codec that consumes those formats (FFmpeg's
// ffv1_vulkan, for one) can only be fed through a copy. supportsSharedFormat() answers that per
// format, so a caller stages those and goes zero-copy everywhere else.
namespace avp::vkinterop {

/// One CUDA-owned buffer that Vulkan binds as a VkImage. Rows start rowPitch() bytes apart on both
/// sides, so a CUDA kernel that writes the image has to honour it.
class SharedImage {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit SharedImage(std::unique_ptr<Impl> impl);

public:
    /// context is a CUcontext (the one the frames around it live in); vkFormat is a single-plane
    /// VkFormat. Throws, naming the format and the reason, when the driver cannot share it
    /// zero-copy or the allocation fails.
    static std::shared_ptr<SharedImage> create(void *context, uint32_t vkFormat, int width, int height);
    ~SharedImage();
    SharedImage(const SharedImage &) = delete;
    SharedImage &operator=(const SharedImage &) = delete;

    /// Device pointer for CUDA writes and reads. The Vulkan image aliases exactly this memory.
    uint8_t *cudaData() const;
    /// VkImage, to bind as a storage image or transfer target in a caller's command buffer.
    void *vkImage() const;
    /// VkDevice the image, and the queue family a submit has to use.
    void *vkDevice() const;
    uint32_t queueFamily() const;
    /// The imported fd, still owned by Vulkan; exposed for diagnostics only.
    int fd() const;
    size_t size() const;
    size_t rowPitch() const;
    uint32_t vkFormat() const;
    int width() const;
    int height() const;
};

/// One CUDA-owned buffer that Vulkan binds as a VkBuffer. A buffer is linear by definition, so
/// this has none of the format restrictions SharedImage has: any size, any payload.
class SharedBuffer {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit SharedBuffer(std::unique_ptr<Impl> impl);

public:
    static std::shared_ptr<SharedBuffer> create(void *context, size_t bytes);
    ~SharedBuffer();
    SharedBuffer(const SharedBuffer &) = delete;
    SharedBuffer &operator=(const SharedBuffer &) = delete;

    uint8_t *cudaData() const;
    /// VkBuffer with storage, transfer source and transfer destination usage.
    void *vkBuffer() const;
    void *vkDevice() const;
    uint32_t queueFamily() const;
    size_t size() const;
    int fd() const;
};

/// The other direction: Vulkan owns the memory and CUDA gets a mapped buffer over it. The caller
/// keeps the VkDeviceMemory alive for as long as the view; this holds only the import.
class CudaView {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit CudaView(std::unique_ptr<Impl> impl);

public:
    /// vkMemory has to belong to this shim's device and be exportable with
    /// VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT. bytes is mapped from the start of the
    /// allocation, which is how a linear VkImage's texels are reached too - with the image's own
    /// row pitch, not a packed stride.
    static std::shared_ptr<CudaView> ofVulkanMemory(void *context, void *vkMemory, size_t bytes);
    ~CudaView();
    CudaView(const CudaView &) = delete;
    CudaView &operator=(const CudaView &) = delete;

    uint8_t *cudaData() const;
    size_t size() const;
};

/// Whether a VkFormat can be shared with CUDA on the matched device. False means the caller must
/// copy through host memory; it is a property of the format and driver, not of this build.
bool supportsSharedFormat(uint32_t vkFormat);

/// Whether a Vulkan compute device with external memory support exists at all.
bool available();

/// Name of the matched Vulkan device, empty when !available().
std::string deviceName();

}
