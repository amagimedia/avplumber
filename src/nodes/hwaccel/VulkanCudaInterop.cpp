#include "VulkanCudaInterop.hpp"

#include "CudaVmmMemory.hpp"

#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#if HAVE_VULKAN && HAVE_VULKAN_CUDA
#include <vulkan/vulkan.h>
// The reverse direction imports an opaque fd and maps a buffer. Use the bundled declarations the
// rest of the tree compiles against: the SDK's cuda.h defines the same enums differently and the
// two in one binary trip the ODR check.
#include "../../../deps/cuda_loader/cuda.h"
#endif

namespace avp::vkinterop {

#if HAVE_VULKAN && HAVE_VULKAN_CUDA

namespace {

const char *resultName(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "VkResult";
    }
}

void check(VkResult r, const char *what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + ": " + resultName(r) + " (" + std::to_string(int(r)) + ")");
}

bool deviceExtension(VkPhysicalDevice physical, const char *name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, props.data());
    for (const auto &p : props)
        if (!std::strcmp(p.extensionName, name)) return true;
    return false;
}

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetMemoryFdKHR getMemoryFd = nullptr;
    uint32_t queueFamily = 0;
    std::string name;
    uint8_t uuid[16] = {};
};

/// Picks the Vulkan device that belongs to `wanted` (a CUDA device UUID), or the first discrete
/// GPU when there is nothing to match against. Vulkan and CUDA report the same GPU UUID.
Device makeDevice(const uint8_t *wanted) {
    Device d;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "avplumber";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    check(vkCreateInstance(&ici, nullptr, &d.instance), "vkCreateInstance");

    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(d.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physical(count);
    check(vkEnumeratePhysicalDevices(d.instance, &count, physical.data()), "vkEnumeratePhysicalDevices");

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (VkPhysicalDevice p : physical) {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &ids;
        vkGetPhysicalDeviceProperties2(p, &props);
        if (wanted && !std::memcmp(ids.deviceUUID, wanted, 16)) {
            d.physical = p;
            break;
        }
        if (fallback == VK_NULL_HANDLE && props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            fallback = p;
    }
    if (d.physical == VK_NULL_HANDLE) {
        if (wanted) {
            vkDestroyInstance(d.instance, nullptr);
            throw std::runtime_error("no Vulkan device matches the CUDA device UUID");
        }
        d.physical = fallback;
    }
    if (d.physical == VK_NULL_HANDLE) {
        vkDestroyInstance(d.instance, nullptr);
        throw std::runtime_error("no Vulkan device");
    }
    if (!deviceExtension(d.physical, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
        vkDestroyInstance(d.instance, nullptr);
        throw std::runtime_error("Vulkan device has no VK_KHR_external_memory_fd");
    }

    VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &ids;
    vkGetPhysicalDeviceProperties2(d.physical, &props);
    std::memcpy(d.uuid, ids.deviceUUID, 16);
    d.name = props.properties.deviceName;
    vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory);

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, nullptr);
    std::vector<VkQueueFamilyProperties> familyProps(families);
    vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, familyProps.data());
    d.queueFamily = families;
    for (uint32_t i = 0; i < families; ++i)
        if (familyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            d.queueFamily = i;
            break;
        }
    if (d.queueFamily == families) {
        vkDestroyInstance(d.instance, nullptr);
        throw std::runtime_error("Vulkan device has no compute queue");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = d.queueFamily;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const char *extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue;
    create.enabledExtensionCount = 1;
    create.ppEnabledExtensionNames = extensions;
    check(vkCreateDevice(d.physical, &create, nullptr, &d.device), "vkCreateDevice");
    d.getMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(d.device, "vkGetMemoryFdKHR");
    if (!d.getMemoryFd) {
        vkDestroyDevice(d.device, nullptr);
        vkDestroyInstance(d.instance, nullptr);
        throw std::runtime_error("Vulkan device has no vkGetMemoryFdKHR");
    }
    return d;
}

Device &device(const uint8_t *wantedUuid) {
    static std::mutex mutex;
    static Device shared;
    static bool created = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (!created) {
        shared = makeDevice(wantedUuid);
        created = true;
    } else if (wantedUuid && std::memcmp(shared.uuid, wantedUuid, 16)) {
        throw std::runtime_error("Vulkan interop is already bound to a different CUDA device");
    }
    return shared;
}

/// Whether the driver offers external memory for this format with the usage the shaders need.
bool formatShareable(const Device &d, VkFormat format) {
    if (!format) return false;
    VkPhysicalDeviceExternalImageFormatInfo external{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    external.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkPhysicalDeviceImageFormatInfo2 info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    info.pNext = &external;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_LINEAR;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkExternalImageFormatProperties externalProps{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 props{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    props.pNext = &externalProps;
    if (vkGetPhysicalDeviceImageFormatProperties2(d.physical, &info, &props) != VK_SUCCESS) return false;
    // One allocation per object, so a driver that wants a dedicated allocation is fine; what
    // matters is that memory can be imported at all.
    return (externalProps.externalMemoryProperties.externalMemoryFeatures &
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
}

uint32_t memoryTypeFor(const Device &d, VkMemoryRequirements req) {
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (d.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return i;
    for (uint32_t i = 0; i < d.memory.memoryTypeCount; ++i)
        if (req.memoryTypeBits & (1u << i)) return i;
    throw std::runtime_error("no Vulkan memory type for the imported allocation");
}

/// CUDA-owned VMM memory, imported into Vulkan as one VkDeviceMemory.
struct VulkanAllocation {
    Device *d = nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::shared_ptr<avp::vmm::Memory> cuda;
    int fd = -1;
    bool ownedByVulkan = false;

    VulkanAllocation(Device &device, void *context, VkMemoryRequirements req) : d(&device) {
        cuda = avp::vmm::Memory::allocate(context, req.size);
        fd = cuda->exportFd();
        VkImportMemoryFdInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
        import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        import.fd = fd;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.pNext = &import;
        alloc.allocationSize = cuda->size();
        alloc.memoryTypeIndex = memoryTypeFor(device, req);
        check(vkAllocateMemory(d->device, &alloc, nullptr, &memory), "vkAllocateMemory(import fd)");
        ownedByVulkan = true;   // a successful import takes the fd's ownership
    }
    ~VulkanAllocation() {
        if (d && d->device && memory) vkFreeMemory(d->device, memory, nullptr);
        // The CUDA allocation goes last: Vulkan stops referencing the fd above.
        if (fd >= 0 && !ownedByVulkan) ::close(fd);
    }
    VulkanAllocation(const VulkanAllocation &) = delete;
    VulkanAllocation &operator=(const VulkanAllocation &) = delete;
};

/// The CUDA external-memory entry points the reverse direction needs. avp::vmm loads the VMM ones;
/// importing an opaque fd is a different path, so load its own three symbols here.
struct CudaDriver {
    decltype(&cuImportExternalMemory) import = nullptr;
    decltype(&cuExternalMemoryGetMappedBuffer) map = nullptr;
    decltype(&cuDestroyExternalMemory) destroy = nullptr;
    decltype(&cuCtxPushCurrent) push = nullptr;
    decltype(&cuCtxPopCurrent) pop = nullptr;

    CudaDriver() {
        void *lib = dlopen("libcuda.so.1", RTLD_LOCAL | RTLD_LAZY);
        if (!lib) return;
        import = (decltype(import))dlsym(lib, "cuImportExternalMemory");
        map = (decltype(map))dlsym(lib, "cuExternalMemoryGetMappedBuffer");
        destroy = (decltype(destroy))dlsym(lib, "cuDestroyExternalMemory");
        push = (decltype(push))dlsym(lib, "cuCtxPushCurrent_v2");
        pop = (decltype(pop))dlsym(lib, "cuCtxPopCurrent_v2");
        if (!import || !map || !destroy || !push || !pop) {
            import = nullptr;
            dlclose(lib);
        }
    }
};

CudaDriver &cudaDriver() {
    static CudaDriver driver;
    if (!driver.import) throw std::runtime_error("CUDA external memory API is unavailable");
    return driver;
}

} // namespace

struct SharedImage::Impl {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format{};
    int width = 0;
    int height = 0;
    size_t rowPitch = 0;
    VulkanAllocation allocation;

    Impl(Device &d, void *context, VkFormat fmt, int w, int h, VkMemoryRequirements req)
        : format(fmt), width(w), height(h), allocation(d, context, req) {}
    ~Impl() {
        if (image) vkDestroyImage(allocation.d->device, image, nullptr);
    }
};

struct SharedBuffer::Impl {
    VkBuffer buffer = VK_NULL_HANDLE;
    size_t bytes = 0;
    VulkanAllocation allocation;

    Impl(Device &d, void *context, size_t size, VkMemoryRequirements req)
        : bytes(size), allocation(d, context, req) {}
    ~Impl() {
        if (buffer) vkDestroyBuffer(allocation.d->device, buffer, nullptr);
    }
};

struct CudaView::Impl {
    CUexternalMemory external = nullptr;
    uint8_t *data = nullptr;
    size_t bytes = 0;

    ~Impl() {
        if (external) {
            CudaDriver &driver = cudaDriver();
            CUcontext dummy;
            driver.pop(&dummy);
            driver.destroy(external);
        }
    }
};

SharedImage::SharedImage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SharedImage::~SharedImage() = default;

std::shared_ptr<SharedImage> SharedImage::create(void *context, uint32_t vkFormat, int width, int height) {
    if (!context) throw std::runtime_error("SharedImage::create needs a CUDA context");
    if (width <= 0 || height <= 0)
        throw std::runtime_error("SharedImage::create needs positive dimensions");
    const avp::vmm::DeviceUuid uuid = avp::vmm::deviceUuid(context);
    Device &d = device(uuid.data());
    if (!formatShareable(d, VkFormat(vkFormat)))
        throw std::runtime_error("VkFormat " + std::to_string(vkFormat) + " (" + d.name +
                                 ") has no external memory support with storage usage: multi-plane "
                                 "YUV formats cannot be shared with CUDA, so this has to be staged "
                                 "through host memory");

    VkExternalMemoryImageCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.pNext = &external;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VkFormat(vkFormat);
    imageInfo.extent = {uint32_t(width), uint32_t(height), 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    check(vkCreateImage(d.device, &imageInfo, nullptr, &image), "vkCreateImage");
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(d.device, image, &req);

    std::unique_ptr<Impl> impl;
    try {
        impl = std::make_unique<Impl>(d, context, VkFormat(vkFormat), width, height, req);
    } catch (...) {
        vkDestroyImage(d.device, image, nullptr);
        throw;
    }
    impl->image = image;
    check(vkBindImageMemory(d.device, image, impl->allocation.memory, 0), "vkBindImageMemory");

    VkImageSubresource subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout layout{};
    vkGetImageSubresourceLayout(d.device, image, &subresource, &layout);
    impl->rowPitch = size_t(layout.rowPitch);
    if (!impl->rowPitch || impl->rowPitch * size_t(height) > impl->allocation.cuda->size())
        throw std::runtime_error("Vulkan row pitch " + std::to_string(impl->rowPitch) +
                                 " does not fit the CUDA allocation");
    return std::shared_ptr<SharedImage>(new SharedImage(std::move(impl)));
}

uint8_t *SharedImage::cudaData() const { return impl_->allocation.cuda->data(); }
void *SharedImage::vkImage() const { return impl_->image; }
void *SharedImage::vkDevice() const { return impl_->allocation.d->device; }
uint32_t SharedImage::queueFamily() const { return impl_->allocation.d->queueFamily; }
int SharedImage::fd() const { return impl_->allocation.fd; }
size_t SharedImage::size() const { return impl_->allocation.cuda->size(); }
size_t SharedImage::rowPitch() const { return impl_->rowPitch; }
uint32_t SharedImage::vkFormat() const { return impl_->format; }
int SharedImage::width() const { return impl_->width; }
int SharedImage::height() const { return impl_->height; }

SharedBuffer::SharedBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SharedBuffer::~SharedBuffer() = default;

std::shared_ptr<SharedBuffer> SharedBuffer::create(void *context, size_t bytes) {
    if (!context) throw std::runtime_error("SharedBuffer::create needs a CUDA context");
    if (!bytes) throw std::runtime_error("SharedBuffer::create needs a positive size");
    const avp::vmm::DeviceUuid uuid = avp::vmm::deviceUuid(context);
    Device &d = device(uuid.data());

    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext = &external;
    bufferInfo.size = bytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    check(vkCreateBuffer(d.device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(d.device, buffer, &req);

    std::unique_ptr<Impl> impl;
    try {
        impl = std::make_unique<Impl>(d, context, bytes, req);
    } catch (...) {
        vkDestroyBuffer(d.device, buffer, nullptr);
        throw;
    }
    impl->buffer = buffer;
    check(vkBindBufferMemory(d.device, buffer, impl->allocation.memory, 0), "vkBindBufferMemory");
    return std::shared_ptr<SharedBuffer>(new SharedBuffer(std::move(impl)));
}

uint8_t *SharedBuffer::cudaData() const { return impl_->allocation.cuda->data(); }
void *SharedBuffer::vkBuffer() const { return impl_->buffer; }
void *SharedBuffer::vkDevice() const { return impl_->allocation.d->device; }
uint32_t SharedBuffer::queueFamily() const { return impl_->allocation.d->queueFamily; }
size_t SharedBuffer::size() const { return impl_->bytes; }
int SharedBuffer::fd() const { return impl_->allocation.fd; }

CudaView::CudaView(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaView::~CudaView() = default;

std::shared_ptr<CudaView> CudaView::ofVulkanMemory(void *context, void *vkMemory, size_t bytes) {
    if (!context) throw std::runtime_error("CudaView::ofVulkanMemory needs a CUDA context");
    if (!vkMemory) throw std::runtime_error("CudaView::ofVulkanMemory needs a VkDeviceMemory");
    if (!bytes) throw std::runtime_error("CudaView::ofVulkanMemory needs a positive size");
    Device &d = device(nullptr);
    CudaDriver &driver = cudaDriver();

    VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    info.memory = (VkDeviceMemory)vkMemory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    check(d.getMemoryFd(d.device, &info, &fd), "vkGetMemoryFdKHR");

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
    handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    handle.handle.fd = fd;
    handle.size = bytes;

    auto impl = std::make_unique<Impl>();
    impl->bytes = bytes;
    driver.push((CUcontext)context);
    bool pushed = true;
    try {
        if (driver.import(&impl->external, &handle) != CUDA_SUCCESS)
            throw std::runtime_error("cuImportExternalMemory rejected the Vulkan opaque fd");
        pushed = false;   // Impl's destructor pops
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer{};
        buffer.offset = 0;
        buffer.size = bytes;
        CUdeviceptr mapped = 0;
        if (driver.map(&mapped, impl->external, &buffer) != CUDA_SUCCESS)
            throw std::runtime_error("cuExternalMemoryGetMappedBuffer failed");
        impl->data = reinterpret_cast<uint8_t *>(mapped);
    } catch (...) {
        if (pushed) {
            CUcontext dummy;
            driver.pop(&dummy);
        }
        throw;
    }
    return std::shared_ptr<CudaView>(new CudaView(std::move(impl)));
}

uint8_t *CudaView::cudaData() const { return impl_->data; }
size_t CudaView::size() const { return impl_->bytes; }

bool supportsSharedFormat(uint32_t vkFormat) {
    try {
        return formatShareable(device(nullptr), VkFormat(vkFormat));
    } catch (...) {
        return false;
    }
}

bool available() {
    try {
        return device(nullptr).device != VK_NULL_HANDLE;
    } catch (...) {
        return false;
    }
}

std::string deviceName() {
    try {
        return device(nullptr).name;
    } catch (...) {
        return {};
    }
}

#else

// Built without Vulkan or without the CUDA VMM allocator: the API exists so callers link, and
// reports that sharing is unavailable so they fall back to a host copy.
struct SharedImage::Impl {};
struct SharedBuffer::Impl {};
struct CudaView::Impl {};

SharedImage::SharedImage(std::unique_ptr<Impl>) {}
SharedImage::~SharedImage() = default;
SharedBuffer::SharedBuffer(std::unique_ptr<Impl>) {}
SharedBuffer::~SharedBuffer() = default;
CudaView::CudaView(std::unique_ptr<Impl>) {}
CudaView::~CudaView() = default;

std::shared_ptr<SharedImage> SharedImage::create(void *, uint32_t, int, int) {
    throw std::runtime_error("CUDA/Vulkan zero-copy is not built in (needs HAVE_VULKAN=1 and "
                             "HAVE_CUDA_VMM_IPC=1)");
}

std::shared_ptr<SharedBuffer> SharedBuffer::create(void *, size_t) {
    throw std::runtime_error("CUDA/Vulkan zero-copy is not built in (needs HAVE_VULKAN=1 and "
                             "HAVE_CUDA_VMM_IPC=1)");
}

std::shared_ptr<CudaView> CudaView::ofVulkanMemory(void *, void *, size_t) {
    throw std::runtime_error("CUDA/Vulkan zero-copy is not built in (needs HAVE_VULKAN=1 and "
                             "HAVE_CUDA_VMM_IPC=1)");
}

uint8_t *SharedImage::cudaData() const { return nullptr; }
void *SharedImage::vkImage() const { return nullptr; }
void *SharedImage::vkDevice() const { return nullptr; }
uint32_t SharedImage::queueFamily() const { return 0; }
int SharedImage::fd() const { return -1; }
size_t SharedImage::size() const { return 0; }
size_t SharedImage::rowPitch() const { return 0; }
uint32_t SharedImage::vkFormat() const { return 0; }
int SharedImage::width() const { return 0; }
int SharedImage::height() const { return 0; }

uint8_t *SharedBuffer::cudaData() const { return nullptr; }
void *SharedBuffer::vkBuffer() const { return nullptr; }
void *SharedBuffer::vkDevice() const { return nullptr; }
uint32_t SharedBuffer::queueFamily() const { return 0; }
size_t SharedBuffer::size() const { return 0; }
int SharedBuffer::fd() const { return -1; }

uint8_t *CudaView::cudaData() const { return nullptr; }
size_t CudaView::size() const { return 0; }

bool supportsSharedFormat(uint32_t) { return false; }
bool available() { return false; }
std::string deviceName() { return {}; }

#endif

}
