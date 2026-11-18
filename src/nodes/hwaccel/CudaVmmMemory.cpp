#include "CudaVmmMemory.hpp"

#include <cuda.h>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace avp::vmm {
namespace {

void check(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": CUDA error " + std::to_string(result));
}

class Driver {
    void *library_ = dlopen("libcuda.so.1", RTLD_LOCAL | RTLD_LAZY);

    template<typename T> T load(const char *symbol) const {
        return library_ ? reinterpret_cast<T>(dlsym(library_, symbol)) : nullptr;
    }

public:
    decltype(&cuCtxPushCurrent) push = load<decltype(push)>("cuCtxPushCurrent_v2");
    decltype(&cuCtxPopCurrent) pop = load<decltype(pop)>("cuCtxPopCurrent_v2");
    decltype(&cuCtxGetDevice) get_device = load<decltype(get_device)>("cuCtxGetDevice");
    decltype(&cuDeviceGetAttribute) get_attribute = load<decltype(get_attribute)>("cuDeviceGetAttribute");
    decltype(&cuDeviceGetUuid) uuid = load<decltype(uuid)>("cuDeviceGetUuid_v2");
    decltype(&cuMemGetAllocationGranularity) granularity = load<decltype(granularity)>("cuMemGetAllocationGranularity");
    decltype(&cuMemGetAllocationPropertiesFromHandle) properties =
        load<decltype(properties)>("cuMemGetAllocationPropertiesFromHandle");
    decltype(&cuMemCreate) create = load<decltype(create)>("cuMemCreate");
    decltype(&cuMemRelease) release = load<decltype(release)>("cuMemRelease");
    decltype(&cuMemAddressReserve) reserve = load<decltype(reserve)>("cuMemAddressReserve");
    decltype(&cuMemAddressFree) free_address = load<decltype(free_address)>("cuMemAddressFree");
    decltype(&cuMemMap) map = load<decltype(map)>("cuMemMap");
    decltype(&cuMemUnmap) unmap = load<decltype(unmap)>("cuMemUnmap");
    decltype(&cuMemSetAccess) set_access = load<decltype(set_access)>("cuMemSetAccess");
    decltype(&cuMemExportToShareableHandle) export_fd = load<decltype(export_fd)>("cuMemExportToShareableHandle");
    decltype(&cuMemImportFromShareableHandle) import_fd = load<decltype(import_fd)>("cuMemImportFromShareableHandle");

    Driver() {
        if (!uuid) uuid = load<decltype(uuid)>("cuDeviceGetUuid");
        if (!push || !pop || !get_device || !get_attribute || !uuid || !granularity || !properties || !create ||
            !release || !reserve || !free_address || !map || !unmap || !set_access || !export_fd || !import_fd) {
            if (library_) dlclose(library_);
            throw std::runtime_error("CUDA VMM Driver API is unavailable");
        }
    }

    ~Driver() { dlclose(library_); }
};

Driver &driver() {
    static Driver instance;
    return instance;
}

class Context {
    CUcontext context_;

public:
    explicit Context(void *context) : context_(static_cast<CUcontext>(context)) {
        check(driver().push(context_), "cuCtxPushCurrent");
    }
    ~Context() {
        CUcontext previous = nullptr;
        driver().pop(&previous);
    }
};

CUdevice selectedDevice() {
    CUdevice device = 0;
    check(driver().get_device(&device), "cuCtxGetDevice");
    int vmm = 0;
    check(driver().get_attribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device),
          "query CUDA VMM support");
    int posix_fd = 0;
    check(driver().get_attribute(&posix_fd, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                                 device), "query CUDA POSIX FD support");
    if (!vmm || !posix_fd)
        throw std::runtime_error("selected CUDA device does not support VMM with POSIX file descriptors");
    return device;
}

}

DeviceUuid deviceUuid(void *context) {
    Context guard(context);
    CUuuid uuid{};
    check(driver().uuid(&uuid, selectedDevice()), "cuDeviceGetUuid");
    DeviceUuid bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(uuid.bytes[i]);
    return bytes;
}

struct Memory::Mapping {
    void *context;
    CUdeviceptr address = 0;
    size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool allocated = false;
    bool mapped = false;

    explicit Mapping(void *context) : context(context) {}

    ~Mapping() noexcept {
        try {
            Context guard(context);
            if (mapped) driver().unmap(address, size);
            if (address) driver().free_address(address, size);
            if (allocated) driver().release(handle);
        } catch (...) {
            // A failed device/context has already made this mapping unusable.
        }
    }

    void map(CUdevice device, CUmemAccess_flags flags) {
        check(driver().reserve(&address, size, 0, 0, 0), "cuMemAddressReserve");
        check(driver().map(address, size, 0, handle, 0), "cuMemMap");
        mapped = true;
        CUmemAccessDesc access{{CU_MEM_LOCATION_TYPE_DEVICE, device}, flags};
        check(driver().set_access(address, size, &access, 1), "cuMemSetAccess");
    }
};

Memory::Memory(std::unique_ptr<Mapping> mapping) : mapping_(std::move(mapping)) {}
Memory::~Memory() = default;

std::shared_ptr<Memory> Memory::allocate(void *context, size_t bytes) {
    Context guard(context);
    CUdevice device = selectedDevice();
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t granularity = 0;
    check(driver().granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "cuMemGetAllocationGranularity");
    if (!bytes || !granularity || bytes > SIZE_MAX - (granularity - 1))
        throw std::runtime_error("invalid CUDA VMM allocation size");
    auto mapping = std::make_unique<Mapping>(context);
    mapping->size = ((bytes + granularity - 1) / granularity) * granularity;
    check(driver().create(&mapping->handle, mapping->size, &prop, 0), "cuMemCreate");
    mapping->allocated = true;
    mapping->map(device, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    return std::shared_ptr<Memory>(new Memory(std::move(mapping)));
}

std::shared_ptr<Memory> Memory::import(void *context, int fd, size_t bytes) {
    Context guard(context);
    CUdevice device = selectedDevice();
    auto mapping = std::make_unique<Mapping>(context);
    mapping->size = bytes;
    check(driver().import_fd(&mapping->handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
                             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
          "cuMemImportFromShareableHandle");
    mapping->allocated = true;
    CUmemAllocationProp properties{};
    check(driver().properties(&properties, mapping->handle), "cuMemGetAllocationPropertiesFromHandle");
    if (properties.type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        properties.location.type != CU_MEM_LOCATION_TYPE_DEVICE || properties.location.id != device)
        throw std::runtime_error("CUDA VMM FD does not belong to the selected device");
    size_t granularity = 0;
    check(driver().granularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "cuMemGetAllocationGranularity");
    if (!granularity || !bytes || bytes % granularity)
        throw std::runtime_error("invalid imported CUDA VMM allocation size");
    mapping->map(device, CU_MEM_ACCESS_FLAGS_PROT_READ);
    return std::shared_ptr<Memory>(new Memory(std::move(mapping)));
}

int Memory::exportFd() const {
    int fd = -1;
    Context guard(mapping_->context);
    check(driver().export_fd(&fd, mapping_->handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
          "cuMemExportToShareableHandle");
    return fd;
}

size_t Memory::size() const { return mapping_->size; }
uint8_t *Memory::data() const { return reinterpret_cast<uint8_t *>(mapping_->address); }

}
