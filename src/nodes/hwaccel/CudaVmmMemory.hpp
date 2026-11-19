#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace avp::vmm {

using DeviceUuid = std::array<uint8_t, 16>;

// The implementation uses the installed CUDA SDK. Keeping its types here
// opaque avoids mixing it with AVPlumber's older bundled CUDA declarations.
DeviceUuid deviceUuid(void *context);

class Memory {
    struct Mapping;
    std::unique_ptr<Mapping> mapping_;
    explicit Memory(std::unique_ptr<Mapping> mapping);

public:
    ~Memory();
    Memory(const Memory &) = delete;
    Memory &operator=(const Memory &) = delete;

    static std::shared_ptr<Memory> allocate(void *context, size_t bytes);
    static std::shared_ptr<Memory> import(void *context, int fd, size_t bytes);
    int exportFd() const;
    size_t size() const;
    uint8_t *data() const;
};

}
