#pragma once
#include "util.hpp"
#include <cstdint>
#include <limits>

inline uint32_t parseBitmask(const Parameters& value) {
    if (value.is_string()) {
        uint32_t mask = 0;
        auto s = value.get<std::string>();
        if (s.size() > std::numeric_limits<uint32_t>::digits)
            throw Error("bitmask string exceeds mask width");
        for (size_t i = 0; i < s.size(); i++)
            if (s[i] == '1') mask |= (1u << i);
        return mask;
    }
    return value.get<uint32_t>();
}
