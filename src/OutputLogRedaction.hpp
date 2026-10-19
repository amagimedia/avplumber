#pragma once
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
extern "C" {
#include <libavutil/log.h>
}

// FFmpeg's FIFO worker logs its URL independently of the Output node. Install
// this only for explicitly redacted destinations, preserving the default sink.
class OutputLogRedaction {
    inline static std::mutex mutex_;
    inline static std::vector<std::string> secrets_;
    inline static std::once_flag installed_;
    static void forward(void* ptr, int level, const char* format, ...) {
        va_list args;
        va_start(args, format);
        av_log_default_callback(ptr, level, format, args);
        va_end(args);
    }
    static void callback(void* ptr, int level, const char* format, va_list args) {
        va_list copy;
        va_copy(copy, args);
        const int length = vsnprintf(nullptr, 0, format, copy);
        va_end(copy);
        if (length < 0) return;
        std::vector<char> buffer(static_cast<size_t>(length) + 1);
        vsnprintf(buffer.data(), buffer.size(), format, args);
        std::string message(buffer.data(), static_cast<size_t>(length));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& secret: secrets_) {
                size_t at = 0;
                while ((at = message.find(secret, at)) != std::string::npos) {
                    message.replace(at, secret.size(), "[redacted]");
                    at += 10;
                }
            }
        }
        forward(ptr, level, "%s", message.c_str());
    }
public:
    static void add(const std::string& url) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            secrets_.push_back(url);
            const auto scheme = url.find("://");
            const auto path = scheme == std::string::npos ? std::string::npos : url.find('/', scheme + 3);
            if (path != std::string::npos && path + 1 < url.size()) {
                secrets_.push_back(url.substr(path));
                const auto key = url.rfind('/');
                if (key + 1 < url.size()) secrets_.push_back(url.substr(key + 1));
            }
            std::sort(secrets_.begin(), secrets_.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
        }
        std::call_once(installed_, [] { av_log_set_callback(callback); });
    }
};
