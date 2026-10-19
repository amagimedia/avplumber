#pragma once
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

// Optional bounded publication receipt. Counts only successful writePacket calls;
// bytes are encoded payload bytes, not an estimate of transport overhead.
class OutputProgress {
    std::string path_;
    bool submitted_ = false;
    uint64_t packets_ = 0, bytes_ = 0;
    double first_ = 0, updated_ = 0, published_ = 0;
public:
    void setSubmitted(bool value) { submitted_ = value; }
    void setPath(const std::string& path) { path_ = path; }
    void wrote(uint64_t bytes, double now) {
        if (path_.empty()) return;
        if (packets_ == 0) first_ = now;
        ++packets_;
        bytes_ += bytes;
        updated_ = now;
        if (packets_ == 1 || now - published_ >= 1.0) flush();
    }
    void flush() {
        if (path_.empty() || packets_ == 0) return;
        const std::string temporary = path_ + ".tmp";
        std::ofstream output(temporary, std::ios::trunc);
        output << std::setprecision(17);
        if (submitted_) {
            output << "{\"delivery_semantics\":\"submitted\",\"submitted_packets\":" << packets_
                   << ",\"submitted_bytes\":" << bytes_
                   << ",\"last_submit_at\":" << updated_
                   << ",\"first_submit_at\":" << first_ << "}\n";
        } else {
            output << "{\"written_packets\":" << packets_
                   << ",\"written_bytes\":" << bytes_
                   << ",\"updated_at\":" << updated_
                   << ",\"first_packet_at\":" << first_ << "}\n";
        }
        output.close();
        if (output && std::rename(temporary.c_str(), path_.c_str()) == 0) {
            published_ = updated_;
        } else {
            std::remove(temporary.c_str());
        }
    }
};
