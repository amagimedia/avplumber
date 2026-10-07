#include "node_common.hpp"

#include <cstdio>

extern "C" {
#include <libavutil/parseutils.h>
}

extern "C" void avpshim_destroy(void* self) {
    delete static_cast<avpshim::Holder*>(self);
}

extern "C" AvpFlow avpshim_process(void* self) {
    auto* holder = static_cast<avpshim::Holder*>(self);
    try {
        holder->process();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cpp node: %s\n", ex.what());
        return AVP_FLOW_ERROR;
    } catch (...) {
        std::fprintf(stderr, "cpp node: unknown exception\n");
        return AVP_FLOW_ERROR;
    }
    return holder->finished() ? AVP_FLOW_EOF : AVP_FLOW_PUSHED;
}

// src/util.hpp declares these and calls print_stack_trace from Error's
// constructor. The node only needs that path; the others stay defined so a
// logstream expansion in this TU can link.
void print_stack_trace() {}
void set_thread_name(const std::string&) {}
std::string now_str() { return {}; }
std::list<std::string> jsonToStringList(const Parameters& jitem) {
    std::list<std::string> out;
    if (jitem.is_string()) {
        out.push_back(jitem.get<std::string>());
    } else if (jitem.is_array()) {
        for (const auto& item : jitem) {
            out.push_back(item.get<std::string>());
        }
    } else {
        throw Error("Single string or list of strings should be provided!");
    }
    return out;
}

av::Rational parseRatio(const std::string ratio) {
    AVRational parsed{};
    if (av_parse_ratio(&parsed, ratio.c_str(), 1 << 24, AV_LOG_MAX_OFFSET, nullptr) < 0) {
        throw Error("Invalid ratio " + ratio);
    }
    return av::Rational(parsed.num, parsed.den);
}
std::shared_ptr<Logger> default_logger;
thread_local ThreadInfo current_thread;
