#include "node_common.hpp"

#include <cstdio>

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
std::list<std::string> jsonToStringList(const Parameters&) { return {}; }
std::shared_ptr<Logger> default_logger;
thread_local ThreadInfo current_thread;
