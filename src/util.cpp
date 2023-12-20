#include "util.hpp"
#include <cassert>
#include <thread>
#include <sys/prctl.h>
#include <sys/time.h>
#include <time.h>
//#include <execinfo.h>
#include <boost/stacktrace.hpp>
#include "logger_impls.hpp"

thread_local ThreadInfo current_thread;
std::shared_ptr<Logger> default_logger = std::make_shared<StdErrLogger>();

std::string now_str() {
    static constexpr size_t slen = 40;
    char s[slen];
    struct tm timeinfo;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    localtime_r(&tv.tv_sec, &timeinfo);
    strftime(s, slen-1, "%Y-%m-%d %H:%M:%S.xxxxxx %z", &timeinfo);
    assert(tv.tv_usec < 1000000);
    sprintf(s+20, "%06ld", tv.tv_usec);
    s[26] = ' '; // sprintf will store null terminator, restore the space
    return {s};
}

void set_thread_name_internal(const char* name) {
    prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);
}

void set_thread_name(const std::string &name) {
    current_thread.name = name;
    set_thread_name_internal(name.c_str());
}

void print_stack_trace() {
    logstream << "Stack trace:";
#if 0
    const size_t maxlen = 100;
    void* bt[maxlen];
    size_t size = backtrace(bt, maxlen);
    backtrace_symbols_fd(bt, size, 2); // 2 = stderr
#else
    logstream << boost::stacktrace::stacktrace();
#endif
}

std::list< std::string > jsonToStringList(const Parameters& jitem) {
    std::list<std::string> r;
    if (jitem.is_string()) {
        r.push_back(jitem.get<std::string>());
    } else if (jitem.is_array()) {
        for (auto &s: jitem) {
            r.push_back(s.get<std::string>());
        }
    } else {
        throw Error("Single string or list of strings should be provided!");
    }
    return r;
}

std::thread start_thread(const std::string name, std::function<void()> whattodo) {
    std::shared_ptr<Logger> logger = current_thread.logger;
    return std::thread([name, logger, whattodo]() {
        current_thread.logger = logger; // inherit from calling thread
        set_thread_name(name);
        whattodo();
    });
}

