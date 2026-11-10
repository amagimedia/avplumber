#include "util.hpp"

#include "app_version.hpp"

#include <cassert>
#include <cstdlib>
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

std::string now_utc_rfc3339() {
    struct tm timeinfo;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    gmtime_r(&tv.tv_sec, &timeinfo);
    char s[32];
    strftime(s, sizeof(s)-1, "%Y-%m-%dT%H:%M:%S", &timeinfo);
    char out[40];
    assert(tv.tv_usec < 1000000);
    snprintf(out, sizeof(out), "%s.%03ldZ", s, static_cast<long>(tv.tv_usec / 1000));
    return {out};
}

static std::string env_or(const char *primary, const char *secondary, const char *fallback) {
    const char *value = getenv(primary);
    if (value && *value) return {value};
    value = getenv(secondary);
    if (value && *value) return {value};
    return {fallback};
}

LogLineFormat log_line_format() {
    static const LogLineFormat format = []{
        const char *value = getenv("AVPLUMBER_LOG_FORMAT");
        return (value && std::string(value) == "json") ? LogLineFormat::Json : LogLineFormat::Text;
    }();
    return format;
}

std::string format_log_line(
    const std::string &timestamp, const std::string &thread_name,
    const std::string &message
) {
    if (log_line_format() == LogLineFormat::Text) {
        return timestamp + " [" + thread_name + "] " + message;
    }
    // The OTel JSON log line: one object per line with the field names of the
    // OTel log data model. The engine's logstream has no levels, so every line is
    // INFO; the legacy local-time stamp is not carried.
    nlohmann::json fields;
    fields["timestamp"] = timestamp;
    fields["severity_text"] = "INFO";
    fields["severity_number"] = 9;
    fields["body"] = message;
    fields["service.name"] = env_or("OTEL_SERVICE_NAME", "", "avplumber");
    fields["service.version"] = env_or("OTEL_SERVICE_VERSION", "", APP_VERSION);
    fields["deployment.environment"] = env_or("DEPLOYMENT_ENVIRONMENT", "OTEL_DEPLOYMENT_ENVIRONMENT", "unknown");
    fields["logger"] = "avplumber";
    fields["thread"] = thread_name;
    return fields.dump();
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
