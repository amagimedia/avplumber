#pragma once
#include <iostream>
#include <stdexcept>
#include <thread>
#include <functional>
#include <sstream>
#include <unordered_map>
#include <assert.h>
#include <mutex>
#include <list>
#include <string>
#include <json.hpp>
#include "app_version.hpp"

#ifdef __GNUC__
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif


template<typename T> using Shared = std::shared_ptr<T>;

using Parameters = nlohmann::json;

void print_stack_trace();

class Error: public std::runtime_error {
public:
    Error(const std::string &desc): std::runtime_error(desc) {
        print_stack_trace();
    }
};
class NotReallyError: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


namespace {
    #if __cplusplus >= 201402L
    using std::make_unique;
    #else
    template<typename T, typename ...Args> std::unique_ptr<T> make_unique( Args&& ...args )
    {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }
    #endif
    template<typename T> void ensureNotNull(T pointer, std::string error = "NullPointerException") {
        if (pointer==nullptr) throw Error(error);
    }
    static void soft_assert(bool requirement, std::string error = "soft assertion failed") {
        if (!requirement) {
            throw Error(error);
        }
    }
}

void set_thread_name(const std::string &name);
std::string now_str();

std::list< std::string > jsonToStringList(const Parameters& jitem);

class Logger {
protected:
    virtual void write(const std::string &) = 0;
public:
    void log(const std::string &line) {
        if (line.empty()) return;
        if (line.back() != '\n') {
            write(line + '\n');
        } else {
            write(line);
        }
    }
    virtual ~Logger() {
    }
};

extern std::shared_ptr<Logger> default_logger;

struct ThreadInfo {
    std::string name = "?";
    std::shared_ptr<Logger> logger = default_logger;
};

extern thread_local ThreadInfo current_thread;

std::thread start_thread(const std::string name, std::function<void()> whattodo);

class LogLine {
protected:
    std::ostringstream ss_;
    Logger* logger_;
public:
    LogLine(Logger* logger): logger_(logger) {
        ss_ << now_str() << " [" << current_thread.name << "] ";
    }
    std::ostringstream& stream() {
        return ss_;
    }
    ~LogLine() {
        if (logger_) {
            logger_->log(ss_.str());
        } else {
            std::cerr << ss_.str() << std::endl;
        }
    }
};

#define logstream (LogLine(current_thread.logger.get()).stream())

