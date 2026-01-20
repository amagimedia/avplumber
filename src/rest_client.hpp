#pragma once
#include <string>
#include "util.hpp"
#include "avutils.hpp"
#include <readerwriterqueue/readerwriterqueue.h>

class RESTEndpoint {
protected:
    std::string base_url_;
    AVTS min_interval_ = -1;
    AVTS last_send_ = -86400000;
public:
    RESTEndpoint(const std::string url);
    RESTEndpoint();
    void setBaseURL(const std::string url);
    void setMinimumInterval(const float seconds);
    void send(const std::string path, const std::string data = "");
protected:
    AVTS sendInternal(const std::string path, const std::string data); // returns after how much ms should we retry
};

class ThreadedRESTEndpoint: public RESTEndpoint {
protected:
    struct QueueEntry {
        std::string path;
        std::string data;
        bool finish;
    };
    bool has_thread_;
    std::thread thread_;
    moodycamel::BlockingReaderWriterQueue<QueueEntry> queue_;
public:
    void send(const std::string path, const std::string data = "");
    using RESTEndpoint::RESTEndpoint;
    ~ThreadedRESTEndpoint();
};