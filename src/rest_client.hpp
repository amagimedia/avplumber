#pragma once
#include <string>
#include <cpr/cpr.h>
#include "util.hpp"
#include "avutils.hpp"
#include <readerwriterqueue/readerwriterqueue.h>

class RESTEndpoint {
protected:
    std::string base_url_;
    cpr::Header get_headers_ = {{"User-Agent", APP_VERSION}, {"charset", "utf-8"}};
    cpr::Header post_headers_ = {{"User-Agent", APP_VERSION}, {"charset", "utf-8"}, {"Content-Type", "application/json"}};
    AVTS min_interval_ = -1;
    AVTS last_send_ = -86400000;
public:
    RESTEndpoint(const std::string url): base_url_(url) {
    }
    RESTEndpoint() {
    }
    void setBaseURL(const std::string url) {
        base_url_ = url;
    }
    void setMinimumInterval(const float seconds) {
        min_interval_ = float(wallclock.timeBase().den) * float(seconds) / float(wallclock.timeBase().num);
    }
    void send(const std::string path, const std::string data = "") {
        sendInternal(path, data);
    }
protected:
    AVTS sendInternal(const std::string path, const std::string data) { // returns after how much ms should we retry
        if (min_interval_ > 0) {
            AVTS now = wallclock.pts();
            AVTS delta = now - last_send_;
            if (delta < min_interval_) {
                return min_interval_ - delta;
            }
            last_send_ = now;
        }

        const std::string url_str = base_url_ + path;
        if (url_str.empty() || url_str == std::string("-")) {
            // empty url or "-" = debug mode, output to terminal
            logstream << data;
            return -1;
        }
        try {
            cpr::Url url(url_str);
            cpr::Response resp;
            //logstream << "before REST " << url;
            if (data.empty()) {
                resp = cpr::Get(url, get_headers_);
            } else {
                resp = cpr::Post(url, post_headers_, cpr::Body(data));
            }
            //logstream << "after REST, " << resp.status_line;
            return -1;
        } catch (std::exception &e) {
            logstream << "curl error when accessing " << path << ": " << e.what() << std::endl;
            return 1000;
        }
    }
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
    void send(const std::string path, const std::string data = "") {
        if (!has_thread_) {
            thread_ = start_thread("REST sender", [this]() {
                AVTS retry_in = -1;
                QueueEntry entry;
                bool have_entry = false;
                while(true) {
                    bool got;
                    if (retry_in >= 0) {
                        got = queue_.wait_dequeue_timed(entry, retry_in * 1000000 * wallclock.timeBase().num / wallclock.timeBase().den);
                    } else {
                        queue_.wait_dequeue(entry);
                        got = true;
                    }
                    if (got) {
                        have_entry = true;
                    }
                    if (have_entry) {
                        if (entry.finish) break;
                        retry_in = sendInternal(entry.path, entry.data);
                        if (retry_in < 0) {
                            have_entry = false;
                        }
                    }
                    // TODO: the last entry with data (last before finish=true) will be lost when hitting the rate limiter and then calling the destructor
                }
            });
            has_thread_ = true;
        }
        queue_.enqueue({path, data, false});
    }
    using RESTEndpoint::RESTEndpoint;
    ~ThreadedRESTEndpoint() {
        if (has_thread_) {
            queue_.enqueue({"", "", true});
            thread_.join();
        }
    }
};