#include "rest_client.hpp"
#include <cpr/cpr.h>

namespace {
    static cpr::Header get_headers = {{"User-Agent", APP_VERSION}, {"charset", "utf-8"}};
    static cpr::Header post_headers = {{"User-Agent", APP_VERSION}, {"charset", "utf-8"}, {"Content-Type", "application/json"}};
}

RESTEndpoint::RESTEndpoint(const std::string url): base_url_(url) {
}

RESTEndpoint::RESTEndpoint() {
}

void RESTEndpoint::setBaseURL(const std::string url) {
    base_url_ = url;
}

void RESTEndpoint::setMinimumInterval(const float seconds) {
    min_interval_ = float(wallclock.timeBase().den) * float(seconds) / float(wallclock.timeBase().num);
}

void RESTEndpoint::send(const std::string path, const std::string data) {
    sendInternal(path, data);
}

AVTS RESTEndpoint::sendInternal(const std::string path, const std::string data) {
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
            resp = cpr::Get(url, get_headers);
        } else {
            resp = cpr::Post(url, post_headers, cpr::Body(data));
        }
        //logstream << "after REST, " << resp.status_line;
        return -1;
    } catch (std::exception &e) {
        logstream << "curl error when accessing " << path << ": " << e.what() << std::endl;
        return 1000;
    }
}

void ThreadedRESTEndpoint::send(const std::string path, const std::string data) {
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

ThreadedRESTEndpoint::~ThreadedRESTEndpoint() {
    if (has_thread_) {
        queue_.enqueue({"", "", true});
        thread_.join();
    }
}
