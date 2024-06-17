#include "graph_interfaces.hpp"
#include "instance_shared.hpp"
#include <jack/jack.h>
#include <jack/types.h>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/samplefmt.h>
}

class JackClient : public InstanceShared<JackClient> {
protected:
  jack_client_t *jack_client_;
  std::string client_name_;
  std::vector<std::weak_ptr<IJackSink>> sinks_;

  static int jack_process_callback(jack_nframes_t nframes, void *p) {
    auto *jc = (JackClient *)(p);
    for (const auto &sptr : jc->sinks_) {
      auto s = sptr.lock();
      if (s) s->jack_process(nframes);
    }
    return 0;
  }

public:

  JackClient(std::string name) {
    client_name_ = name;
    jack_status_t status;
    jack_client_ =
        jack_client_open(client_name_.c_str(), JackNoStartServer, &status);
    if (jack_client_ == nullptr) {
      throw Error("unable to create jack client, status: " +
                  std::to_string(status));
    }

    jack_set_process_callback(jack_client_, jack_process_callback, this);
    if (jack_activate(jack_client_) != 0) {
      throw Error("cannot activate client");
    }
  }

  ~JackClient() {
    jack_deactivate(jack_client_);
    if (jack_client_) {
      jack_client_close(jack_client_);
    }
  }

  void addSink(std::weak_ptr<IJackSink> sink) {
    if (!sinks_.empty()) {
      sinks_.erase(
        std::remove_if(
          sinks_.begin(), sinks_.end(),
          [](std::weak_ptr<IJackSink> &s) { return s.expired(); }
        ),
        sinks_.end()
      );
    }
    sinks_.push_back(sink);
  }

  jack_client_t *instance() {
    return jack_client_;
  }
};