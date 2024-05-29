#include "node_common.hpp"
#include <jack/jack.h>
#include <jack/types.h>
#include <string>

class JackSink : public NodeSingleInput<av::AudioSamples> {
protected:
  int channels_count;
  std::string port_basename;

  jack_client_t *jack_client;
  jack_port_t **jack_ports;

public:
  JackSink(const JackSink &) = delete;
  JackSink(JackSink &&) = delete;
  JackSink &operator=(const JackSink &) = delete;
  JackSink &operator=(JackSink &&) = delete;

  void prepare() {
    jack_status_t status;
    logstream << "open client";
    jack_client = jack_client_open("test_client", JackNullOption, &status);
    if (jack_client == nullptr) {
      throw Error("unable to create jack client, status: " +
                  std::to_string(status));
    }

    jack_ports = (jack_port_t **)malloc(sizeof(jack_port_t *) * channels_count);
    for (int i = 0; i < channels_count; i++) {
      std::string out_port = port_basename + "_out" + std::to_string(i);

      jack_ports[i] =
          jack_port_register(jack_client, out_port.c_str(),
                             JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
      logstream << "port registered: " << jack_ports[i];
      if (jack_ports[i] == nullptr) {
        throw Error("unable to register jack port");
      }

      // Will it be neccessary to link in/out ports inside avplumber?
      /*std::string in_port = port_basename + "_in" + std::to_string(i);
      int connect_res =
          jack_connect(jack_client, out_port.c_str(), in_port.c_str());
      if (connect_res > 1) {
        throw Error("could not link in/out jack port");
      }*/
    }
  }

  ~JackSink() {
    for (int i = 0; i < channels_count; i++) {
      if (jack_ports[i]) {
        jack_port_unregister(jack_client, jack_ports[i]);
      }
    }

    if (jack_client) {
      jack_client_close(jack_client);
    }
  }

  using NodeSingleInput<av::AudioSamples>::NodeSingleInput;
  virtual void process() override {
    av::AudioSamples as = this->source_->get();
    if (!as.isComplete())
      return;

    // logstream << "got audio samples: " << as.samplesCount() << " "
    //          << as.channelsCount() << " " << as.channelsLayoutString();

    jack_default_audio_sample_t *buf[as.channelsCount()];

    if (!as.isPlanar()) {
      logstream << "audio input not planar";
      return;
    }

    for (int i = 0; i < as.channelsCount(); i++) {
      if (jack_ports[i] == nullptr) {
        throw Error("no jack_port");
      }

      buf[i] = (jack_default_audio_sample_t *)jack_port_get_buffer(
          jack_ports[i], as.samplesCount());
      if (buf[i] == nullptr) {
        throw Error("failed to get jack buffer");
      }
      memset(buf[i], 0, as.samplesCount() * sizeof(buf[i]));
      for (int j = 0; j < as.samplesCount(); j++) {
        buf[i][j] = (as.data() + i * as.samplesCount())[j];
      }
    }
  }

  static std::shared_ptr<JackSink> create(NodeCreationInfo &nci) {

    EdgeManager &edges = nci.edges;
    const Parameters &params = nci.params;

    std::shared_ptr<Edge<av::AudioSamples>> src_edge =
        edges.find<av::AudioSamples>(params["src"]);
    auto r = std::make_shared<JackSink>(
        make_unique<EdgeSource<av::AudioSamples>>(src_edge));

    if (params.count("channels_count")) {
      r->channels_count = params["channels_count"];
    }

    if (params.count("port_basename")) {
      r->port_basename = params["port_basename"];
    }

    r->prepare();
    return r;
  }
};

DECLNODE(jack_sink, JackSink);
