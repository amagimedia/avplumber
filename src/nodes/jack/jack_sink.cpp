#include "../JackClient.hpp"
#include "../graph_interfaces.hpp"
#include "node_common.hpp"
#include <jack/jack.h>
#include <jack/types.h>
#include <memory>
#include <string>
extern "C" {
#include <libavutil/samplefmt.h>
}

#include <boost/lockfree/spsc_queue.hpp>

static constexpr size_t ring_buffer_size = 16 * 1024;
typedef boost::lockfree::spsc_queue<float, boost::lockfree::capacity<ring_buffer_size> > spsc_queue;

class JackSink : public NodeSingleInput<av::AudioSamples>, public IJackSink {

protected:
  static float ZEROS[ring_buffer_size];
  int channels_count_ = 2;
  std::string port_prefix_;
  std::string connect_port_prefix_;

  std::shared_ptr<JackClient> jack_client_;
  std::vector<jack_port_t *> jack_ports_;

  std::unique_ptr<spsc_queue[]> channel_buffers_;

public:
  void prepare() {
    channel_buffers_ = std::make_unique<spsc_queue[]>(channels_count_);
    jack_ports_.reserve(channels_count_);
    for (int i = 0; i < channels_count_; i++) {
      std::string out_port = port_prefix_ + std::to_string(i + 1);
      jack_port_t *port =
          jack_port_register(jack_client_->instance(), out_port.c_str(),
                             JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
      logstream << "port registered: " << port;
      if (port == nullptr) {
        throw Error("unable to register jack port");
      }
      jack_ports_.push_back(port);

      if (!connect_port_prefix_.empty()) {
        std::string in_port =
            connect_port_prefix_ + std::to_string(i);
        int connect_res = jack_connect(jack_client_->instance(),
                                       out_port.c_str(), in_port.c_str());
        if (connect_res > 1) {
          throw Error("could not link in/out jack port");
        }
      }
    }
  }

  ~JackSink() {
    for (jack_port_t *port : jack_ports_) {
      jack_port_unregister(jack_client_->instance(), port);
    }
  }

  using NodeSingleInput<av::AudioSamples>::NodeSingleInput;

  virtual void process() override {
    av::AudioSamples as = this->source_->get();
    if (!as.isComplete()) {
      return;
    }

    if (as.sampleFormat().get() != AV_SAMPLE_FMT_FLTP) {
      logstream << "audio input not compatible with jack (must be fltp)";
      return;
    }

    for (int ch = 0; ch < channels_count_; ch++) {
      if (ch < as.channelsCount()) {
        float *src_buffer = reinterpret_cast<float *>(as.data(ch));
        if (!channel_buffers_[ch].write_available()) {
          logstream << "ringbuffer overflow";
          return;
        }
        channel_buffers_[ch].push(src_buffer, as.samplesCount());
      } else {
        channel_buffers_[ch].push(ZEROS, ZEROS + as.samplesCount());
      }
    }
  }

  void jack_process(size_t nframes) override {
    if (jack_ports_.size() != channels_count_) {
      return;
    }
    for (int ch = 0; ch < channels_count_; ch++) {
      if (jack_ports_[ch] == nullptr) {
        throw Error("no jack_port");
      }

      jack_default_audio_sample_t *buf =
          (jack_default_audio_sample_t *)jack_port_get_buffer(jack_ports_[ch],
                                                              nframes);
      if (buf == nullptr) {
        throw Error("failed to get jack buffer");
      }

      if (!channel_buffers_[ch].read_available()) {
        //logstream << "ringbuffer underflow";
        std::fill(buf, buf + nframes, 0);
      }
      channel_buffers_[ch].pop(buf, nframes);
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
      r->channels_count_ = params["channels_count"];
    }
    if (params.count("port_prefix")) {
      r->port_prefix_ = params["port_prefix"];
    }
    if (params.count("connect_port_prefix")) {
      r->connect_port_prefix_ = params["connect_port_prefix"];
    }
    if (params.count("client_name")) {
      std::string name = params["client_name"];
      InstanceSharedObjects<JackClient>::emplace(nci.instance, name, InstanceSharedObjects<JackClient>::PolicyIfExists::Ignore, name);
      r->jack_client_ =
          InstanceSharedObjects<JackClient>::get(nci.instance, name);

      std::weak_ptr<JackSink> w(r);
      r->jack_client_->addSink(w);
    } else {
      throw Error("client_name not specified");
    }

    r->prepare();
    return r;
  }
};

float JackSink::ZEROS[ring_buffer_size] = {0};

DECLNODE(jack_sink, JackSink);
