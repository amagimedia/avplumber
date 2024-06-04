#include "node_common.hpp"
#include <jack/jack.h>
#include <jack/types.h>
#include <string>
extern "C" {
#include <libavutil/samplefmt.h>
}
#include "../RingBuffer.hpp"

class JackSink : public NodeSingleInput<av::AudioSamples> {

protected:
  size_t ring_buffer_size_ = 131072;
  int channels_count_ = 2;
  std::string port_basename_;
  std::string client_name_;

  jack_client_t *jack_client_;
  jack_port_t **jack_ports_;

  std::vector<RingBuffer<float>> channel_buffers_;

public:
  JackSink(const JackSink &) = delete;
  JackSink(JackSink &&) = delete;
  JackSink &operator=(const JackSink &) = delete;
  JackSink &operator=(JackSink &&) = delete;

  void prepare() {
    jack_status_t status;
    jack_client_ =
        jack_client_open(client_name_.c_str(), JackNoStartServer, &status);
    if (jack_client_ == nullptr) {
      throw Error("unable to create jack client, status: " +
                  std::to_string(status));
    }

    channel_buffers_.reserve(channels_count_);
    jack_ports_ = (jack_port_t **)malloc(sizeof(jack_port_t *) * channels_count_);
    for (int i = 0; i < channels_count_; i++) {
      channel_buffers_.push_back(RingBuffer<float>(ring_buffer_size_));

      std::string out_port = port_basename_ + "_out " + std::to_string(i + 1);

      jack_ports_[i] =
          jack_port_register(jack_client_, out_port.c_str(),
                             JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
      logstream << "port registered: " << jack_ports_[i];
      if (jack_ports_[i] == nullptr) {
        throw Error("unable to register jack port");
      }
      /*Will it be neccessary to link in/out ports inside avplumber?
      std::string in_port = port_basename + "_in" + std::to_string(i);
      int connect_res =
          jack_connect(jack_client, out_port.c_str(), in_port.c_str());
      if (connect_res > 1) {
        throw Error("could not link in/out jack port");
      }*/
    }

    jack_set_process_callback(jack_client_, jack_process_callback, this);
    if (jack_activate(jack_client_) != 0) {
      throw Error("cannot activate client");
    }
  }

  ~JackSink() {
    jack_deactivate(jack_client_);
    for (int i = 0; i < channels_count_; i++) {
      if (jack_ports_[i]) {
        jack_port_unregister(jack_client_, jack_ports_[i]);
      }
    }
    if (jack_client_) {
      jack_client_close(jack_client_);
    }
    free(jack_ports_);
  }

  using NodeSingleInput<av::AudioSamples>::NodeSingleInput;

  static int jack_process_callback(jack_nframes_t nframes, void *p) {
    auto *t = (JackSink *)(p);
    t->jack_process(nframes);
    return 0;
  }

  virtual void process() override {
    av::AudioSamples as = this->source_->get();
    if (!as.isComplete())
      return;

    // logstream << "got audio samples: " << as.samplesCount() << " "
    //          << as.channelsCount() << " " << as.channelsLayoutString();
    if (!as.isPlanar()) {
      logstream << "audio input not planar";
      return;
    }

    if (as.sampleFormat().get() != AV_SAMPLE_FMT_FLTP) {
      logstream << "audio input not compatible with jack (must be fltp)";
      return;
    }

    for (int ch=0; ch<channels_count_; ch++) {
      if (ch < as.channelsCount()) {
        float* src_buffer = reinterpret_cast<float*>(as.data(ch));
        if (!channel_buffers_[ch].writeFrom(src_buffer, src_buffer + as.samplesCount())) {
          logstream << "ringbuffer overflow";
        }
      } else {
        channel_buffers_[ch].fill(as.samplesCount(), 0);
      }
    }

  }

  void jack_process(jack_nframes_t nframes) {
    for (int ch = 0; ch < channels_count_; ch++) {
      if (jack_ports_[ch] == nullptr) {
        throw Error("no jack_port");
      }

      jack_default_audio_sample_t *buf =
          (jack_default_audio_sample_t *)jack_port_get_buffer(
              jack_ports_[ch], nframes);
      if (buf == nullptr) {
        throw Error("failed to get jack buffer");
      }

      if (!channel_buffers_[ch].readTo(buf, buf + nframes)) {
        std::fill(buf, buf + nframes, 0);
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
      r->channels_count_ = params["channels_count"];
    }
    if (params.count("port_basename")) {
      r->port_basename_ = params["port_basename"];
    }
    if (params.count("client_name")) {
      r->client_name_ = params["client_name"];
    }

    r->prepare();
    return r;
  }
};

DECLNODE(jack_sink, JackSink);
