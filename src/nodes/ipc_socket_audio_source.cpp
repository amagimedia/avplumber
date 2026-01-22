#include "node_common.hpp"

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <vector>
#include <string>

// Audio frame wire format (sender: electron-hwaccel, receiver: browser-ingest)
// Little-endian:
//  - uint32  magic = 0x4D524641 ('A''F''R''M')
//  - uint64  timestampNs
//  - uint32  payloadLenBytes
//  - uint32  reserved = 0
//  - payload PCM interleaved float32 (F32LE), channels*c*frames samples

#pragma pack(push, 1)
struct AFRMHeader {
    uint32_t magic;
    uint64_t timestampNs;
    uint32_t payloadLen;
    uint32_t reserved;
};
#pragma pack(pop)

class IPCSocketAudioSource: public NodeSingleOutput<av::AudioSamples>, public ReportsFinishByFlag, public IStoppable, public IAudioMetadataSource, public ITimeBaseSource {
protected:
    static constexpr uint32_t MAGIC_AFRM = 0x4D524641u; // 'AFRM' LE

    // Configuration
    std::string socket_path_;
    int sample_rate_ = 48000;
    int channels_ = 2;
    int bytes_per_sample_ = 4; // float32
    int reconnect_delay_ms_ = 50;

    // State
    int sock_fd_ = -1;
    std::vector<uint8_t> inbuf_;

public:
    using NodeSingleOutput::NodeSingleOutput;

    IPCSocketAudioSource(std::unique_ptr<SinkType> &&sink,
                const std::string &socket_path,
                int sample_rate,
                int channels,
                int bytes_per_sample,
                int reconnect_delay_ms)
        : NodeSingleOutput(std::move(sink)),
          socket_path_(socket_path),
          sample_rate_(sample_rate > 0 ? sample_rate : 48000),
          channels_(channels > 0 ? channels : 2),
          bytes_per_sample_(bytes_per_sample > 0 ? bytes_per_sample : 4),
          reconnect_delay_ms_(reconnect_delay_ms > 0 ? reconnect_delay_ms : 50) {
    }

    // IAudioMetadataSource implementation
    virtual int sampleRate() override {
        return sample_rate_;
    }
    
    virtual av::SampleFormat sampleFormat() override {
        return av::SampleFormat(AV_SAMPLE_FMT_FLT);
    }
    
    virtual uint64_t channelLayout() override {
#if API_NEW_CHANNEL_LAYOUT
        return av::ChannelLayout(channels_).layout();
#else
        return av_get_default_channel_layout(channels_);
#endif
    }
    
    virtual av::Rational timeBase() override {
        return {1, sample_rate_};
    }

    virtual ~IPCSocketAudioSource() {
        closeSocket();
    }

    virtual void stop() {
        this->finished_ = true;
        closeSocket();
    }

protected:
    void closeSocket() {
        if (sock_fd_ >= 0) {
            close(sock_fd_);
            sock_fd_ = -1;
        }
    }

    bool ensureConnected() {
        if (sock_fd_ >= 0) return true;
        if (socket_path_.empty()) return false;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        sockaddr_un addr; memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
        int rc = connect(fd, (struct sockaddr*)&addr, sizeof(addr));
        if (rc < 0) {
            close(fd);
            return false;
        }
        sock_fd_ = fd;
        return true;
    }

    bool recvSome() {
        const size_t chunk = 65536;
        const size_t old_size = inbuf_.size();
        inbuf_.resize(old_size + chunk);
        ssize_t r = recv(sock_fd_, inbuf_.data() + old_size, chunk, 0);
        if (r < 0) {
            inbuf_.resize(old_size);
            if (errno == EINTR) return true; // retry by caller
            return false;
        }
        if (r == 0) {
            // peer closed
            inbuf_.resize(old_size);
            return false;
        }
        inbuf_.resize(old_size + (size_t)r);
        return true;
    }

    static bool haveEnoughBytes(const std::vector<uint8_t> &buf, size_t n) {
        return buf.size() >= n;
    }

public:
    virtual void process() {
        if (this->finished_) {
            wallclock.sleepms(50);
            return;
        }

        if (!ensureConnected()) {
            logstream << "ipc_socket_audio_source: connect failed to '" << socket_path_ << "'";
            wallclock.sleepms(reconnect_delay_ms_);
            return;
        }

        // Ensure we have at least one full packet; block until we do or error occurs
        for (;;) {
            if (!haveEnoughBytes(inbuf_, sizeof(AFRMHeader))) {
                if (!recvSome()) { closeSocket(); wallclock.sleepms(reconnect_delay_ms_); return; }
                continue;
            }
            AFRMHeader hdr;
            memcpy(&hdr, inbuf_.data(), sizeof(hdr));
            if (hdr.magic != MAGIC_AFRM) {
                logstream << "ipc_socket_audio_source: invalid magic: " << std::hex << hdr.magic << ", should be " << MAGIC_AFRM;
                // search for magic by discarding one byte at a time
                inbuf_.erase(inbuf_.begin());
                continue;
            }
            const uint64_t ts_ns = hdr.timestampNs;
            const uint32_t payload_len = hdr.payloadLen;
            if (payload_len > 65536*4) {
                logstream << "ipc_socket_audio_source: payload length too large: " << payload_len;
                inbuf_.erase(inbuf_.begin());
                continue;
            }
            size_t total_needed = sizeof(AFRMHeader) + (size_t)payload_len;
            if (!haveEnoughBytes(inbuf_, total_needed)) {
                // in the middle of read
                if (!recvSome()) { closeSocket(); wallclock.sleepms(reconnect_delay_ms_); return; }
                continue;
            }

            const uint8_t *payload = inbuf_.data() + sizeof(AFRMHeader);
            const int ch = channels_ > 0 ? channels_ : 2;
            const int bps = bytes_per_sample_ > 0 ? bytes_per_sample_ : 4;
            if (ch <= 0 || bps <= 0) {
                inbuf_.erase(inbuf_.begin(), inbuf_.begin() + total_needed);
                logstream << "ipc_socket_audio_source: invalid channels or bytes per sample: " << ch << " " << bps;
                break; // nothing to do this cycle
            }
            int frames = (int)(payload_len / (uint32_t)(ch * bps));
            if (frames <= 0) {
                inbuf_.erase(inbuf_.begin(), inbuf_.begin() + total_needed);
                logstream << "ipc_socket_audio_source: invalid number of frames: " << frames;
                break;
            }

            // Create output frame: interleaved float32
            av::SampleFormat fmt(AV_SAMPLE_FMT_FLT);
#if API_NEW_CHANNEL_LAYOUT
            int64_t channel_layout = av::ChannelLayout(ch).layout();
#else
            int64_t channel_layout = av_get_default_channel_layout(ch);
#endif
            av::AudioSamples outfrm(fmt, frames, channel_layout, sample_rate_, av::SampleFormat::Alignment::AlignDefault);
            size_t copy_bytes = (size_t)frames * (size_t)ch * (size_t)sizeof(float);
            uint8_t *dst = outfrm.data(0);
            memcpy(dst, payload, copy_bytes);

            outfrm.setTimeBase({1, sample_rate_});
            outfrm.raw()->pts = __int128(ts_ns) * __int128(sample_rate_) / __int128(1000000000);
            outfrm.setComplete(true);
            this->sink_->put(outfrm);

            // consume packet
            inbuf_.erase(inbuf_.begin(), inbuf_.begin() + total_needed);
            break; // produce at most one frame per process()
        }
    }

    static std::shared_ptr<IPCSocketAudioSource> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<av::AudioSamples>> edge = edges.find<av::AudioSamples>(params["dst"]);
        std::string socket_path = params.count("socket") ? (std::string)params["socket"] : std::string();
        int sample_rate = params.count("sample_rate") ? (int)params["sample_rate"] : 48000;
        int channels = params.count("channels") ? (int)params["channels"] : 2;
        int bytes_per_sample = params.count("bytes_per_sample") ? (int)params["bytes_per_sample"] : 4;
        int reconnect_delay_ms = params.count("reconnect_delay_ms") ? (int)params["reconnect_delay_ms"] : 50;
        return std::make_shared<IPCSocketAudioSource>(make_unique<EdgeSink<av::AudioSamples>>(edge), socket_path, sample_rate, channels, bytes_per_sample, reconnect_delay_ms);
    }
};

DECLNODE(ipc_socket_audio_source, IPCSocketAudioSource);


