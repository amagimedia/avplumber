#include "../node_common.hpp"
#include "CudaVmmFrames.hpp"
#include "../../cuda.hpp"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unordered_map>
#include <unistd.h>

namespace avp::vmm {
namespace {

constexpr unsigned protocol_version = 1;
constexpr size_t max_message = 8192;
constexpr size_t max_slots = 32;
constexpr size_t max_slot_bytes = 256 * 1024 * 1024;

class Fd {
    int fd_ = -1;

public:
    Fd() = default;
    explicit Fd(int fd) : fd_(fd) {}
    Fd(Fd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    Fd &operator=(Fd &&other) noexcept {
        Fd old(std::move(*this));
        fd_ = std::exchange(other.fd_, -1);
        return *this;
    }
    ~Fd() { if (fd_ >= 0) close(fd_); }
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
};

AVCUDADeviceContext *cudaDevice(HWAccelDevice &device) {
    const auto *context = reinterpret_cast<const AVHWDeviceContext *>(device.deviceContext()->data);
    if (context->type != AV_HWDEVICE_TYPE_CUDA)
        throw Error("cuda_ipc requires a CUDA hwaccel device");
    return static_cast<AVCUDADeviceContext *>(context->hwctx);
}

class Context {
public:
    explicit Context(CUcontext context) {
        if (CHECK_CU(cuCtxPushCurrent(context))) throw Error("cuda_ipc: cannot enter CUDA context");
    }
    ~Context() {
        CUcontext popped{};
        CHECK_CU(cuCtxPopCurrent(&popped));
    }
};

struct Layout {
    AVPixelFormat format;
    int width, height, pitch;
    size_t chroma_offset, bytes;

    static Layout make(AVPixelFormat format, int width, int height) {
        if (format != AV_PIX_FMT_NV12 && format != AV_PIX_FMT_P010LE && format != AV_PIX_FMT_RGBA)
            throw Error("cuda_ipc: format must be nv12, p010le or rgba");
        if (width <= 0 || height <= 0 || width > 8192 || height > 8192 ||
            (format != AV_PIX_FMT_RGBA && ((width | height) & 1)))
            throw Error("cuda_ipc: unsupported frame dimensions");
        const unsigned samples = format == AV_PIX_FMT_RGBA ? 4 : format == AV_PIX_FMT_P010LE ? 2 : 1;
        const int pitch = (width * samples + 255) / 256 * 256;
        const size_t luma = size_t(pitch) * height;
        const size_t chroma = format == AV_PIX_FMT_RGBA ? 0 : luma / 2;
        return {format, width, height, pitch, chroma ? luma : 0, luma + chroma};
    }

    Parameters message() const {
        return {{"format", av_get_pix_fmt_name(format)}, {"width", width}, {"height", height},
                {"pitch", pitch}, {"chroma_offset", chroma_offset}};
    }

    static Layout parse(const Parameters &body) {
        const auto layout = make(av_get_pix_fmt(body.at("format").get<std::string>().c_str()),
                                 body.at("width").get<int>(), body.at("height").get<int>());
        if (body.at("pitch").get<int>() != layout.pitch ||
            body.at("chroma_offset").get<size_t>() != layout.chroma_offset)
            throw Error("cuda_ipc: invalid plane layout");
        return layout;
    }

    AvBufferRef frames(HWAccelDevice &device) const {
        AvBufferRef ref(av_hwframe_ctx_alloc(device.deviceContext()));
        if (!ref) throw Error("cuda_ipc: cannot allocate frame context");
        auto *context = reinterpret_cast<AVHWFramesContext *>(ref->data);
        context->format = AV_PIX_FMT_CUDA;
        context->sw_format = format;
        context->width = width;
        context->height = height;
        if (const int error = av_hwframe_ctx_init(ref.get()); error < 0)
            throw Error("cuda_ipc: " + av::error2string(error));
        return ref;
    }

    av::VideoFrame frame(AVBufferRef *frames, AVBufferRef *buffer) const {
        av::VideoFrame result;
        auto *raw = result.raw();
        raw->buf[0] = buffer;
        raw->hw_frames_ctx = av_buffer_ref(frames);
        if (!raw->hw_frames_ctx) throw Error("cuda_ipc: cannot retain frame context");
        raw->format = AV_PIX_FMT_CUDA;
        raw->width = width;
        raw->height = height;
        raw->data[0] = buffer->data;
        raw->linesize[0] = pitch;
        if (chroma_offset) {
            raw->data[1] = buffer->data + chroma_offset;
            raw->linesize[1] = pitch;
        }
        result.setComplete(true);
        return result;
    }
};

sockaddr_un socketAddress(const std::string &path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.empty() || path.front() != '/' || path.size() >= sizeof(address.sun_path))
        throw Error("cuda_ipc: socket must be a short absolute path");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

Fd endpoint() {
    Fd result(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    if (!result) throw Error("cuda_ipc: cannot create Unix socket");
    return result;
}

struct Message {
    Parameters body;
    std::vector<Fd> fds;
};

bool transmit(const Fd &peer, const Parameters &fields, const std::vector<Fd> &fds = {}) {
    const auto payload = fields.dump();
    if (payload.size() > max_message || fds.size() > max_slots)
        throw Error("cuda_ipc: outbound message is too large");
    iovec io{const_cast<char *>(payload.data()), payload.size()};
    alignas(cmsghdr) char control[CMSG_SPACE(max_slots * sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    if (!fds.empty()) {
        msg.msg_control = control;
        msg.msg_controllen = CMSG_SPACE(fds.size() * sizeof(int));
        auto *header = CMSG_FIRSTHDR(&msg);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
        for (size_t i = 0; i < fds.size(); ++i) {
            const int fd = fds[i].get();
            std::memcpy(CMSG_DATA(header) + i * sizeof(fd), &fd, sizeof(fd));
        }
    }
    ssize_t size;
    do { size = sendmsg(peer.get(), &msg, MSG_NOSIGNAL); } while (size < 0 && errno == EINTR);
    if (size < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return false;
    if (size < 0 || size_t(size) != payload.size()) throw Error("cuda_ipc: socket send failed");
    return true;
}

std::optional<Message> receive(const Fd &peer) {
    char payload[max_message];
    iovec io{payload, sizeof(payload)};
    alignas(cmsghdr) char control[CMSG_SPACE(max_slots * sizeof(int))]{};
    msghdr msg{};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    ssize_t size;
    do { size = recvmsg(peer.get(), &msg, MSG_CMSG_CLOEXEC); } while (size < 0 && errno == EINTR);
    if (size < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return std::nullopt;
    if (size <= 0) throw Error("cuda_ipc: peer disconnected");
    Message result;
    result.fds.reserve(max_slots);
    for (auto *header = CMSG_FIRSTHDR(&msg); header; header = CMSG_NXTHDR(&msg, header)) {
        if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
            header->cmsg_len < CMSG_LEN(0))
            continue;
        const size_t count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
        for (size_t i = 0; i < count; ++i) {
            int fd;
            std::memcpy(&fd, CMSG_DATA(header) + i * sizeof(fd), sizeof(fd));
            result.fds.emplace_back(fd);
        }
    }
    if ((msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) || result.fds.size() > max_slots)
        throw Error("cuda_ipc: truncated packet or descriptors");
    result.body = Parameters::parse(payload, payload + size);
    return result;
}

struct ExportGeneration {
    std::shared_ptr<HWAccelDevice> device;
    Layout layout;
    AvBufferRef frames;
    std::vector<std::shared_ptr<Memory>> slots;
    std::vector<bool> used;
    std::mutex mutex;
    bool retired = false;
};

struct Lease {
    std::shared_ptr<ExportGeneration> pool;
    size_t slot;
};

std::mutex registry_mutex;
std::unordered_map<uint8_t *, Lease *> registry;

void releaseExport(void *opaque, uint8_t *address) noexcept {
    std::unique_ptr<Lease> lease(static_cast<Lease *>(opaque));
    {
        std::lock_guard guard(registry_mutex);
        registry.erase(address);
    }
    std::lock_guard guard(lease->pool->mutex);
    lease->pool->used[lease->slot] = false;
}

std::optional<Lease> exportLease(const av::VideoFrame &frame) {
    auto *raw = frame.raw();
    if (raw->format != AV_PIX_FMT_CUDA || !raw->buf[0] || !raw->hw_frames_ctx) return std::nullopt;
    std::lock_guard guard(registry_mutex);
    auto it = registry.find(raw->data[0]);
    if (it == registry.end() || av_buffer_get_opaque(raw->buf[0]) != it->second) return std::nullopt;
    const auto &lease = *it->second;
    const auto &layout = lease.pool->layout;
    if (raw->width != layout.width || raw->height != layout.height || raw->linesize[0] != layout.pitch ||
        (layout.chroma_offset && (raw->data[1] != raw->data[0] + layout.chroma_offset ||
                                  raw->linesize[1] != layout.pitch)))
        return std::nullopt;
    return lease;
}

struct ImportGeneration {
    std::shared_ptr<HWAccelDevice> device;
    Layout layout;
    AvBufferRef frames;
    std::vector<std::shared_ptr<Memory>> slots;
    std::mutex mutex;
    std::vector<std::pair<size_t, uint64_t>> released;
    std::vector<bool> held;
    uint64_t last_sequence = 0;
    uint64_t id = 0;
};

struct ImportLease {
    std::shared_ptr<ImportGeneration> generation;
    size_t slot;
    uint64_t sequence;
};

void releaseImport(void *opaque, uint8_t *) noexcept {
    std::unique_ptr<ImportLease> lease(static_cast<ImportLease *>(opaque));
    auto &pool = *lease->generation;
    std::lock_guard guard(pool.mutex);
    pool.released.emplace_back(lease->slot, lease->sequence);
}

}

struct FramePool::State {
    std::shared_ptr<HWAccelDevice> device;
    Layout layout;
    unsigned capacity;
    std::mutex mutex;
    std::shared_ptr<ExportGeneration> generation;
    std::atomic<uint64_t> skipped{0};
};

FramePool::FramePool(std::shared_ptr<HWAccelDevice> device, AVPixelFormat format, int width, int height, unsigned slots)
    : state_(std::make_shared<State>()) {
    if (slots < 2 || slots > max_slots) throw Error("cuda_ipc: pool size must be 2–32");
    state_->device = std::move(device);
    state_->layout = Layout::make(format, width, height);
    state_->capacity = slots;
    deviceUuid(cudaDevice(*state_->device)->cuda_ctx);
}

void FramePool::requireLayout(const std::shared_ptr<HWAccelDevice> &device, AVPixelFormat format,
                              int width, int height, unsigned slots) const {
    const auto &layout = state_->layout;
    if (state_->device != device || layout.format != format || layout.width != width ||
        layout.height != height || state_->capacity != slots)
        throw Error("cuda_ipc: shared VMM pool does not match producer");
}

av::VideoFrame FramePool::acquire() {
    auto &state = *state_;
    std::lock_guard state_guard(state.mutex);
    if (state.generation) {
        const auto previous = state.generation;
        std::lock_guard guard(previous->mutex);
        if (previous->retired) state.generation.reset();
    }
    if (!state.generation) {
        auto pool = std::make_shared<ExportGeneration>();
        pool->device = state.device;
        pool->layout = state.layout;
        pool->frames = pool->layout.frames(*pool->device);
        pool->used.resize(state.capacity);
        for (unsigned i = 0; i < state.capacity; ++i)
            pool->slots.push_back(Memory::allocate(cudaDevice(*pool->device)->cuda_ctx, pool->layout.bytes));
        state.generation = std::move(pool);
    }
    const auto &pool = state.generation;
    size_t slot;
    {
        std::lock_guard guard(pool->mutex);
        const auto free = std::find(pool->used.begin(), pool->used.end(), false);
        if (pool->retired || free == pool->used.end()) {
            ++state.skipped;
            return {};
        }
        slot = free - pool->used.begin();
        *free = true;
    }
    auto *address = pool->slots[slot]->data();
    auto lease = std::make_unique<Lease>(Lease{pool, slot});
    auto *owner = lease.get();
    auto *buffer = av_buffer_create(address, pool->slots[slot]->size(), releaseExport, owner, 0);
    if (!buffer) {
        releaseExport(lease.release(), address);
        throw Error("cuda_ipc: cannot wrap VMM producer frame");
    }
    lease.release();
    AvBufferRef reference(buffer);
    {
        std::lock_guard guard(registry_mutex);
        registry.emplace(address, owner);
    }
    return pool->layout.frame(pool->frames.get(), reference.release());
}

uint64_t FramePool::skipped() const { return state_->skipped.load(); }

} // namespace avp::vmm

using namespace avp::vmm;

class CudaIpcSink : public NodeSingleInput<av::VideoFrame>, public ReportsFinishByFlag, public IReturnsObjects {
    std::shared_ptr<HWAccelDevice> device_;
    DeviceUuid uuid_;
    std::string path_;
    Fd lock_;
    Fd listener_;
    Fd peer_;
    std::shared_ptr<ExportGeneration> pool_;
    std::vector<std::pair<uint64_t, av::VideoFrame>> outstanding_;
    uint64_t generation_ = 0, sequence_ = 0;
    std::atomic<uint64_t> published_{0}, dropped_{0}, released_{0}, disconnects_{0};
    bool bound_ = false;

    void retire() {
        if (peer_) ++disconnects_;
        peer_ = Fd();
        if (pool_) {
            std::lock_guard guard(pool_->mutex);
            pool_->retired = true;
        }
        outstanding_.clear();
        pool_.reset();
    }

    void publish(av::VideoFrame frame, const Lease &lease) {
        const auto &pool = lease.pool;
        {
            std::lock_guard guard(pool->mutex);
            if (pool->retired) return;
        }
        if (pool != pool_) {
            if (pool_) {
                retire();
                return;
            }
            std::vector<Fd> fds;
            for (const auto &slot : pool->slots) fds.emplace_back(slot->exportFd());
            const auto &layout = pool->layout;
            const auto provider = this->findNodeUp<IFrameRateSource>();
            const auto rate = provider ? provider->frameRate().getValue() : AVRational{0, 1};
            if (!transmit(peer_, {{"type", "pool"}, {"version", protocol_version},
                                  {"generation", ++generation_}, {"gpu", uuid_}, {"layout", layout.message()},
                                  {"bytes", pool->slots.front()->size()}, {"fps_num", rate.num}, {"fps_den", rate.den}}, fds)) {
                ++dropped_;
                return;
            }
            pool_ = pool;
            outstanding_.resize(fds.size());
        }
        if (outstanding_[lease.slot].first) { ++dropped_; return; }
        const auto timestamp = frame.pts();
        if (!timestamp.isValid()) throw Error("cuda_ipc_sink: input frame has no PTS");
        const auto *context = reinterpret_cast<const AVHWFramesContext *>(frame.raw()->hw_frames_ctx->data);
        const auto *cuda = static_cast<const AVCUDADeviceContext *>(context->device_ctx->hwctx);
        {
            Context guard(cuda->cuda_ctx);
            if (CHECK_CU(cuStreamSynchronize(cuda->stream)))
                throw Error("cuda_ipc_sink: producer GPU synchronization failed");
        }
        const auto *raw = frame.raw();
        const auto timebase = timestamp.timebase().getValue();
        const uint64_t sequence = ++sequence_;
        if (!transmit(peer_, {{"type", "frame"}, {"generation", generation_}, {"slot", lease.slot},
                              {"sequence", sequence}, {"pts", timestamp.timestamp()}, {"tb_num", timebase.num},
                              {"tb_den", timebase.den}, {"space", raw->colorspace}, {"range", raw->color_range},
                              {"primaries", raw->color_primaries}, {"transfer", raw->color_trc},
                              {"chroma_location", raw->chroma_location}})) {
            ++dropped_;
            return;
        }
        outstanding_[lease.slot] = {sequence, std::move(frame)};
        ++published_;
    }

    void pollPeer() {
        if (!peer_) peer_ = Fd(accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK));
        if (!peer_) return;
        for (size_t i = 0; i < max_slots; ++i) {
            auto msg = receive(peer_);
            if (!msg) return;
            const auto &body = msg->body;
            const size_t slot = body.at("slot").get<size_t>();
            if (!msg->fds.empty() || body.at("type") != "release" || body.at("generation") != generation_ ||
                slot >= outstanding_.size() || outstanding_[slot].first == 0 ||
                outstanding_[slot].first != body.at("sequence").get<uint64_t>())
                throw Error("cuda_ipc_sink: invalid release");
            outstanding_[slot] = {};
            ++released_;
        }
    }

public:
    CudaIpcSink(std::unique_ptr<SourceType> source, std::shared_ptr<HWAccelDevice> device, std::string path)
        : NodeSingleInput(std::move(source)), device_(std::move(device)),
          uuid_(deviceUuid(cudaDevice(*device_)->cuda_ctx)), path_(std::move(path)), listener_(endpoint()) {
        const auto address = socketAddress(path_);
        lock_ = Fd(open((path_ + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600));
        if (!lock_ || flock(lock_.get(), LOCK_EX | LOCK_NB) != 0)
            throw Error("cuda_ipc_sink: socket is already owned: " + path_);
        struct stat existing{};
        if (lstat(path_.c_str(), &existing) == 0 && S_ISSOCK(existing.st_mode)) {
            auto probe = endpoint();
            if (connect(probe.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0 &&
                errno == ECONNREFUSED)
                unlink(path_.c_str());
        }
        if (bind(listener_.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0)
            throw Error("cuda_ipc_sink: bind " + path_ + ": " + std::strerror(errno));
        bound_ = true;
        if (listen(listener_.get(), 1) != 0) {
            unlink(path_.c_str());
            throw Error("cuda_ipc_sink: listen failed");
        }
        auto_eof_ = false;
    }

    ~CudaIpcSink() override {
        retire();
        if (bound_) unlink(path_.c_str());
    }

    void process() override {
        try { pollPeer(); }
        catch (const std::exception &error) { logstream << "cuda_ipc_sink: " << error.what(); retire(); }
        av::VideoFrame frame;
        if (!source_->tryGet(frame, 5)) return;
        if (isEofMarker(frame) || frame.isNull() || !frame.isComplete()) return;
        const auto lease = exportLease(frame);
        if (!lease || lease->pool->device.get() != device_.get())
            throw Error("cuda_ipc_sink requires uncropped frames from the VMM pool on its hwaccel device");
        if (!peer_) { ++dropped_; return; }
        try { publish(std::move(frame), *lease); }
        catch (const std::exception &error) { logstream << "cuda_ipc_sink: " << error.what(); retire(); }
    }

    void stop() override {
        NodeSingleInput::stop();
        finished_ = true;
    }

    Parameters getObject(const std::string key) override {
        if (key != "stats") throw Error("cuda_ipc_sink: unknown object");
        return {{"published", published_.load()}, {"released", released_.load()},
                {"dropped", dropped_.load()}, {"disconnects", disconnects_.load()}};
    }

    static std::shared_ptr<CudaIpcSink> create(NodeCreationInfo &node) {
        return std::make_shared<CudaIpcSink>(
            make_unique<EdgeSource<av::VideoFrame>>(node.edges.find<av::VideoFrame>(node.params.at("src"))),
            InstanceSharedObjects<HWAccelDevice>::get(node.instance, node.params.at("hwaccel")),
            node.params.at("socket"));
    }
};

class CudaIpcSource : public NodeSingleOutput<av::VideoFrame>, public IStoppable, public ReportsFinishByFlag, public IFrameRateSource {
    std::shared_ptr<HWAccelDevice> device_;
    DeviceUuid uuid_;
    sockaddr_un address_;
    Fd peer_;
    std::shared_ptr<ImportGeneration> pool_;
    avp::AvBufferRef output_frames_;
    std::optional<Layout> output_layout_;
    std::atomic<uint64_t> rate_{1};

    void importPool(const Message &msg) {
        const auto &body = msg.body;
        const auto layout = Layout::parse(body.at("layout"));
        const size_t size = body.at("bytes").get<size_t>();
        if (pool_ || body.at("version") != protocol_version || body.at("gpu").get<DeviceUuid>() != uuid_ ||
            !body.at("generation").get<uint64_t>() ||
            msg.fds.size() < 2 || size < layout.bytes || size > max_slot_bytes)
            throw Error("cuda_ipc_source: incompatible pool");
        auto pool = std::make_shared<ImportGeneration>();
        pool->device = device_;
        pool->layout = layout;
        pool->id = body.at("generation").get<uint64_t>();
        pool->released.reserve(msg.fds.size());
        pool->held.resize(msg.fds.size());
        for (const auto &fd : msg.fds)
            pool->slots.push_back(Memory::import(cudaDevice(*device_)->cuda_ctx, fd.get(), size));
        const auto fps_num = body.at("fps_num").get<int>();
        const auto fps_den = body.at("fps_den").get<int>();
        if (fps_num < 0 || fps_den <= 0) throw Error("cuda_ipc_source: invalid frame rate");
        // Same-layout reconnects keep the frames context FFmpeg configured.
        // The AVBuffer holding the mapped pixels still belongs to its generation.
        if (!output_layout_ || output_layout_->width != layout.width ||
            output_layout_->height != layout.height || output_layout_->format != layout.format) {
            output_frames_ = layout.frames(*device_);
            output_layout_ = layout;
        }
        pool->frames.reset(av_buffer_ref(output_frames_.get()));
        if (!pool->frames) throw Error("cuda_ipc_source: cannot retain output frames context");
        rate_.store((uint64_t(fps_num) << 32) | uint32_t(fps_den));
        pool_ = std::move(pool);
    }

    void flushReleases() {
        if (!pool_) return;
        std::lock_guard guard(pool_->mutex);
        while (!pool_->released.empty()) {
            const auto [slot, sequence] = pool_->released.back();
            if (!transmit(peer_, {{"type", "release"}, {"generation", pool_->id},
                                  {"slot", slot}, {"sequence", sequence}}))
                return;
            pool_->held[slot] = false;
            pool_->released.pop_back();
        }
    }

    void receiveFrame(const Message &msg) {
        const auto &body = msg.body;
        if (!pool_ || !msg.fds.empty() || body.at("generation") != pool_->id)
            throw Error("cuda_ipc_source: frame from unknown pool");
        const size_t slot = body.at("slot").get<size_t>();
        const uint64_t sequence = body.at("sequence").get<uint64_t>();
        const int num = body.at("tb_num").get<int>();
        const int den = body.at("tb_den").get<int>();
        if (slot >= pool_->slots.size() || pool_->held[slot] || sequence <= pool_->last_sequence || num <= 0 || den <= 0)
            throw Error("cuda_ipc_source: invalid frame");
        const int64_t pts = body.at("pts").get<int64_t>();
        const auto space = body.at("space").get<AVColorSpace>();
        const auto range = body.at("range").get<AVColorRange>();
        const auto primaries = body.at("primaries").get<AVColorPrimaries>();
        const auto transfer = body.at("transfer").get<AVColorTransferCharacteristic>();
        const auto location = body.at("chroma_location").get<AVChromaLocation>();
        if (pts == AV_NOPTS_VALUE || space < 0 || space >= AVCOL_SPC_NB || range < 0 || range >= AVCOL_RANGE_NB ||
            primaries < 0 || primaries >= AVCOL_PRI_NB || transfer < 0 || transfer >= AVCOL_TRC_NB ||
            location < 0 || location >= AVCHROMA_LOC_NB)
            throw Error("cuda_ipc_source: invalid frame metadata");

        auto lease = std::make_unique<ImportLease>(ImportLease{pool_, slot, sequence});
        const auto &mapping = pool_->slots[slot];
        auto *buffer = av_buffer_create(mapping->data(), mapping->size(), releaseImport, lease.get(),
                                        AV_BUFFER_FLAG_READONLY);
        if (!buffer) throw Error("cuda_ipc_source: cannot wrap imported CUDA frame");
        lease.release();
        pool_->held[slot] = true;
        pool_->last_sequence = sequence;
        auto frame = pool_->layout.frame(pool_->frames.get(), buffer);
        frame.setTimeBase({num, den});
        auto *raw = frame.raw();
        raw->pts = pts;
        raw->colorspace = space;
        raw->color_range = range;
        raw->color_primaries = primaries;
        raw->color_trc = transfer;
        raw->chroma_location = location;
        this->sink_->put(frame, true);
    }

public:
    CudaIpcSource(std::unique_ptr<SinkType> sink, std::shared_ptr<HWAccelDevice> device, std::string path)
        : NodeSingleOutput(std::move(sink)), device_(std::move(device)),
          uuid_(deviceUuid(cudaDevice(*device_)->cuda_ctx)), address_(socketAddress(path)) {}

    void process() override {
        if (!peer_) {
            auto connection = endpoint();
            if (connect(connection.get(), reinterpret_cast<const sockaddr *>(&address_), sizeof(address_)) != 0) {
                wallclock.sleepms(20);
                return;
            }
            peer_ = std::move(connection);
        }
        try {
            flushReleases();
            const auto msg = receive(peer_);
            if (!msg) {
                pollfd fd{peer_.get(), POLLIN, 0};
                poll(&fd, 1, 2);
                return;
            }
            if (msg->body.at("type") == "pool") importPool(*msg);
            else if (msg->body.at("type") == "frame") receiveFrame(*msg);
            else throw Error("cuda_ipc_source: unknown message");
        } catch (const std::exception &error) {
            logstream << "cuda_ipc_source: " << error.what();
            peer_ = Fd();
            pool_.reset();
            wallclock.sleepms(20);
        }
    }

    void stop() override { finished_ = true; }

    av::Rational frameRate() override {
        const uint64_t rate = rate_.load();
        return {int(rate >> 32), int(uint32_t(rate))};
    }

    static std::shared_ptr<CudaIpcSource> create(NodeCreationInfo &node) {
        return std::make_shared<CudaIpcSource>(
            make_unique<EdgeSink<av::VideoFrame>>(node.edges.find<av::VideoFrame>(node.params.at("dst"))),
            InstanceSharedObjects<HWAccelDevice>::get(node.instance, node.params.at("hwaccel")),
            node.params.at("socket"));
    }
};

DECLNODE(cuda_ipc_sink, CudaIpcSink)
DECLNODE(cuda_ipc_source, CudaIpcSource)
