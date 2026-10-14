#include "../node_common.hpp"
#include "../../frame_color_tags.hpp"
#include "CudaPinnedUploader.hpp"
#include "device_memory_wait.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
}

// A headerless raw NV12/P010 file as CUDA frames. Each picture is read from the file straight
// into page-locked memory and the device takes it from there, so the samples are copied once on
// the CPU (page cache to the block) where demuxing, wrapping and a staged upload copy them twice,
// on one thread where that chain has four. Nothing is kept between passes: every picture shown
// was read from the file and transferred for that showing.
//
// `resident` keeps a looped file's pictures in that memory instead: each is read when it is
// first shown and only transferred after that, and the file is read again once it has changed.
// The memory is the size of the file and cannot be swapped out.
class RawVideoFileCuda: public NodeSingleOutput<av::VideoFrame>, public ReportsFinishByFlag, public IStoppable,
                        public IVideoFormatSource, public IFrameRateSource, public ITimeBaseSource,
                        public IReturnsObjects {
protected:
    std::string path_;
    AVPixelFormat sw_format_ = AV_PIX_FMT_NONE;
    int width_ = 0;
    int height_ = 0;
    av::Rational frame_rate_;
    av::Rational time_base_;   // one tick per picture
    bool loop_ = false;
    bool loop_continuous_ts_ = false;
    bool send_eof_ = true;
    bool write_combined_ = false;
    bool resident_ = false;
    int pool_frames_ = 0;
    FrameColorTags color_;
    std::shared_ptr<HWAccelDevice> hwaccel_;

    int fd_ = -1;
    std::unique_ptr<RawFrameIndex> index_;
    std::unique_ptr<CudaPinnedUploader> uploader_;
    uint64_t next_ = 0;   // pictures emitted since start
    std::vector<bool> held_;   // resident: the pictures already in the block
    // How long a picture waits for device memory before the node gives up: long enough for a
    // large show to finish starting, during which every source allocates at once.
    static constexpr int kDeviceMemoryWaitMs = 5000;
    struct timespec modified_{};
    std::atomic<uint64_t> frames_{0}, file_reads_{0}, read_us_{0}, upload_us_{0};

    struct stat fileStatus() {
        struct stat st{};
        if (fstat(fd_, &st) != 0) throw Error("raw_video_file_cuda: fstat " + path_ + ": " + std::strerror(errno));
        return st;
    }
    // A file written since its pictures were read is read again, picture by picture as shown.
    void forgetChangedFile() {
        const struct stat st = fileStatus();
        if (st.st_mtim.tv_sec == modified_.tv_sec && st.st_mtim.tv_nsec == modified_.tv_nsec) return;
        modified_ = st.st_mtim;
        held_.assign(held_.size(), false);
    }
    void readPicture(int64_t offset, size_t slot) {
        uint8_t *block = uploader_->block(slot);
        for (int64_t done = 0; done < index_->frame_size;) {
            const ssize_t r = pread(fd_, block + done, size_t(index_->frame_size - done), offset + done);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) throw Error("raw_video_file_cuda: reading " + path_ + ": " + std::strerror(errno));
            if (r == 0) throw Error("raw_video_file_cuda: " + path_ + " ends inside a picture; has it shrunk?");
            done += r;
        }
        ++file_reads_;
    }

public:
    using NodeSingleOutput::NodeSingleOutput;

    void init(EdgeManager &edges, const Parameters &params) override {
        NodeSingleOutput::init(edges, params);
        fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw Error("raw_video_file_cuda: cannot open " + path_ + ": " + std::strerror(errno));
        const struct stat st = fileStatus();
        modified_ = st.st_mtim;
        try {
            index_ = std::make_unique<RawFrameIndex>(sw_format_, width_, height_, frame_rate_.getValue(), st.st_size);
        } catch (const std::invalid_argument &e) {
            throw Error("raw_video_file_cuda: " + path_ + ": " + e.what());
        }
        if (st.st_size % index_->frame_size)
            logstream << "raw_video_file_cuda: " << path_ << " ends with part of a picture, which is not shown";
        if (resident_) held_.assign(size_t(index_->frames), false);
        uploader_ = std::make_unique<CudaPinnedUploader>(hwaccel_, sw_format_, width_, height_, write_combined_,
                                                         resident_ ? held_.size() : 1, pool_frames_);
    }

    void process() override {
        const int64_t offset = index_->offsetOf(next_, loop_);
        if (offset < 0) {
            if (send_eof_) this->emitEof();
            markFinished();
            return;
        }
        using Clock = std::chrono::steady_clock;
        const auto microseconds = [](Clock::duration d) {
            return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
        };
        const size_t slot = resident_ ? size_t(offset / index_->frame_size) : 0;
        if (resident_ && slot == 0 && next_) forgetChangedFile();
        const auto started = Clock::now();
        if (!resident_ || !held_[slot]) {
            readPicture(offset, slot);
            if (resident_) held_[slot] = true;
        }
        const auto was_read = Clock::now();
        av::VideoFrame frame;
        bool waited = false;
        if (!waitForDeviceMemory([&] { return uploader_->upload(frame, slot); },
                                 [&](int ms) {
                                     if (!std::exchange(waited, true))
                                         logstream << "raw_video_file_cuda: " << path_ << ": no device frame free for a picture, waiting";
                                     std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                                 },
                                 kDeviceMemoryWaitMs, [&] { return bool(this->finished_); }))
            throw Error("raw_video_file_cuda: " + path_ + ": no device frame free for a picture in " +
                        std::to_string(kDeviceMemoryWaitMs / 1000) + " s");
        const auto uploaded = Clock::now();

        AVFrame *raw = frame.raw();
        frame.setTimeBase(time_base_);
        raw->pts = index_->ptsOf(next_, loop_, loop_continuous_ts_);
        raw->duration = 1;
        raw->sample_aspect_ratio = {1, 1};
        raw->flags |= AV_FRAME_FLAG_KEY;
        color_.apply(raw);
        frame.setComplete(true);
        ++next_;
        frames_ = next_;
        read_us_ = microseconds(was_read - started);
        upload_us_ = microseconds(uploaded - was_read);
        // Blocks while the edge is full: the consumer paces the reading.
        this->sink_->put(frame);
    }

    void stop() override { this->finished_ = true; }

    int width() override { return width_; }
    int height() override { return height_; }
    av::PixelFormat pixelFormat() override { return AV_PIX_FMT_CUDA; }
    av::PixelFormat realPixelFormat() override { return sw_format_; }
    av::Rational frameRate() override { return frame_rate_; }
    av::Rational timeBase() override { return time_base_; }

    Parameters getObject(const std::string key) override {
        if (key != "stats") throw Error("raw_video_file_cuda: unknown object " + key);
        const uint64_t frames = frames_.load();
        return {{"frames", frames}, {"passes", index_ ? frames / uint64_t(index_->frames) : 0},
                {"file_reads", file_reads_.load()}, {"frames_allocated", uploader_ ? uploader_->framesAllocated() : 0}, {"read_us", read_us_.load()}, {"upload_us", upload_us_.load()}};
    }

    ~RawVideoFileCuda() {
        uploader_.reset();
        if (fd_ >= 0) close(fd_);
    }

    static std::shared_ptr<RawVideoFileCuda> create(NodeCreationInfo &nci) {
        const Parameters &params = nci.params;
        for (const char *required: {"dst", "url", "pixel_format", "video_size", "framerate", "hwaccel"})
            if (!params.count(required)) throw Error(std::string("raw_video_file_cuda requires ") + required);
        std::shared_ptr<Edge<av::VideoFrame>> dst = nci.edges.find<av::VideoFrame>(params["dst"]);
        auto r = std::make_shared<RawVideoFileCuda>(make_unique<EdgeSink<av::VideoFrame>>(dst));
        r->path_ = params["url"].get<std::string>();
        const std::string format = params["pixel_format"].get<std::string>();
        r->sw_format_ = av_get_pix_fmt(format.c_str());
        if (r->sw_format_ != AV_PIX_FMT_NV12 && r->sw_format_ != AV_PIX_FMT_P010LE)
            throw Error("raw_video_file_cuda: pixel_format must be nv12 or p010le, not " + format);
        const std::string size = params["video_size"].get<std::string>();
        if (av_parse_video_size(&r->width_, &r->height_, size.c_str()) < 0)
            throw Error("raw_video_file_cuda: video_size must be WIDTHxHEIGHT, not " + size);
        r->frame_rate_ = parseRatio(params["framerate"].get<std::string>());
        if (r->frame_rate_.getNumerator() <= 0 || r->frame_rate_.getDenominator() <= 0)
            throw Error("raw_video_file_cuda: framerate must be positive");
        AVRational tick;
        av_reduce(&tick.num, &tick.den, r->frame_rate_.getDenominator(), r->frame_rate_.getNumerator(), INT32_MAX);
        r->time_base_ = tick;
        r->loop_ = params.value("loop", false);
        r->loop_continuous_ts_ = params.value("loop_continuous_ts", false);
        r->send_eof_ = params.value("send_eof", true);
        r->write_combined_ = params.value("write_combined", false);
        r->resident_ = r->loop_ && params.value("resident", false);
        r->pool_frames_ = params.value("pool_frames", 0);
        if (r->pool_frames_ < 0) throw Error("raw_video_file_cuda: pool_frames must not be negative");
        if (params.count("frame_color")) r->color_ = FrameColorTags::parse(params["frame_color"]);
        r->hwaccel_ = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
        if (!r->hwaccel_ || r->hwaccel_->hardwarePixelFormat() != AV_PIX_FMT_CUDA)
            throw Error("raw_video_file_cuda: hwaccel must name a CUDA device");
        return r;
    }
};

DECLNODE(raw_video_file_cuda, RawVideoFileCuda);
