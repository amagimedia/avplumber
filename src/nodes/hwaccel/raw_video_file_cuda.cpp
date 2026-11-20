#include "../node_common.hpp"
#include "../../frame_color_tags.hpp"
#include "CudaPinnedUploader.hpp"
#include "device_memory_wait.hpp"
#include "mxl_flow_reader.hpp"

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
// `pixel_format` v210 is packed 10-bit 4:2:2: the device takes the packed picture and unpacks it
// into `storage` there. Its pictures also come from a live MXL flow (`url` mxl://...), each grain
// copied once out of the flow's ring into that same memory; see mxl_flow_reader.hpp for what a
// stalled, lapping or restarted writer does.
//
// `resident` keeps a looped file's pictures in that memory instead: each is read when it is
// first shown and only transferred after that, and the file is read again once it has changed.
// The memory is the size of the file and cannot be swapped out.
class RawVideoFileCuda: public NodeSingleOutput<av::VideoFrame>, public ReportsFinishByFlag, public IStoppable,
                        public IVideoFormatSource, public IFrameRateSource, public ITimeBaseSource,
                        public IReturnsObjects {
protected:
    std::string path_;
    AVPixelFormat sw_format_ = AV_PIX_FMT_NONE;   // of the frames: v210 is unpacked into it
    bool v210_ = false;
    int field_flags_ = 0;   // AV_FRAME_FLAG_INTERLACED and the field order, for woven fields
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
    std::unique_ptr<MxlFlowReader> flow_;   // the pictures' source instead of the file
    std::unique_ptr<CudaPinnedUploader> uploader_;
    uint64_t next_ = 0;   // pictures emitted since start
    std::vector<bool> held_;   // resident: the pictures already in the block
    // A picture waits for a device frame for as long as it takes: the consumers hold this
    // source's frames, and while they do its last picture stays on air, which is what a source
    // short of frames looks like. Giving up instead would end the node, and a restart tears
    // down and makes again what every thread using the device then waits for. The wait is
    // logged this often, and counted for the operator.
    static constexpr int kDeviceMemoryLogEveryMs = 5000;
    struct timespec modified_{};
    std::atomic<uint64_t> frames_{0}, file_reads_{0}, read_us_{0}, upload_us_{0};
    std::atomic<uint64_t> frame_waits_{0}, longest_wait_ms_{0};
    std::atomic<bool> waiting_{false};
    std::atomic<int64_t> due_us_{0}, due_max_us_{INT64_MIN};   // MXL: a picture's time after its grain's

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
        const auto makeUploader = [&](size_t pictures) {
            uploader_ = std::make_unique<CudaPinnedUploader>(hwaccel_, sw_format_, width_, height_, write_combined_,
                                                             pictures, pool_frames_, v210_, field_flags_ != 0);
        };
        if (isMxlUrl(path_)) {
            flow_ = std::make_unique<MxlFlowReader>(path_, width_, height_);
            // Timestamps are the grains' indices, at the flow's own rate whatever `framerate` says.
            frame_rate_ = flow_->frameRate();
            time_base_ = av::Rational(frame_rate_.getDenominator(), frame_rate_.getNumerator());
            return makeUploader(1);
        }
        fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw Error("raw_video_file_cuda: cannot open " + path_ + ": " + std::strerror(errno));
        const struct stat st = fileStatus();
        modified_ = st.st_mtim;
        try {
            index_ = std::make_unique<RawFrameIndex>(v210_ ? AV_PIX_FMT_GRAY8 : sw_format_, v210_ ? v210Stride(width_) : width_,
                                                     height_, frame_rate_.getValue(), st.st_size);
        } catch (const std::invalid_argument &e) {
            throw Error("raw_video_file_cuda: " + path_ + ": " + e.what());
        }
        if (st.st_size % index_->frame_size)
            logstream << "raw_video_file_cuda: " << path_ << " ends with part of a picture, which is not shown";
        if (resident_) held_.assign(size_t(index_->frames), false);
        makeUploader(resident_ ? held_.size() : 1);
    }

    void process() override {
        using Clock = std::chrono::steady_clock;
        const auto microseconds = [](Clock::duration d) {
            return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
        };
        size_t slot = 0;
        int64_t pts = 0;
        auto started = Clock::now();
        if (flow_) {
            // `started` becomes the start of the copy: the wait for a grain is not reading.
            if (!flow_->read(uploader_->block(), pts, started)) return;   // no grain yet, or one to leave out
        } else {
            const int64_t offset = index_->offsetOf(next_, loop_);
            if (offset < 0) {
                if (send_eof_) this->emitEof();
                markFinished();
                return;
            }
            if (resident_) slot = size_t(offset / index_->frame_size);
            if (resident_ && slot == 0 && next_) forgetChangedFile();
            if (!resident_ || !held_[slot]) {
                readPicture(offset, slot);
                if (resident_) held_[slot] = true;
            }
            pts = index_->ptsOf(next_, loop_, loop_continuous_ts_);
        }
        const auto was_read = Clock::now();
        av::VideoFrame frame;
        int waited_ms = 0;
        const auto attempt = [&] { return uploader_->upload(frame, slot); };
        const auto sleep = [&](int ms) {
            if (!waiting_.exchange(true)) {
                ++frame_waits_;
                logstream << "raw_video_file_cuda: " << path_ << ": no device frame free for a picture, waiting"
                          << " (consumers hold all " << uploader_->framesAllocated() << "; wait " << frame_waits_.load() << ")";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        };
        const auto stopping = [&] { return bool(this->finished_); };
        while (!waitForDeviceMemory(attempt, sleep, kDeviceMemoryLogEveryMs, stopping)) {
            if (stopping()) return;
            waited_ms += kDeviceMemoryLogEveryMs;
            logstream << "raw_video_file_cuda: " << path_ << ": still no device frame free after " << waited_ms / 1000
                      << " s; its last picture stays on air until one comes back";
        }
        if (waiting_.exchange(false)) {
            const auto ms = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - was_read).count());
            if (ms > longest_wait_ms_.load()) longest_wait_ms_ = ms;
            logstream << "raw_video_file_cuda: " << path_ << ": device frame free again after " << ms << " ms";
        }
        const auto uploaded = Clock::now();

        AVFrame *raw = frame.raw();
        frame.setTimeBase(time_base_);
        raw->pts = pts;
        raw->duration = 1;
        raw->sample_aspect_ratio = {1, 1};
        raw->flags |= AV_FRAME_FLAG_KEY | field_flags_;
        color_.apply(raw);
        frame.setComplete(true);
        ++next_;
        frames_ = next_;
        read_us_ = microseconds(was_read - started);
        upload_us_ = microseconds(uploaded - was_read);
        if (flow_) {
            due_us_ = flow_->sinceDueNs(pts) / 1000;
            if (due_us_.load() > due_max_us_.load()) due_max_us_ = due_us_.load();
        }
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
        Parameters stats = {{"frames", frames}, {"passes", index_ ? frames / uint64_t(index_->frames) : 0},
                {"file_reads", file_reads_.load()}, {"frames_allocated", uploader_ ? uploader_->framesAllocated() : 0},
                {"read_us", read_us_.load()}, {"upload_us", upload_us_.load()},
                // Device-frame shortages: how often a picture waited for one, whether one waits
                // now (the source repeats its last picture), and the longest wait.
                {"frame_waits", frame_waits_.load()}, {"waiting", waiting_.load()}, {"longest_wait_ms", longest_wait_ms_.load()}};
        if (flow_) {
            stats.update(flow_->stats());
            // How long after its grain was due (its index on the flow's clock, TAI) the last
            // picture went out, and the longest since the last read of these stats: what
            // `realtime`'s tai_delay must cover for pictures to be released on time.
            const int64_t longest = due_max_us_.exchange(INT64_MIN);
            stats.update({{"due_us", due_us_.load()}, {"due_max_us", longest == INT64_MIN ? due_us_.load() : longest}});
        }
        return stats;
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
        r->v210_ = format == "v210";
        const std::string storage = r->v210_ ? params.value("storage", "p210le") : format;
        r->sw_format_ = av_get_pix_fmt(storage.c_str());
        if (r->sw_format_ != AV_PIX_FMT_NV12 && r->sw_format_ != AV_PIX_FMT_P010LE &&
            !(r->v210_ && r->sw_format_ == AV_PIX_FMT_P210LE))
            throw Error("raw_video_file_cuda: pixel_format must be nv12, p010le or v210 (storage p210le, p010le or nv12), not " +
                        format + (r->v210_ ? " to " + storage : ""));
        if (isMxlUrl(r->path_) && !r->v210_)
            throw Error("raw_video_file_cuda: an MXL flow is v210; pixel_format must say so");
        const std::string fields = params.value("field_order", "progressive");
        if (fields != "progressive" && fields != "tff" && fields != "bff")
            throw Error("raw_video_file_cuda: field_order must be progressive, tff or bff, not " + fields);
        if (fields != "progressive")
            r->field_flags_ = AV_FRAME_FLAG_INTERLACED | (fields == "tff" ? AV_FRAME_FLAG_TOP_FIELD_FIRST : 0);
        const std::string size = params["video_size"].get<std::string>();
        if (av_parse_video_size(&r->width_, &r->height_, size.c_str()) < 0 || (r->v210_ && r->width_ % 2))
            throw Error("raw_video_file_cuda: video_size must be WIDTHxHEIGHT (an even width for v210), not " + size);
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
