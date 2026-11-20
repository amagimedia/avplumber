#pragma once
// One MXL video flow (v210 grains in a shared-memory ring, https://github.com/dmf-mxl/mxl) read
// grain by grain into memory of the caller, who can page-lock it: CUDA does not register the
// ring's own read-only shared mapping, so one copy on the CPU is the least a grain costs.
//
// It behaves as FFmpeg's mxl demuxer does with `blocking=1:grain_index_init=head:
// on_too_late=reset:reset_on_drop=1` under `input_rec`:
// - it starts at the newest grain and waits one grain period at a time for the next, for as
//   long as a stalled writer takes (logged when a second has passed, and when grains return);
// - a reader that has fallen out of the ring goes back to the newest grain, and timestamps run
//   on from the last picture instead of jumping;
// - a flow that is gone or was created again, or any other failed read, is an Error: the node
//   finishes with it and its `auto_restart` opens the flow again.
// Unlike the demuxer:
// - a grain the writer overwrote while it was being copied is dropped;
// - a grain the writer left out or flagged invalid is passed over with its timestamp, not asked
//   for again until it is too late;
// - a writer that started again below the reader's index is noticed, and the reader goes to its
//   newest grain;
// - whether the writer is alive is not asked (mxlIsFlowActive): a flow whose files remain after
//   its writer died is waited on, logged once, and not opened again every second;
// - a flow that is not progressive video/v210 of the picture size is an Error when it is opened.
#include "../../util.hpp"
#include "raw_frame_index.hpp"

#include <avcpp/rational.h>

#include <chrono>

inline bool isMxlUrl(const std::string &url) { return url.rfind("mxl://", 0) == 0; }

#if HAVE_MXL
#include <mxl/flow.h>
#include <mxl/mxl.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

extern "C" {
#include <libavutil/mathematics.h>
}

class MxlFlowReader {
    // One libmxl instance per domain for the whole process, shared by its readers: every
    // instance holds an inotify instance, of which a user has 128 by default. libmxl guards an
    // instance's readers with a mutex, and each reader here is used by one thread.
    static std::shared_ptr<mxlInstance_t> instanceOf(const std::string &domain) {
        static std::mutex mutex;
        static std::map<std::string, std::weak_ptr<mxlInstance_t>> instances;
        const std::lock_guard<std::mutex> lock(mutex);
        std::shared_ptr<mxlInstance_t> instance = instances[domain].lock();
        if (!instance) {
            mxlInstance made = mxlCreateInstance(domain.c_str(), nullptr);
            if (!made) throw Error("mxl: cannot open the domain " + domain);
            instances[domain] = instance = std::shared_ptr<mxlInstance_t>(made, mxlDestroyInstance);
        }
        return instance;
    }

    std::string flow_;
    std::shared_ptr<mxlFlowReader_t> reader_;   // its deleter holds the instance
    size_t grain_bytes_ = 0;
    av::Rational rate_;
    uint64_t wait_ns_ = 0;     // one grain period
    uint64_t index_ = 0;       // the next grain to read
    uint64_t origin_ = 0;      // the grain index of timestamp zero
    uint64_t idle_ = 0;        // waits in a row that brought no grain
    uint64_t idle_logged_ = 1; // as many waits as make a second
    std::atomic<uint64_t> lapped_{0}, resets_{0}, skipped_{0};

    [[noreturn]] void fail(const std::string &what, mxlStatus status) const {
        throw Error("mxl flow " + flow_ + ": " + what + " (libmxl status " + std::to_string(int(status)) + ")");
    }
    uint64_t head() const {
        mxlFlowRuntimeInfo now{};
        if (const mxlStatus status = mxlFlowReaderGetRuntimeInfo(reader_.get(), &now); status != MXL_STATUS_OK)
            fail("cannot read its head index", status);
        return now.headIndex;
    }
    // On to the newest grain, which gets the timestamp the grain left behind would have had.
    void jumpTo(uint64_t newest) {
        origin_ += newest - index_;
        index_ = newest;
        ++resets_;
    }
    // The grain asked for, complete: a writer flags the grains it skips as invalid, and stamps
    // a slot with its new index before writing it. libmxl stores that index without a barrier,
    // so seeing it before the payload written after it relies on x86 keeping the order of stores.
    bool whole(const mxlGrainInfo &info) const {
        return info.index == index_ && !(info.flags & MXL_GRAIN_FLAG_INVALID) && info.validSlices == info.totalSlices;
    }

public:
    /// `url`: `mxl://<domain directory>?id=<video flow id>`, as the demuxer takes it. Throws
    /// unless that flow exists now and is progressive v210 of this picture size in host memory.
    MxlFlowReader(const std::string &url, int width, int height): grain_bytes_(size_t(v210Stride(width)) * size_t(height)) {
        const size_t id = url.find("?id=");
        if (!isMxlUrl(url) || id == std::string::npos)
            throw Error("mxl: the url must be mxl://<domain directory>?id=<video flow id>, not " + url);
        flow_ = url.substr(id + 4);
        const std::shared_ptr<mxlInstance_t> instance = instanceOf(url.substr(6, id - 6));
        mxlFlowReader reader = nullptr;
        if (const mxlStatus status = mxlCreateFlowReader(instance.get(), flow_.c_str(), "", &reader); status != MXL_STATUS_OK)
            fail("cannot open it", status);
        reader_ = std::shared_ptr<mxlFlowReader_t>(reader, [instance](mxlFlowReader r) { mxlReleaseFlowReader(instance.get(), r); });

        size_t length = 0;
        mxlGetFlowDef(instance.get(), flow_.c_str(), nullptr, &length);   // answers with the length
        std::string text(length, '\0');
        if (const mxlStatus status = mxlGetFlowDef(instance.get(), flow_.c_str(), text.data(), &length); status != MXL_STATUS_OK)
            fail("cannot read its definition", status);
        const Parameters def = Parameters::parse(text.c_str(), nullptr, false);
        mxlFlowConfigInfo config{};
        if (const mxlStatus status = mxlFlowReaderGetConfigInfo(reader, &config); status != MXL_STATUS_OK)
            fail("cannot read its configuration", status);
        const mxlRational &rate = config.common.grainRate;
        // An interlaced flow has a grain per field, of half the rows at twice the rate, and
        // nothing in the SDK says which grain index is the top field: it is refused, not woven.
        if (!def.is_object() || def.value("media_type", "") != "video/v210" ||
            def.value("interlace_mode", "progressive") != "progressive" ||
            def.value("frame_width", 0) != width || def.value("frame_height", 0) != height ||
            config.common.format != MXL_DATA_FORMAT_VIDEO || config.common.payloadLocation != MXL_PAYLOAD_LOCATION_HOST_MEMORY ||
            // With the picture size this is the size of every grain's payload, which libmxl
            // allocates from the definition; a grain's own grainSize is what its writer says.
            config.discrete.sliceSizes[0] != uint32_t(v210Stride(width)) ||
            rate.numerator <= 0 || rate.denominator <= 0)
            throw Error("mxl flow " + flow_ + ": only progressive video/v210 of the source's " + std::to_string(width) + "x" +
                        std::to_string(height) + " in host memory is read; its definition is " + text.c_str());
        rate_ = av::Rational(int(rate.numerator), int(rate.denominator));
        wait_ns_ = uint64_t(av_rescale(1'000'000'000, rate.denominator, rate.numerator));
        idle_logged_ = uint64_t(std::max<int64_t>(1, rate.numerator / rate.denominator));
        // The grain after the newest: the newest may have been written up to a period ago, and
        // pacing downstream takes the first picture's arrival as the timeline of all that follow,
        // which then arrive later than it and repeat and drop pairs on the output ticks.
        index_ = origin_ = head() + 1;
    }

    av::Rational frameRate() const { return rate_; }

    /// The next grain into `block` (a v210 picture at v210Stride() bytes a row) and its
    /// timestamp, in grain periods; `copied_from` is when the wait for it ended and the copy
    /// began. False: no picture this time, and the caller calls again; that returns within a
    /// grain period, so a node stays stoppable while a writer stalls.
    bool read(uint8_t *block, int64_t &pts, std::chrono::steady_clock::time_point &copied_from) {
        mxlGrainInfo info{};
        uint8_t *payload = nullptr;
        switch (const mxlStatus status = mxlFlowReaderGetGrain(reader_.get(), index_, wait_ns_, &info, &payload)) {
        case MXL_STATUS_OK:
            break;
        case MXL_ERR_OUT_OF_RANGE_TOO_LATE:
            jumpTo(head());
            return false;
        case MXL_ERR_OUT_OF_RANGE_TOO_EARLY:
        case MXL_ERR_TIMEOUT:
            // Usually the writer is merely late. One that started again below this index would
            // never reach it.
            if (const uint64_t newest = head(); newest + 1 < index_)
                jumpTo(newest);
            else if (++idle_ == idle_logged_)
                logstream << "mxl flow " << flow_ << ": no grain for a second, waiting for its writer";
            return false;
        default:   // MXL_ERR_FLOW_INVALID: the writer has made the flow again
            // libmxl hands every reader of a flow in an instance the same mapping, the stale one
            // for as long as any of them holds it: let go of it before the restart opens the flow.
            // A second node on this flow that is blocked on a full edge still holds it, and the
            // restart fails again until that node has read and failed as well.
            reader_.reset();
            fail("cannot read a grain; the flow has to be opened again", status);
        }
        if (!whole(info)) {
            ++skipped_;
            ++index_;   // its time has passed all the same
            return false;
        }
        copied_from = std::chrono::steady_clock::now();
        std::memcpy(block, payload, grain_bytes_);
        // The writer never waits for a reader: it may have started on this slot during the copy.
        // libmxl answers "too late" from the moment the slot is the writer's next one, which is
        // before the writer touches it, so a grain it still returns whole was copied intact.
        if (mxlFlowReaderGetGrainNonBlocking(reader_.get(), index_, &info, &payload) != MXL_STATUS_OK || !whole(info)) {
            ++lapped_;
            jumpTo(head());
            return false;
        }
        if (idle_ >= idle_logged_) logstream << "mxl flow " << flow_ << ": grains again";
        idle_ = 0;
        pts = int64_t(index_++ - origin_);
        return true;
    }

    /// `lapped`: grains dropped because the writer overwrote them during the copy; `resets`:
    /// jumps to the newest grain (those, a reader fallen out of the ring, a writer that started
    /// over); `skipped`: grains the writer left out or flagged invalid.
    Parameters stats() const {
        return {{"lapped", lapped_.load()}, {"resets", resets_.load()}, {"skipped", skipped_.load()}};
    }
};
#else
// A build without libmxl (HAVE_MXL=1 links it) has the type, and no flow to read.
struct MxlFlowReader {
    MxlFlowReader(const std::string &, int, int) {
        throw Error("mxl: this build has no libmxl (make HAVE_MXL=1); FFmpeg's mxl demuxer can read the flow instead "
                    "(the mixer's canvas.mxl_reader \"demuxer\")");
    }
    av::Rational frameRate() const { return {}; }
    bool read(uint8_t *, int64_t &, std::chrono::steady_clock::time_point &) { return false; }
    Parameters stats() const { return {}; }
};
#endif
