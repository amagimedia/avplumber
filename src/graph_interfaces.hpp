#pragma once
#include "avutils.hpp"
#include "util.hpp"
#include <avcpp/pixelformat.h>
#include <avcpp/codec.h>
#include <avcpp/formatcontext.h>
#include "video_parameters.hpp"
#include "audio_parameters.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

class IReportsFinish {
public:
    virtual bool finished() = 0;
};

class IStoppable {
public:
    virtual void stop() = 0;
};

class IInterruptible {
public:
    virtual void interrupt() = 0;
};

class EdgeManager;

class IInitAfterCreate {
public:
    virtual void init(EdgeManager &edges, const Parameters &params) = 0;
};

class IFlushable {
public:
    virtual void flush() = 0;
};

class IWaitsSinksEmpty {
public:
    virtual void waitSinksEmpty() = 0;
    virtual void stopSinks() = 0;
};

struct StreamTarget {
    enum class ETargetType {
        tt_Timestamp,
        tt_Wallclock,
        tt_Bytes,
        tt_Live,
        tt_Stop
    };
    av::Timestamp ts = NOTS;
    size_t bytes = 0;
    ETargetType type = ETargetType::tt_Timestamp;

    static StreamTarget from_timestamp(av::Timestamp ts) {
        return { ts: ts, bytes: 0, type: ETargetType::tt_Timestamp };
    }
    static StreamTarget from_wallclock(const std::string& s) {
        // format: ISO8601: YYYY-MM-DDTHH:MM:SS:mmm
        std::tm tm = {};
        std::istringstream ss(s);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
        int64_t t = timegm(&tm) * 1000;
        size_t ms_sep = s.find('.');
        if (ms_sep != std::string::npos) {
            const char* ms_str = s.c_str() + ms_sep + 1;
            int mmm = atoi(ms_str);
            size_t l = strlen(ms_str);
            // convert from values with more than 3 digits
            while (l > 3) {
                --l;
                mmm /= 10;
            }
            // convert from values with less than 3 digits
            while (l < 3) {
                ++l;
                mmm *= 10;
            }
            t += mmm;
        }

        return { ts: av::Timestamp(t, {1, 1000}), bytes: 0, type: ETargetType::tt_Wallclock };
    }
    static StreamTarget from_string(std::string& s) {
        if (s.find('T') != std::string::npos) {
            return StreamTarget::from_wallclock(s);
        }
        size_t t_sep = s.find(':');
        if (t_sep != std::string::npos) {
            // HH:MM:SS.mmm value
            int hh, mm, ss = -1, mmm = 0;
            int res = sscanf(s.c_str(), "%d:%d:%d", &hh, &mm, &ss);
            if (ss < 0) {
                // mm:ss format
                ss = mm;
                mm = hh;
                hh = 0;
            }
            size_t ms_sep = s.find('.', t_sep + 1);
            if (ms_sep != std::string::npos) {
                const char* ms_str = s.c_str() + ms_sep + 1;
                mmm = atoi(s.c_str() + ms_sep + 1);
                size_t l = strlen(ms_str);
                // convert from values with more than 3 digits
                while (l > 3) {
                    --l;
                    mmm /= 10;
                }
                // convert from values with less than 3 digits
                while (l < 3) {
                    ++l;
                    mmm *= 10;
                }
            }
            int64_t ts = (((hh * 60) + mm) * 60 + ss) * 1000 + mmm;
            return StreamTarget::from_timestamp(av::Timestamp(ts, {1, 1000}));
        }
        // just a number (timestamp expressed in ms)
        return StreamTarget::from_timestamp(av::Timestamp(std::atoll(s.c_str()), {1, 1000}));
    }
    static StreamTarget live() {
        return { type: ETargetType::tt_Live };
    }
    static StreamTarget stop() {
        return { type: ETargetType::tt_Stop };
    }
    bool isLive() {
        return type == ETargetType::tt_Live;
    }
    bool isBytes() {
        return type == ETargetType::tt_Bytes;
    }
    bool isStop() {
        return type == ETargetType::tt_Stop;
    }
    bool isWallclock() {
        return type == ETargetType::tt_Wallclock;
    }
    bool isTimestamp() {
        return type == ETargetType::tt_Timestamp;
    }
};

class IStreamsInput {
public:
    virtual size_t streamsCount() = 0;
    virtual av::Stream stream(size_t) = 0;
    virtual void discardAllStreams() = 0;
    virtual void enableStream(size_t) = 0;
    virtual av::FormatContext& formatContext() = 0;
    virtual void seekAndPause(StreamTarget target) = 0;
    virtual void resumeAfterSeek() = 0;
    virtual void fixInputTimestamp(StreamTarget& ts) = 0;
    virtual void setFrameMetadataTimestamps(av::VideoFrame& frame) = 0;
};

class IFlushAndSeek {
public:
    virtual void flushAndSeek(StreamTarget target) = 0;
};

class ISeekAt {
public:
    virtual void seekAtAdd(const StreamTarget& when, const StreamTarget& target) = 0;
    virtual void seekAtClear() = 0;
};

class IInputReset {
public:
    virtual void resetInput() = 0;
};

class INeedsOutputFrameSize {
public:
    virtual void setOutputFrameSize(const size_t size) = 0;
};

class IEncoder {
public:
    virtual av::Codec& encodingCodec() = 0;
    virtual AVCodecParameters* codecParameters() = 0;
    virtual void setOutput(av::Stream &stream, av::FormatContext &octx) = 0;
    virtual void setOutputPostOpen(av::Stream &stream, av::FormatContext &octx) { /* noop */ };
    virtual void openEncoder(av::Stream stream = av::Stream()) { /* noop */ };
};

class IDecoder {
public:
    virtual std::string codecName() const = 0;
    virtual std::string codecMediaTypeString() const = 0;
    virtual std::string fieldOrderString() const = 0;
    virtual void discardUntil(av::Timestamp pts) = 0;
};

class IMuxer {
public:
    virtual void initFromFormatContext(av::FormatContext &octx) = 0;
    virtual void initFromFormatContextPostOpenPreWriteHeader(av::FormatContext &octx) = 0;
    virtual void initFromFormatContextPostOpen(av::FormatContext &octx) { /* noop */ };
};

// interface which returns current parameters of video
// decoder and scale filter should implement it.

class IVideoFormatSource {
public:
    virtual int width() = 0;
    virtual int height() = 0;
    virtual av::PixelFormat pixelFormat() = 0;
    virtual av::PixelFormat realPixelFormat() { return pixelFormat(); };
    VideoParameters videoParameters() {
        VideoParameters r;
        r.width = width();
        r.height = height();
        r.pixel_format = pixelFormat();
        r.real_pixel_format = realPixelFormat();
        return r;
    }
};
class IFrameRateSource {
public:
    virtual av::Rational frameRate() = 0;
};


class IAudioMetadataSource {
public:
    virtual int sampleRate() = 0;
    virtual av::SampleFormat sampleFormat() = 0;
    virtual uint64_t channelLayout() = 0;
    AudioParameters audioParameters() {
        AudioParameters r;
        r.sample_rate = sampleRate();
        r.sample_format = sampleFormat();
        r.channel_layout = channelLayout();
        return r;
    }
};

class ITimeBaseSource {
public:
    virtual av::Rational timeBase() = 0;
};

class ISentinel {
public:
    // returns:
    //  bool - there is currently card instead of main input;
    //  uint64_t - timestamp of last card switch, milliseconds in output clock
    virtual std::pair<bool, uint64_t> getCardStatus() = 0;
};

class IReturnsObjects {
public:
    virtual Parameters getObject(const std::string) = 0;
};

class IPreferredFormatReceiver {
#define WARN_NOT_OVERRIDEN { logstream << "Warning: Called NOOP " << __func__ << " which should be overriden."; }
public:
    virtual void setPreferredPixelFormat(av::PixelFormat) WARN_NOT_OVERRIDEN;
    virtual void setPreferredResolution(int width, int height) WARN_NOT_OVERRIDEN;
#undef WARN_NOT_OVERRIDEN
};

class IJackSink {
public:
    virtual void jack_process(size_t nframes) = 0;  
};

struct EdgeMetadata {
    virtual ~EdgeMetadata() {
    }
};

struct InputStreamMetadata: public EdgeMetadata {
    av::Stream source_stream;
};

#pragma GCC diagnostic pop
