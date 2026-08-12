#pragma once
#include <chrono>
#include <ctime>
#include <thread>
#include <avcpp/timestamp.h>
#include <libavutil/rational.h>
#include <avcpp/frame.h>
#include <avcpp/packet.h>
#include <json.hpp>
#include <avcpp/dictionary.h>
#include "util.hpp"
#include "hwaccel/EglImageFrame.hpp"
#include "metadata_frame.hpp"

typedef int64_t AVTS;
const av::Timestamp NOTS = {AV_NOPTS_VALUE, {0, 1}};
using nlohmann::json;

void silenceAudioFrame(av::AudioSamples &frm, av::SampleFormat::Alignment align = av::SampleFormat::Alignment::AlignDefault);

av::Rational parseRatio(const std::string ratio);

av::Dictionary parametersToDict(const json &params);

std::string mediaTypeToString(AVMediaType mt);
std::string fieldOrderToString(AVFieldOrder fo);
uint64_t stringToChannelLayout(const std::string);

// Parse ISO 8601 date-time string (e.g. "2025-06-15T14:30:00Z" or "2025-06-15T14:30:00+02:00")
// Returns Unix epoch milliseconds. Throws std::runtime_error on parse failure.
int64_t parseIso8601ToMs(const std::string &iso);

class Wallclock {
protected:
    using TimeUnit = std::chrono::milliseconds;
    static constexpr AVRational time_base = {TimeUnit::period::num, TimeUnit::period::den};

    static AVTS monotonicMs() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<AVTS>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
    }

public:
    Wallclock() {
    }
    AVTS pts() {
        return monotonicMs();
    }
    av::Timestamp absolute_ts() {
        return av::Timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), {1, 1000});
    }
    av::Timestamp ts() {
        return av::Timestamp(pts(), timeBase());
    }
    static AVRational timeBase() {
        return time_base;
    }
    static void sleepms(const AVTS ms) {
        if (ms<=0) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    static void sleepAVTS(const AVTS ts) {
        sleepms(ts);
    }
    static AVTS secondsToAVTS(const float sec) {
        return sec*(float)time_base.den / (float)time_base.num + 0.5;
    }
    static void sleep(const av::Timestamp duration) {
        sleepms(duration.timestamp({1, 1000}));
    }
};

extern Wallclock wallclock;


class DiscontinuityDetector {
protected:
    av::Timestamp last_ts_ = NOTS;
public:
    bool check(const av::Timestamp ts) {
        bool r = false;
        if (last_ts_.isValid()) {
            double diff = (ts - last_ts_).seconds();
            if ( (diff<0) || (diff>1) ) {
                r = true;
            }
        }
        last_ts_ = ts;
        return r;
    }
};


namespace {
    av::Timestamp rescaleTS(const av::Timestamp src, const av::Rational dest_tb) {
        if (src.isValid()) {
            av::Timestamp result = { av_rescale_q_rnd(src.timestamp(), src.timebase().getValue(), dest_tb.getValue(), AV_ROUND_NEAR_INF), dest_tb };
            if (fabs(result.seconds() - src.seconds()) > 1.0) {
                logstream << "BAD RESCALE " << src.seconds() << " -> " << result.seconds();
            }
            return result;
        } else {
            return { AV_NOPTS_VALUE, dest_tb };
        }
    }
    /*int lcm(const int a, const int b) {
        return a * b / av_gcd(a, b);
    }*/
    inline av::Timestamp addTSSameTB(const av::Timestamp only) {
        return only;
    }
    template<typename ...Tss> inline av::Timestamp addTSSameTB(const av::Timestamp first, const av::Timestamp second, const Tss ... remainder) {
        av::Timestamp rem_sum = addTSSameTB(second, remainder...);
        assert(rem_sum.timebase() == first.timebase());
        av::Timestamp result = { rem_sum.timestamp() + first.timestamp(), first.timebase() };
        if (fabs(result.seconds() - (first.seconds()+rem_sum.seconds())) > 1.0) {
            logstream << "BAD addTSSameTB: " << first.seconds() << " + " << rem_sum.seconds() << " = " << result.seconds();
        }

        return result;
    }

    inline av::Timestamp addTS(const av::Timestamp only) {
        return only;
    }
    template<typename ...Tss> inline av::Timestamp addTS(const av::Timestamp first, const av::Timestamp second, const Tss ... remainder) {
        av::Timestamp rem_sum = addTS(second, remainder...);
        av::Rational tb = std::min(first.timebase(), rem_sum.timebase());
        av::Timestamp result = { rem_sum.timestamp(tb) + first.timestamp(tb), tb };
        if (fabs(result.seconds() - (first.seconds()+rem_sum.seconds())) > 1.0) {
            logstream << "BAD addTS: " << first.seconds() << " + " << rem_sum.seconds() << " = " << result.seconds();
        }
        return result;
    }
    inline av::Timestamp negateTS(const av::Timestamp ts) {
        av::Timestamp result;
        result = { -ts.timestamp(), ts.timebase() };
        if (fabs(result.seconds()+ts.seconds()) > 1.0) {
            logstream << "BAD negateTS";
        }
        return result;
    }
};

bool isEofMarker(const av::Packet& p);
bool isEofMarker(const av::VideoFrame& f);
bool isEofMarker(const av::AudioSamples& f);
class EglImageFrame; // fwd
bool isEofMarker(const EglImageFrame& f);
bool isEofMarker(const MetadataFrame& f);

av::Packet createEofPacket(int streamIndex = -1);

template<typename T> T createEofMarker() {
    return T();
}
template<> inline av::Packet createEofMarker<av::Packet>() {
    return createEofPacket();
}

template<typename T> struct TSGetter {
};
template<typename T> struct FrameTSGetter {
    static AVTS get(const T& data, const AVRational time_base) {
        return rescaleTS(data.pts(), time_base).timestamp();
    }
    static av::Timestamp getWithTB(const T& data) {
        return data.pts();
    }
    static bool isValid(const T& data) {
        return data.isComplete() && data.pts().isValid();
    }
};
template<> struct TSGetter<av::Packet> {
    static AVTS get(const av::Packet& data, const AVRational time_base) {
        return rescaleTS(data.dts(), time_base).timestamp();
    }
    static av::Timestamp getWithTB(const av::Packet& data) {
        return data.dts();
    }
    static bool isValid(const av::Packet& data) {
        return data.isComplete() && data.dts().isValid();
    }
};
template<> struct TSGetter<av::AudioSamples>: public FrameTSGetter<av::AudioSamples> {
};
template<> struct TSGetter<av::VideoFrame>: public FrameTSGetter<av::VideoFrame> {
};
template<> struct TSGetter<EglImageFrame> {
    static AVTS get(const EglImageFrame& data, const AVRational time_base) {
        return rescaleTS(data.pts(), time_base).timestamp();
    }
    static av::Timestamp getWithTB(const EglImageFrame& data) {
        return data.pts();
    }
    static bool isValid(const EglImageFrame& data) {
        return data.isComplete() && data.pts().isValid();
    }
};
template<> struct TSGetter<MetadataFrame> {
    static AVTS get(const MetadataFrame& data, const AVRational time_base) {
        return rescaleTS(data.pts(), time_base).timestamp();
    }
    static av::Timestamp getWithTB(const MetadataFrame& data) {
        return data.pts();
    }
    static bool isValid(const MetadataFrame& data) {
        return data.isComplete();
    }
};
