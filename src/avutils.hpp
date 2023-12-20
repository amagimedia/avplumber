#pragma once
#include <chrono>
#include <thread>
#include <avcpp/timestamp.h>
#include <libavutil/rational.h>
#include <avcpp/frame.h>
#include <avcpp/packet.h>
#include <json.hpp>
#include <avcpp/dictionary.h>
#include "util.hpp"

typedef int64_t AVTS;
const av::Timestamp NOTS = {AV_NOPTS_VALUE, {0, 1}};
using nlohmann::json;

template<typename T> struct TSGetter {
};
template<typename T> struct FrameTSGetter {
    static AVTS get(const T& data, const AVRational time_base) {
        return data.pts().timestamp(time_base);
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
        return data.dts().timestamp(time_base);
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

void silenceAudioFrame(av::AudioSamples &frm, av::SampleFormat::Alignment align = av::SampleFormat::Alignment::AlignDefault);

av::Rational parseRatio(const std::string ratio);

av::Dictionary parametersToDict(const json &params);

std::string mediaTypeToString(AVMediaType mt);
std::string fieldOrderToString(AVFieldOrder fo);

class Wallclock {
protected:
    using Clock = std::chrono::steady_clock;
    std::chrono::time_point<Clock> start;
public:
    using TimeUnit = std::chrono::milliseconds;
protected:
    static constexpr AVRational time_base = {TimeUnit::period::num, TimeUnit::period::den};
public:
    Wallclock() {
        start = Clock::now();
    }
    AVTS pts() {
        return std::chrono::duration_cast<TimeUnit>(Clock::now()-start).count();
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
