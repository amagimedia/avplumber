#pragma once

#include "avutils.hpp"
#include <avcpp/packet.h>
#include <cstdint>

struct FabricPacket {
    av::Packet packet;
    uint64_t stream_id_hash = 0;
    uint32_t replica_id = 0;
    uint64_t generation = 0;
    int64_t raw_pts = AV_NOPTS_VALUE;
    int64_t raw_dts = AV_NOPTS_VALUE;
    int32_t time_base_num = 0;
    int32_t time_base_den = 0;
    uint32_t media_type = 0;
    uint32_t codec = 0;
    uint32_t packet_format = 0;
    uint32_t packet_flags = 0;
    int64_t duration = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pixel_format = 0;
    uint32_t real_pixel_format = 0;
    uint64_t sender_wallclock_ns = 0;
    uint64_t receiver_wallclock_ns = 0;
    bool complete = false;

    av::Timestamp pts() const {
        if (time_base_num <= 0 || time_base_den <= 0 || raw_pts == AV_NOPTS_VALUE) {
            return NOTS;
        }
        return av::Timestamp(raw_pts, av::Rational(time_base_num, time_base_den));
    }

    av::Timestamp dts() const {
        if (time_base_num <= 0 || time_base_den <= 0 || raw_dts == AV_NOPTS_VALUE) {
            return pts();
        }
        return av::Timestamp(raw_dts, av::Rational(time_base_num, time_base_den));
    }

    bool isComplete() const {
        return complete;
    }

    void setTimeBase(av::Rational tb) {
        time_base_num = tb.getNumerator();
        time_base_den = tb.getDenominator();
        packet.setTimeBase(tb);
    }

    void setPts(av::Timestamp ts) {
        if (ts.isValid()) {
            raw_pts = ts.timestamp();
            time_base_num = ts.timebase().getNumerator();
            time_base_den = ts.timebase().getDenominator();
        } else {
            raw_pts = AV_NOPTS_VALUE;
        }
        packet.setPts(ts);
    }
};

inline bool isEofMarker(const FabricPacket &p) {
    return !p.complete && p.packet.pts().isNoPts();
}

template<> struct TSGetter<FabricPacket>: public FrameTSGetter<FabricPacket> {
};
