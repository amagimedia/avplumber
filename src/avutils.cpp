#include "avutils.hpp"
#include "util.hpp"
#include <cstring>
#include <cctype>
#include <avcpp/channellayout.h>
#include "hwaccel/EglImageFrame.hpp"

void silenceAudioFrame(av::AudioSamples &frm, av::SampleFormat::Alignment align) {
    if (frm.sampleFormat().isPlanar()) {
        size_t size1ch = frm.sampleFormat().requiredBufferSize(1, frm.samplesCount(), align);
        for (size_t i=0; i<frm.channelsCount(); i++) {
            uint8_t* ptr = frm.data(i);
            std::fill(ptr, ptr+size1ch, 0);
        }
    } else {
        size_t size = frm.sampleFormat().requiredBufferSize(frm.channelsCount(), frm.samplesCount(), align);
        uint8_t* ptr = frm.data(0);
        std::fill(ptr, ptr+size, 0);
    }
    frm.setComplete(true);
}

av::Rational parseRatio(const std::string ratio) {
    AVRational r;
    if (av_parse_ratio(&r, ratio.c_str(), 1<<24, AV_LOG_MAX_OFFSET, nullptr) < 0) {
        throw Error("Invalid ratio " + ratio);
    }
    return {r};
}

constexpr AVRational Wallclock::time_base;

Wallclock wallclock;

av::Dictionary parametersToDict(const json &params) {
    av::Dictionary r;
    if (params.empty()) return r;
    if (!params.is_object()) {
        throw Error("Dictionary { \"key\": \"value\" [ , ... ] } excepted");
    }
    for (json::const_iterator it = params.begin(); it != params.end(); ++it) {
        std::string value;
        if (it.value().is_string()) {
            value = it.value().get<std::string>();
        } else {
            // convert to string
            std::stringstream v;
            v << it.value();
            value = v.str();
        }
        r.set(it.key(), value);
    }
    return r;
}

std::string mediaTypeToString(AVMediaType mt) {
    if (mt==AVMEDIA_TYPE_VIDEO) {
        return "V";
    } else if (mt==AVMEDIA_TYPE_AUDIO) {
        return "A";
    }
    return "?";
}

std::string fieldOrderToString(AVFieldOrder fo) {
    std::string r;
    if (fo==AV_FIELD_PROGRESSIVE) r = "PROGRESSIVE";
    if (fo==AV_FIELD_TT) r = "TT";
    if (fo==AV_FIELD_BB) r = "BB";
    if (fo==AV_FIELD_TB) r = "TB";
    if (fo==AV_FIELD_BT) r = "BT";
    return r;
}

uint64_t stringToChannelLayout(const std::string s) {
#if API_NEW_CHANNEL_LAYOUT
    return av::ChannelLayout(s.c_str()).layout();
#else
    return av_get_channel_layout(s.c_str());
#endif
}

int64_t parseIso8601ToMs(const std::string &iso) {
    struct tm tm;
    memset(&tm, 0, sizeof(tm));

    const char *fmt = "%Y-%m-%dT%H:%M:%S";
    const char *ptr = strptime(iso.c_str(), fmt, &tm);
    if (!ptr) {
        throw std::runtime_error("parseIso8601ToMs: cannot parse ISO 8601 date string: " + iso);
    }

    int64_t tz_offset = 0;
    if (*ptr == 'Z' || *ptr == '\0') {
        if (*ptr == 'Z' && *(ptr + 1) != '\0') {
            throw std::runtime_error("parseIso8601ToMs: trailing characters after Z: " + iso);
        }
    } else if (*ptr == '+' || *ptr == '-') {
        int sign = (*ptr == '+') ? 1 : -1;
        const char *off = ptr + 1;
        if (std::isdigit(off[0]) && std::isdigit(off[1])) {
            int64_t hz = (off[0] - '0') * 10 + (off[1] - '0');
            off += 2;
            if (*off == ':') ++off;
            int64_t mz = 0;
            if (std::isdigit(off[0]) && std::isdigit(off[1])) {
                mz = (off[0] - '0') * 10 + (off[1] - '0');
                off += 2;
            }
            tz_offset = sign * (hz * 60 + mz) * 60;
        }
        if (*off != '\0') {
            throw std::runtime_error("parseIso8601ToMs: trailing characters after TZ offset: " + iso);
        }
    } else {
        throw std::runtime_error("parseIso8601ToMs: unexpected character after time: " + iso);
    }

    // timegm interprets tm as UTC.
    time_t epoch = timegm(&tm);
    if (epoch < 0) {
        throw std::runtime_error("parseIso8601ToMs: timegm failed for: " + iso);
    }
    return static_cast<int64_t>(epoch) * 1000 - tz_offset * 1000;
}

bool isEofMarker(const av::Packet& p)
{
    return p.pts().isNoPts() && (p.size() == 1);
}

bool isEofMarker(const av::VideoFrame& f)
{
    return f.pts().isNoPts();
}

bool isEofMarker(const av::AudioSamples& f)
{
    return f.pts().isNoPts();
}

bool isEofMarker(const EglImageFrame& f)
{
    return f.pts().isNoPts();
}

bool isEofMarker(const MetadataFrame& f)
{
    return f.pts().isNoPts();
}

av::Packet createEofPacket(int streamIndex)
{
    auto p = av::Packet({ 0xFF });
    p.setStreamIndex(streamIndex);
    return p;
}

static const char *const kStreamRestartKey = "avplumber_stream_restart";

// The packet's string metadata side data as a dictionary (empty without one); the caller frees it.
static AVDictionary *packetStrings(const av::Packet &p)
{
    AVDictionary *dict = nullptr;
    size_t size = 0;
    if (const uint8_t *data = av_packet_get_side_data(p.raw(), AV_PKT_DATA_STRINGS_METADATA, &size))
        av_packet_unpack_dictionary(data, size, &dict);
    return dict;
}

// Replaces the packet's string metadata with `dict`, or removes it when `dict` is empty.
static void setPacketStrings(av::Packet &p, const AVDictionary *dict)
{
    AVPacket *pkt = p.raw();
    if (!av_dict_count(dict)) {
        av_packet_side_data_remove(pkt->side_data, &pkt->side_data_elems, AV_PKT_DATA_STRINGS_METADATA);
        return;
    }
    size_t size = 0;
    uint8_t *data = av_packet_pack_dictionary(const_cast<AVDictionary *>(dict), &size);
    if (data && av_packet_add_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA, data, size) < 0)
        av_free(data);
}

void markStreamRestart(av::Packet &p)
{
    AVDictionary *dict = packetStrings(p);
    av_dict_set(&dict, kStreamRestartKey, "1", 0);
    setPacketStrings(p, dict);
    av_dict_free(&dict);
}

bool takeStreamRestart(av::Packet &p)
{
    AVDictionary *dict = packetStrings(p);
    const bool marked = av_dict_get(dict, kStreamRestartKey, nullptr, 0) != nullptr;
    if (marked) {
        av_dict_set(&dict, kStreamRestartKey, nullptr, 0);
        setPacketStrings(p, dict);
    }
    av_dict_free(&dict);
    return marked;
}
