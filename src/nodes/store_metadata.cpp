// store_metadata.cpp — StoreMetadata node
//
// Reads VideoFrame from a source edge, extracts selected metadata keys from the
// frame's AVFrame dictionary, builds a JSON record { "pts": <sec>, "pts_tb":
// "<num>/<den>", "<key>": <value>, ... }, and publishes it to a configured
// backend.  The frame is forwarded to the optional destination edge unchanged.
//
// Node params:
//   src          source edge name (VideoFrame)
//   dst          optional destination edge (pass-through)
//   backend      JSON object: { "type": "kafka", "bootstrap_servers": "...",
//                               "topic": "...", "key": "..." }
//   metadata     JSON array of key names, or the string "*" to forward all keys
//
// Only "kafka" is supported as a backend type.
//
// librdkafka is used for Kafka I/O (https://github.com/confluentinc/librdkafka).

#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/mathematics.h>
}

#include <rdkafka.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>

// ── helpers ──────────────────────────────────────────────────────────────────

static std::string ptsSeconds(const av::VideoFrame &frm) {
    av::Timestamp pts = frm.pts();
    if (!pts.isValid()) return "0.0";
    double sec = pts.seconds();
    std::ostringstream oss;
    oss << sec;
    return oss.str();
}

static std::string ptsTb(const av::VideoFrame &frm) {
    av::Rational tb = frm.timeBase();
    std::ostringstream oss;
    oss << tb.getNumerator() << "/" << tb.getDenominator();
    return oss.str();
}

// Minimal JSON string escaping — values from AVFrame metadata are typically
// short human-readable strings or compact JSON; full escaping suffices.
static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

// Detect whether a string looks like a JSON object/array and embed it raw,
// otherwise embed as a quoted string.
static bool looksLikeJson(const std::string &s) {
    if (s.empty()) return false;
    return (s.front() == '{' && s.back() == '}') ||
           (s.front() == '[' && s.back() == ']');
}

// ── Kafka backend ─────────────────────────────────────────────────────────────

class KafkaProducer {
public:
    KafkaProducer(const std::string &brokers, const std::string &topic,
                  const std::string &msg_key)
        : msg_key_(msg_key) {
        char errstr[512];

        rd_kafka_conf_t *conf = rd_kafka_conf_new();
        if (rd_kafka_conf_set(conf, "bootstrap.servers", brokers.c_str(),
                              errstr, sizeof(errstr)) != RD_KAFKA_CONF_OK) {
            rd_kafka_conf_destroy(conf);
            throw std::runtime_error(std::string("rdkafka conf: ") + errstr);
        }
        // Delivery report callback — we don't need per-message callbacks; errors
        // are surfaced via rd_kafka_poll.
        rd_kafka_conf_set(conf, "acks", "1", nullptr, 0);

        rk_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
        if (!rk_) {
            throw std::runtime_error(std::string("rdkafka new: ") + errstr);
        }

        rkt_ = rd_kafka_topic_new(rk_, topic.c_str(), nullptr);
        if (!rkt_) {
            rd_kafka_destroy(rk_);
            rk_ = nullptr;
            throw std::runtime_error("rdkafka topic_new failed for: " + topic);
        }
    }

    ~KafkaProducer() {
        flush();
        if (rkt_) rd_kafka_topic_destroy(rkt_);
        if (rk_) {
            rd_kafka_flush(rk_, 5000);
            rd_kafka_destroy(rk_);
        }
    }

    void produce(const std::string &payload) {
        const void *key = msg_key_.empty() ? nullptr : (const void *)msg_key_.c_str();
        size_t key_len  = msg_key_.empty() ? 0 : msg_key_.size();

        int rc = rd_kafka_produce(
            rkt_, RD_KAFKA_PARTITION_UA,
            RD_KAFKA_MSG_F_COPY,
            (void *)payload.data(), payload.size(),
            key, key_len,
            nullptr);

        if (rc == -1) {
            // Non-fatal — log but don't crash the pipeline
            logstream << "[StoreMetadata] kafka produce error: "
                      << rd_kafka_err2str(rd_kafka_last_error()) << std::endl;
        }
        // Poll to serve delivery callbacks and avoid queue buildup
        rd_kafka_poll(rk_, 0);
    }

    void flush() {
        if (rk_) rd_kafka_flush(rk_, 3000);
    }

private:
    rd_kafka_t       *rk_  = nullptr;
    rd_kafka_topic_t *rkt_ = nullptr;
    std::string       msg_key_;
};

// ── Node ──────────────────────────────────────────────────────────────────────

// Supports both SISO (pass-through, when "dst" param is present) and SI (sink,
// when "dst" is absent).  Using NodeSingleInput directly to handle the optional
// downstream edge without requiring createCommon which mandates a dst param.
class StoreMetadata : public NodeSingleInput<av::VideoFrame>,
                      public ReportsFinishByFlag {
    std::vector<std::string> metadata_keys_;
    bool all_keys_ = false;
    std::unique_ptr<KafkaProducer> producer_;
    std::unique_ptr<Sink<av::VideoFrame>> opt_sink_;  // null when used as pure sink

    std::string buildRecord(const av::VideoFrame &frm) const {
        std::string rec = "{\"pts\":";
        rec += ptsSeconds(frm);
        rec += ",\"pts_tb\":\"";
        rec += ptsTb(frm);
        rec += "\"";

        const AVFrame *raw = frm.raw();
        if (raw && raw->metadata) {
            if (all_keys_) {
                AVDictionaryEntry *entry = nullptr;
                while ((entry = av_dict_get(raw->metadata, "", entry, AV_DICT_IGNORE_SUFFIX))) {
                    rec += ",\"";
                    rec += jsonEscape(entry->key);
                    rec += "\":";
                    std::string val(entry->value);
                    if (looksLikeJson(val)) {
                        rec += val;
                    } else {
                        rec += "\"";
                        rec += jsonEscape(val);
                        rec += "\"";
                    }
                }
            } else {
                for (const auto &key : metadata_keys_) {
                    AVDictionaryEntry *entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
                    if (!entry || !entry->value) continue;
                    rec += ",\"";
                    rec += jsonEscape(key);
                    rec += "\":";
                    std::string val(entry->value);
                    if (looksLikeJson(val)) {
                        rec += val;
                    } else {
                        rec += "\"";
                        rec += jsonEscape(val);
                        rec += "\"";
                    }
                }
            }
        }

        rec += "}";
        return rec;
    }

public:
    StoreMetadata(std::unique_ptr<Source<av::VideoFrame>> &&source,
                  std::unique_ptr<Sink<av::VideoFrame>> &&sink,
                  std::vector<std::string> keys, bool all_keys,
                  std::unique_ptr<KafkaProducer> producer)
        : NodeSingleInput<av::VideoFrame>(std::move(source)),
          metadata_keys_(std::move(keys)),
          all_keys_(all_keys),
          producer_(std::move(producer)),
          opt_sink_(std::move(sink)) {}

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            if (opt_sink_) opt_sink_->put(av::VideoFrame{});
            this->finished_ = true;
            return;
        }

        std::string record = buildRecord(frm);
        producer_->produce(record);

        if (opt_sink_) opt_sink_->put(frm);
    }

    static std::shared_ptr<StoreMetadata> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        // ── backend ──────────────────────────────────────────────────────────
        if (!params.count("backend")) {
            throw std::runtime_error("StoreMetadata: 'backend' parameter is required");
        }
        Parameters backend = params["backend"];
        if (backend.is_string()) {
            backend = Parameters::parse(backend.get<std::string>());
        }
        std::string btype = backend.value("type", std::string("kafka"));
        if (btype != "kafka") {
            throw std::runtime_error("StoreMetadata: unsupported backend type: " + btype);
        }

        std::string brokers = backend.value("bootstrap_servers", std::string("localhost:9092"));
        std::string topic   = backend.value("topic", std::string(""));
        if (topic.empty()) {
            throw std::runtime_error("StoreMetadata: backend.topic is required");
        }
        std::string msg_key = backend.value("key", std::string(""));

        auto producer = std::make_unique<KafkaProducer>(brokers, topic, msg_key);

        // ── metadata keys ────────────────────────────────────────────────────
        bool all_keys = false;
        std::vector<std::string> keys;
        if (params.count("metadata")) {
            const Parameters &meta = params["metadata"];
            if (meta.is_string()) {
                std::string s = meta.get<std::string>();
                if (s == "*") {
                    all_keys = true;
                } else {
                    keys.push_back(s);
                }
            } else if (meta.is_array()) {
                for (const auto &item : meta) {
                    keys.push_back(item.get<std::string>());
                }
            }
        }

        // Wire source edge
        std::shared_ptr<Edge<av::VideoFrame>> src_edge = edges.find<av::VideoFrame>(params["src"]);

        // Wire optional sink edge
        std::unique_ptr<Sink<av::VideoFrame>> sink;
        if (params.count("dst") && params["dst"].is_string() && !params["dst"].get<std::string>().empty()) {
            std::shared_ptr<Edge<av::VideoFrame>> dst_edge = edges.find<av::VideoFrame>(params["dst"]);
            sink = dst_edge->makeSink();
        }

        return std::make_shared<StoreMetadata>(
            src_edge->makeSource(), std::move(sink),
            std::move(keys), all_keys, std::move(producer));
    }

    ~StoreMetadata() override {
        if (producer_) producer_->flush();
    }
};

DECLNODE(store_metadata, StoreMetadata)
