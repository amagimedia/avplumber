#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>

class ShotAttemptDetector : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string handler_metadata_key_ = "ball_handler";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    std::string seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "shot_events";
    double min_flight_speed_ = 6.0;
    double max_vy_for_release_ = -2.0;
    double hoop_arrival_dist_ = 40.0;
    int seg_side_data_slot_ = 0;
    double mask_threshold_ = 0.5;
    double line_inside_margin_px_ = 12.0;
    uint64_t shooter_hold_frames_ = 20;
    int outcome_vector_window_frames_ = 6;
    double outcome_min_speed_ = 2.0;
    double outcome_min_downward_vy_ = 1.5;
    double outcome_min_downward_ratio_ = 0.45;
    int release_confirm_frames_ = 3;
    int cooldown_frames_ = 30;
    double hoop_min_conf_ = 0.3;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    bool in_flight_ = false;
    uint64_t flight_start_frame_ = 0;
    int flight_frames_ = 0;
    bool release_emitted_ = false;
    bool hoop_emitted_ = false;
    uint64_t cooldown_until_ = 0;
    int total_releases_ = 0;
    int total_hoop_arrivals_ = 0;
    uint64_t last_handler_frame_ = 0;
    double prev_hoop_dist_ = 1e9;
    int pending_attempt_points_ = 0;
    std::string pending_attempt_type_;

    struct FlightSample {
        uint64_t frame = 0;
        double cx = 0.0;
        double cy = 0.0;
        double vx = 0.0;
        double vy = 0.0;
        bool has_velocity = false;
    };

    struct HandlerState {
        bool found = false;
        double foot_x = 0.0;
        double foot_y = 0.0;
    };

    struct MaskInfo {
        int num_masks = 0;
        int w = 0;
        int h = 0;
        const float* data = nullptr;
    };

    struct CourtContext {
        bool usable = false;
        bool has_line = false;
        bool hoop_right = true;
        double model_w = 960.0;
        double model_h = 544.0;
        MaskInfo masks;
        int line_mask_idx = -1;
    };

    struct AttemptValue {
        bool known = false;
        std::string type;
        int points = 0;
    };

    HandlerState last_handler_;
    std::deque<FlightSample> flight_samples_;

    struct ShotOutcome {
        bool classified = false;
        std::string result;
        double confidence = 0.0;
        double vx = 0.0;
        double vy = 0.0;
    };

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    struct BallState {
        bool detected = false;
        double cx = 0, cy = 0;
        double vx = 0, vy = 0;
        double speed = 0;
        std::string source;
    };

    struct HoopState {
        bool found = false;
        double cx = 0, cy = 0;
        double w = 0, h = 0;
    };

    BallState parseBall(const Parameters& md) const {
        BallState b;
        if (!md.contains("detections") || !md["detections"].is_array()) return b;
        for (const auto& det : md["detections"]) {
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            double x1 = det["xyxy"][0].get<double>(), y1 = det["xyxy"][1].get<double>();
            double x2 = det["xyxy"][2].get<double>(), y2 = det["xyxy"][3].get<double>();
            b.detected = true;
            b.cx = (x1 + x2) * 0.5;
            b.cy = (y1 + y2) * 0.5;
            b.source = det.value("source", std::string("detected"));
            if (det.contains("velocity_x") && det.contains("velocity_y")) {
                b.vx = det["velocity_x"].get<double>();
                b.vy = det["velocity_y"].get<double>();
                b.speed = std::sqrt(b.vx * b.vx + b.vy * b.vy);
            }
            break;
        }
        return b;
    }

    HoopState parseHoop(const Parameters& md) const {
        HoopState h;
        if (!md.contains("detections") || !md["detections"].is_array()) return h;
        for (const auto& det : md["detections"]) {
            if (det.value("label", std::string()) != "Hoop") continue;
            if (det.value("conf", 0.0) < hoop_min_conf_) continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            double x1 = det["xyxy"][0].get<double>(), y1 = det["xyxy"][1].get<double>();
            double x2 = det["xyxy"][2].get<double>(), y2 = det["xyxy"][3].get<double>();
            h.found = true;
            h.cx = (x1 + x2) * 0.5;
            h.cy = (y1 + y2) * 0.5;
            h.w = x2 - x1;
            h.h = y2 - y1;
            break;
        }
        return h;
    }

    HandlerState parseHandler(const Parameters& handler_md) const {
        HandlerState h;
        if (!handler_md.contains("detections") || !handler_md["detections"].is_array()) return h;
        for (const auto& det : handler_md["detections"]) {
            if (det.value("label", std::string()) != "BallHandler") continue;
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            h.found = true;
            h.foot_x = (det["xyxy"][0].get<double>() + det["xyxy"][2].get<double>()) * 0.5;
            h.foot_y = det["xyxy"][3].get<double>();
            return h;
        }
        return h;
    }

    static bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
        if (!raw) return false;
        AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(slot));
        if (!sd || !sd->buf || sd->buf->size < 16) return false;
        const uint32_t* header = (const uint32_t*)sd->buf->data;
        out.num_masks = (int)header[0];
        out.w = (int)header[1];
        out.h = (int)header[2];
        size_t expected = 16 + (size_t)out.num_masks * (size_t)out.w * (size_t)out.h * sizeof(float);
        if ((size_t)sd->buf->size < expected) return false;
        out.data = (const float*)(sd->buf->data + 16);
        return out.num_masks > 0 && out.w > 0 && out.h > 0;
    }

    static const float* maskPtr(const MaskInfo& masks, int idx) {
        if (!masks.data || idx < 0 || idx >= masks.num_masks) return nullptr;
        return masks.data + (size_t)idx * (size_t)masks.w * (size_t)masks.h;
    }

    static int roundToMask(double v, double model_dim, int mask_dim) {
        if (mask_dim <= 0 || model_dim <= 0.0) return 0;
        double scaled = v * (double)mask_dim / model_dim;
        return std::max(0, std::min((int)std::lround(scaled), mask_dim - 1));
    }

    CourtContext parseCourtContext(const AVFrame* raw, const Parameters& seg_md, const HoopState& hoop) const {
        CourtContext ctx;
        ctx.model_w = seg_md.value("model_width", 960.0);
        ctx.model_h = seg_md.value("model_height", 544.0);
        ctx.hoop_right = hoop.found ? (hoop.cx >= ctx.model_w * 0.5) : true;
        if (!hoop.found || !readCpuMasks(raw, seg_side_data_slot_, ctx.masks)) return ctx;
        if (seg_md.contains("detections") && seg_md["detections"].is_array()) {
            for (size_t i = 0; i < seg_md["detections"].size(); ++i) {
                const auto& det = seg_md["detections"][i];
                if (det.value("label", std::string()) == "three point line") {
                    ctx.line_mask_idx = (int)i;
                    ctx.has_line = ctx.line_mask_idx < ctx.masks.num_masks;
                    break;
                }
            }
        }
        ctx.usable = ctx.has_line;
        return ctx;
    }

    bool lineBoundaryAtY(const CourtContext& ctx, double model_y, int& line_x_mask) const {
        if (!ctx.usable) return false;
        const float* line_mask = maskPtr(ctx.masks, ctx.line_mask_idx);
        if (!line_mask) return false;
        int mask_y = roundToMask(model_y, ctx.model_h, ctx.masks.h);
        int found_x = ctx.hoop_right ? -1 : ctx.masks.w;
        bool found = false;
        for (int y = std::max(0, mask_y - 3); y <= std::min(ctx.masks.h - 1, mask_y + 3); ++y) {
            for (int x = 0; x < ctx.masks.w; ++x) {
                if (line_mask[y * ctx.masks.w + x] < (float)mask_threshold_) continue;
                found = true;
                if (ctx.hoop_right) found_x = std::max(found_x, x);
                else found_x = std::min(found_x, x);
            }
            if (found) break;
        }
        if (!found) return false;
        line_x_mask = found_x;
        return true;
    }

    AttemptValue classifyAttemptValue(const CourtContext& ctx, const HandlerState& shooter) const {
        AttemptValue out;
        if (!ctx.usable || !shooter.found) return out;
        int line_x_mask = 0;
        if (!lineBoundaryAtY(ctx, shooter.foot_y, line_x_mask)) return out;
        int point_x_mask = roundToMask(shooter.foot_x, ctx.model_w, ctx.masks.w);
        double margin_mask = line_inside_margin_px_ * (double)ctx.masks.w / std::max(1.0, ctx.model_w);
        bool inside_three = ctx.hoop_right
            ? (double)point_x_mask >= (double)line_x_mask - margin_mask
            : (double)point_x_mask <= (double)line_x_mask + margin_mask;
        out.known = true;
        out.type = inside_three ? "2pt" : "3pt";
        out.points = inside_three ? 2 : 3;
        return out;
    }

    void resetFlight() {
        in_flight_ = false;
        flight_frames_ = 0;
        release_emitted_ = false;
        hoop_emitted_ = false;
        prev_hoop_dist_ = 1e9;
        pending_attempt_points_ = 0;
        pending_attempt_type_.clear();
        flight_samples_.clear();
    }

    void recordFlightSample(const BallState& ball) {
        if (!ball.detected) return;
        FlightSample s;
        s.frame = frame_counter_;
        s.cx = ball.cx;
        s.cy = ball.cy;
        s.vx = ball.vx;
        s.vy = ball.vy;
        s.has_velocity = ball.speed > 0.0;
        flight_samples_.push_back(s);

        const uint64_t keep_frames = (uint64_t)std::max(1, outcome_vector_window_frames_ * 3);
        while (!flight_samples_.empty() && frame_counter_ - flight_samples_.front().frame > keep_frames) {
            flight_samples_.pop_front();
        }
    }

    ShotOutcome classifyNearHoopOutcome(const BallState& ball) const {
        ShotOutcome out;
        if (!ball.detected) return out;

        double vx = ball.vx;
        double vy = ball.vy;
        double speed = ball.speed;

        if (speed < outcome_min_speed_) {
            const FlightSample* first = nullptr;
            for (const auto& s : flight_samples_) {
                if (frame_counter_ - s.frame <= (uint64_t)std::max(1, outcome_vector_window_frames_)) {
                    first = &s;
                    break;
                }
            }
            if (first && first->frame != frame_counter_) {
                vx = ball.cx - first->cx;
                vy = ball.cy - first->cy;
                speed = std::sqrt(vx * vx + vy * vy);
            }
        }

        if (speed < outcome_min_speed_) return out;

        const double downward_ratio = vy / std::max(speed, 1e-6);
        const bool downward = vy >= outcome_min_downward_vy_ &&
                              downward_ratio >= outcome_min_downward_ratio_;
        out.classified = true;
        out.result = downward ? "scored" : "missed";
        out.vx = vx;
        out.vy = vy;
        out.confidence = std::min(1.0, std::max(0.0, std::abs(downward_ratio)));
        return out;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            logstream << "shot_attempt_detector: total releases=" << total_releases_
                      << " hoop_arrivals=" << total_hoop_arrivals_
                      << " in " << frame_counter_ << " frames";
            frame_counter_ = 0;
            resetFlight();
            cooldown_until_ = 0;
            total_releases_ = 0;
            total_hoop_arrivals_ = 0;
            last_handler_frame_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto player_md = tryParse(raw, player_metadata_key_);
        auto handler_md = tryParse(raw, handler_metadata_key_);
        auto shot_md = tryParse(raw, camera_shot_metadata_key_);

        bool wide_shot = shot_md.value("camera_shot_type", std::string()) == "wide";

        BallState ball = parseBall(ball_md);
        HoopState hoop = parseHoop(player_md);
        HandlerState handler_state = parseHandler(handler_md);
        bool handler = handler_state.found;
        if (handler) {
            last_handler_ = handler_state;
            last_handler_frame_ = frame_counter_;
        }

        auto seg_md = tryParse(raw, seg_metadata_key_);

        Parameters events;
        events["release"] = false;
        events["hoop_arrival"] = false;
        events["in_flight"] = false;
        events["ball_detected"] = ball.detected;
        events["hoop_detected"] = hoop.found;
        events["total_releases"] = total_releases_;
        events["total_hoop_arrivals"] = total_hoop_arrivals_;

        bool in_cooldown = frame_counter_ < cooldown_until_;

        if (handler || !ball.detected) {
            if (in_flight_ && !hoop_emitted_) {
                resetFlight();
            }
        }

        bool ball_in_air = ball.detected && !handler;
        events["ball_in_air"] = ball_in_air;
        if (ball_in_air || in_flight_) recordFlightSample(ball);

        double hoop_dist = 1e9;
        if (ball.detected && hoop.found) {
            double dx = ball.cx - hoop.cx, dy = ball.cy - hoop.cy;
            hoop_dist = std::sqrt(dx * dx + dy * dy);
            events["ball_hoop_dist"] = (int)std::round(hoop_dist);
        }

        if (!in_cooldown && wide_shot && ball_in_air) {
            bool has_arc = ball.vy <= max_vy_for_release_;
            bool approaching_hoop = hoop.found && hoop_dist < prev_hoop_dist_ && hoop_dist < 120.0;
            if (!in_flight_ && ball.speed >= min_flight_speed_ && (has_arc || approaching_hoop)) {
                in_flight_ = true;
                flight_start_frame_ = frame_counter_;
                flight_frames_ = 0;
            }
            if (in_flight_) flight_frames_++;

            if (flight_frames_ >= release_confirm_frames_ && !release_emitted_) {
                // Disable point-value classification until shooter/line geometry is reliable.
                // The score outcome can still be synthesized from the near-hoop ball vector.
                pending_attempt_type_ = "unknown";
                pending_attempt_points_ = 0;
                release_emitted_ = true;
                total_releases_++;
                events["release"] = true;
                events["release_frame"] = flight_start_frame_;
                events["total_releases"] = total_releases_;
                events["attempt_type"] = pending_attempt_type_;
                if (pending_attempt_points_ > 0) events["attempt_points"] = pending_attempt_points_;
                logstream << "shot_attempt_detector: RELEASE frame=" << flight_start_frame_
                          << " ball=[" << (int)ball.cx << "," << (int)ball.cy << "]"
                          << " speed=" << (int)ball.speed
                          << " vel=[" << (int)ball.vx << "," << (int)ball.vy << "]"
                          << " attempt=" << pending_attempt_type_;
            }
        }

        if (in_flight_ && release_emitted_ && !hoop_emitted_ && ball.detected && hoop.found) {
            double dx = ball.cx - hoop.cx;
            double dy = ball.cy - hoop.cy;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= hoop_arrival_dist_) {
                ShotOutcome outcome = classifyNearHoopOutcome(ball);
                hoop_emitted_ = true;
                total_hoop_arrivals_++;
                events["hoop_arrival"] = true;
                events["ball_hoop_dist"] = (int)std::round(dist);
                events["total_hoop_arrivals"] = total_hoop_arrivals_;
                if (outcome.classified) {
                    events["shot_result"] = true;
                    events["result"] = outcome.result;
                    events["result_source"] = "ball_vector";
                    events["result_conf"] = outcome.confidence;
                    events["result_vx"] = outcome.vx;
                    events["result_vy"] = outcome.vy;
                    events["attempt_type"] = pending_attempt_type_.empty() ? std::string("unknown") : pending_attempt_type_;
                    if (pending_attempt_points_ > 0) {
                        events["attempt_points"] = pending_attempt_points_;
                        if (outcome.result == "scored") events["points"] = pending_attempt_points_;
                    }
                }
                cooldown_until_ = frame_counter_ + (uint64_t)cooldown_frames_;
                logstream << "shot_attempt_detector: HOOP_ARRIVAL frame=" << frame_counter_
                          << " ball=[" << (int)ball.cx << "," << (int)ball.cy << "]"
                          << " hoop=[" << (int)hoop.cx << "," << (int)hoop.cy << "]"
                          << " dist=" << (int)dist
                          << " flight_frames=" << flight_frames_
                          << (outcome.classified ? " result=" + outcome.result : " result=unknown");
                resetFlight();
            }
        }

        events["in_flight"] = in_flight_ && release_emitted_;
        prev_hoop_dist_ = hoop_dist;

        std::string serialized = events.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "shot_attempt_detector: frame=" << frame_counter_
                      << " flight=" << in_flight_
                      << " releases=" << total_releases_
                      << " hoop_arrivals=" << total_hoop_arrivals_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ShotAttemptDetector> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ShotAttemptDetector>(edges, params);

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("seg_metadata_key")) r->seg_metadata_key_ = params["seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("min_flight_speed")) r->min_flight_speed_ = params["min_flight_speed"].get<double>();
        if (params.count("max_vy_for_release")) r->max_vy_for_release_ = params["max_vy_for_release"].get<double>();
        if (params.count("hoop_arrival_dist")) r->hoop_arrival_dist_ = params["hoop_arrival_dist"].get<double>();
        if (params.count("seg_side_data_slot")) r->seg_side_data_slot_ = params["seg_side_data_slot"].get<int>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<double>();
        if (params.count("line_inside_margin_px")) r->line_inside_margin_px_ = params["line_inside_margin_px"].get<double>();
        if (params.count("shooter_hold_frames")) r->shooter_hold_frames_ = params["shooter_hold_frames"].get<uint64_t>();
        if (params.count("outcome_vector_window_frames")) r->outcome_vector_window_frames_ = std::max(1, params["outcome_vector_window_frames"].get<int>());
        if (params.count("outcome_min_speed")) r->outcome_min_speed_ = params["outcome_min_speed"].get<double>();
        if (params.count("outcome_min_downward_vy")) r->outcome_min_downward_vy_ = params["outcome_min_downward_vy"].get<double>();
        if (params.count("outcome_min_downward_ratio")) r->outcome_min_downward_ratio_ = params["outcome_min_downward_ratio"].get<double>();
        if (params.count("release_confirm_frames")) r->release_confirm_frames_ = params["release_confirm_frames"].get<int>();
        if (params.count("cooldown_frames")) r->cooldown_frames_ = params["cooldown_frames"].get<int>();
        if (params.count("hoop_min_conf")) r->hoop_min_conf_ = params["hoop_min_conf"].get<double>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        return r;
    }
};

DECLNODE(shot_attempt_detector, ShotAttemptDetector)
