#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "reframe_metadata_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

class BallPlayerAssociation : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    using DetectionBox = avp_sport_reframe::DetectionBox;

    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "ball_player_association";
    std::string ball_label_ = "basketball";
    std::vector<std::string> player_labels_ = {"Player"};
    double min_ball_conf_ = 0.04;
    double min_player_conf_ = 0.25;
    double ball_box_half_px_ = 18.0;
    double inv_d2_epsilon_px_ = 12.0;
    int max_candidates_ = 8;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    uint64_t stat_frames_with_ball_ = 0;
    uint64_t stat_frames_with_candidates_ = 0;

    bool matchesPlayerLabel(const DetectionBox& d) const {
        if (!d.has_label) return false;
        for (const auto& label : player_labels_) {
            if (d.label == label) return true;
        }
        return false;
    }

    DetectionBox bestBall(const std::vector<DetectionBox>& balls) const {
        DetectionBox best;
        best.conf = -1.0;
        for (const auto& b : balls) {
            if (!avp_sport_reframe::labelMatches(b, ball_label_)) continue;
            if (b.conf < min_ball_conf_) continue;
            if (b.conf > best.conf) best = b;
        }
        return best;
    }

    static double iou(const DetectionBox& a, const DetectionBox& b) {
        const double ix1 = std::max(a.x1, b.x1);
        const double iy1 = std::max(a.y1, b.y1);
        const double ix2 = std::min(a.x2, b.x2);
        const double iy2 = std::min(a.y2, b.y2);
        const double iw = std::max(0.0, ix2 - ix1);
        const double ih = std::max(0.0, iy2 - iy1);
        const double inter = iw * ih;
        if (!(inter > 0.0)) return 0.0;
        const double aa = std::max(0.0, a.x2 - a.x1) * std::max(0.0, a.y2 - a.y1);
        const double ba = std::max(0.0, b.x2 - b.x1) * std::max(0.0, b.y2 - b.y1);
        const double uni = aa + ba - inter;
        return uni > 0.0 ? inter / uni : 0.0;
    }

    DetectionBox ballSquare(double cx, double cy) const {
        const double h = std::max(1.0, ball_box_half_px_);
        DetectionBox b;
        b.label = ball_label_;
        b.has_label = true;
        b.conf = 1.0;
        b.x1 = cx - h;
        b.y1 = cy - h;
        b.x2 = cx + h;
        b.y2 = cy + h;
        return b;
    }

    static DetectionBox heightRoi(double cx, double cy, double model_w, double model_h) {
        const double half = std::max(1.0, model_h * 0.5);
        DetectionBox r;
        r.label = "BallROI";
        r.has_label = true;
        r.conf = 1.0;
        r.x1 = std::max(0.0, std::min(model_w, cx - half));
        r.y1 = std::max(0.0, std::min(model_h, cy - half));
        r.x2 = std::max(0.0, std::min(model_w, cx + half));
        r.y2 = std::max(0.0, std::min(model_h, cy + half));
        return r;
    }

    struct Association {
        DetectionBox player;
        double roi_iou = 0.0;
        double ball_iou = 0.0;
        double distance_px = 0.0;
        double raw_score = 0.0;
        double score = 0.0;
    };

    std::vector<Association> scorePlayers(const std::vector<DetectionBox>& players,
                                          const DetectionBox& ball_box,
                                          const DetectionBox& roi,
                                          double bx,
                                          double by) const {
        std::vector<Association> out;
        double sum = 0.0;
        for (const auto& p : players) {
            if (p.conf < min_player_conf_) continue;
            if (!matchesPlayerLabel(p)) continue;
            const double roi_iou = iou(roi, p);
            if (!(roi_iou > 0.0)) continue;
            const double ball_iou = iou(ball_box, p);
            const double dist = avp_sport_reframe::pointToBoxDistance(bx, by, p);
            const double d_eff = std::max(dist, std::max(1.0, inv_d2_epsilon_px_));
            double raw = (1.0 / (d_eff * d_eff)) * std::sqrt(std::max(0.0, p.conf));
            if (ball_iou > 0.0) raw *= (1.0 + 8.0 * ball_iou);
            if (!(raw > 0.0) || !std::isfinite(raw)) continue;

            Association a;
            a.player = p;
            a.roi_iou = roi_iou;
            a.ball_iou = ball_iou;
            a.distance_px = dist;
            a.raw_score = raw;
            out.push_back(a);
            sum += raw;
        }
        if (sum > 0.0) {
            for (auto& a : out) a.score = a.raw_score / sum;
        }
        std::sort(out.begin(), out.end(), [](const Association& a, const Association& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.distance_px != b.distance_px) return a.distance_px < b.distance_px;
            return a.player.track_id < b.player.track_id;
        });
        if ((int)out.size() > max_candidates_) out.resize((size_t)max_candidates_);
        return out;
    }

    void writeOutput(av::VideoFrame& frm,
                     double model_w,
                     double model_h,
                     const DetectionBox* ball,
                     const DetectionBox* ball_box,
                     const DetectionBox* roi,
                     const std::vector<Association>& assocs) {
        Parameters out;
        out["coord_space"] = "model";
        out["model_width"] = model_w;
        out["model_height"] = model_h;
        out["ball_present"] = ball != nullptr;
        out["strategy"] = "height_roi_inv_d2_ball_iou";
        out["associations"] = Parameters::array();
        if (ball) {
            out["ball_xyxy"] = {ball->x1, ball->y1, ball->x2, ball->y2};
            out["ball_center"] = {avp_sport_reframe::centerX(*ball), avp_sport_reframe::centerY(*ball)};
            out["ball_conf"] = ball->conf;
        }
        if (ball_box) out["ball_box_xyxy"] = {ball_box->x1, ball_box->y1, ball_box->x2, ball_box->y2};
        if (roi) out["roi_xyxy"] = {roi->x1, roi->y1, roi->x2, roi->y2};

        int rank = 0;
        for (const auto& a : assocs) {
            Parameters item;
            item["rank"] = rank++;
            item["player_id"] = a.player.has_track_id ? a.player.track_id : -1;
            item["player_label"] = a.player.label;
            item["player_conf"] = a.player.conf;
            item["xyxy"] = {a.player.x1, a.player.y1, a.player.x2, a.player.y2};
            item["score"] = avp_sport_reframe::clampDouble(a.score, 0.0, 1.0);
            item["raw_score"] = a.raw_score;
            item["roi_iou"] = a.roi_iou;
            item["ball_iou"] = a.ball_iou;
            item["distance_px"] = a.distance_px;
            out["associations"].push_back(item);
        }
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out.dump().c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    bool consumeEofIfPresent() override {
        return false;
    }

    ~BallPlayerAssociation() {
        if (frame_counter_ == 0) return;
        logstream << "ball_player_association: === summary ===";
        logstream << "  total frames:             " << frame_counter_;
        logstream << "  frames with ball:         " << stat_frames_with_ball_;
        logstream << "  frames with candidates:   " << stat_frames_with_candidates_;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            frame_counter_ = 0;
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const AVFrame* raw = frm.raw();
        double model_w = 960.0;
        double model_h = 544.0;

        std::vector<DetectionBox> balls;
        std::vector<DetectionBox> players;
        avp_sport_reframe::parseDetections(raw, ball_metadata_key_, balls, model_w, model_h);
        avp_sport_reframe::parseDetections(raw, player_metadata_key_, players, model_w, model_h);
        const DetectionBox ball = bestBall(balls);
        const bool have_ball = ball.conf >= min_ball_conf_;

        std::vector<Association> assocs;
        DetectionBox b_square;
        DetectionBox roi;
        if (have_ball) {
            ++stat_frames_with_ball_;
            const double bx = avp_sport_reframe::centerX(ball);
            const double by = avp_sport_reframe::centerY(ball);
            b_square = ballSquare(bx, by);
            roi = heightRoi(bx, by, model_w, model_h);
            assocs = scorePlayers(players, b_square, roi, bx, by);
            if (!assocs.empty()) ++stat_frames_with_candidates_;
        }

        writeOutput(frm,
                    model_w,
                    model_h,
                    have_ball ? &ball : nullptr,
                    have_ball ? &b_square : nullptr,
                    have_ball ? &roi : nullptr,
                    assocs);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "ball_player_association: frame=" << frame_counter_
                      << " ball=" << (have_ball ? 1 : 0)
                      << " candidates=" << assocs.size()
                      << " top_id=" << (!assocs.empty() && assocs[0].player.has_track_id ? assocs[0].player.track_id : -1)
                      << " top_score=" << (!assocs.empty() ? assocs[0].score : 0.0)
                      << " top_dist=" << (!assocs.empty() ? assocs[0].distance_px : -1.0);
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<BallPlayerAssociation> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallPlayerAssociation>(edges, params);
        r->auto_eof_ = false;

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& l : params["player_labels"]) r->player_labels_.push_back(l.get<std::string>());
        }
        if (params.count("min_ball_conf")) r->min_ball_conf_ = params["min_ball_conf"].get<double>();
        if (params.count("min_player_conf")) r->min_player_conf_ = params["min_player_conf"].get<double>();
        if (params.count("ball_box_half_px")) r->ball_box_half_px_ = params["ball_box_half_px"].get<double>();
        if (params.count("inv_d2_epsilon_px")) r->inv_d2_epsilon_px_ = params["inv_d2_epsilon_px"].get<double>();
        if (params.count("max_candidates")) r->max_candidates_ = params["max_candidates"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        r->ball_box_half_px_ = std::max(1.0, r->ball_box_half_px_);
        r->inv_d2_epsilon_px_ = std::max(1.0, r->inv_d2_epsilon_px_);
        r->max_candidates_ = std::max(1, r->max_candidates_);
        return r;
    }
};

DECLNODE(ball_player_association, BallPlayerAssociation)
