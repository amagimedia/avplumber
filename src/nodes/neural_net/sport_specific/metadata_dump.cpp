#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

class MetadataDump : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string player_metadata_key_ = "yolo_players";
    std::string ball_metadata_key_ = "yolo_ball";
    std::string ball_handler_metadata_key_ = "ball_handler";
    std::string shot_metadata_key_ = "shot_info";
    std::string scoreboard_metadata_key_ = "scoreboard";
    std::string viewport_metadata_key_ = "smoothed_crop_viewport_v1";
    std::string output_metadata_key_ = "frame_dump";
    std::string output_file_;
    int dump_every_n_ = 1;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    std::string cached_json_;
    std::ofstream file_out_;
    bool file_opened_ = false;

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    static int ri(float v) { return (int)std::round(v); }

    static Parameters compactBox(const Parameters& det) {
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return nullptr;
        return Parameters::array({
            ri(det["xyxy"][0].get<float>()),
            ri(det["xyxy"][1].get<float>()),
            ri(det["xyxy"][2].get<float>()),
            ri(det["xyxy"][3].get<float>())
        });
    }

    Parameters buildDump(const av::VideoFrame& frm) {
        const AVFrame* raw = frm.raw();
        Parameters out;
        out["pts"] = frm.pts().timestamp();
        out["frame"] = frame_counter_;
        out["w"] = frm.width();
        out["h"] = frm.height();

        auto players_md = tryParse(raw, player_metadata_key_);
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto handler_md = tryParse(raw, ball_handler_metadata_key_);
        auto shot_md = tryParse(raw, shot_metadata_key_);
        auto scoreboard_md = tryParse(raw, scoreboard_metadata_key_);
        auto viewport_md = tryParse(raw, viewport_metadata_key_);

        if (players_md.contains("model_width")) out["model_w"] = players_md["model_width"];
        if (players_md.contains("model_height")) out["model_h"] = players_md["model_height"];

        // Shot type
        if (shot_md.contains("shot_type")) out["shot"] = shot_md["shot_type"];

        // Scoreboard
        if (scoreboard_md.is_object() && !scoreboard_md.is_null()) {
            if (scoreboard_md.contains("time_remaining"))
                out["clock"] = scoreboard_md["time_remaining"].value("text", std::string());
            if (scoreboard_md.contains("period"))
                out["period"] = scoreboard_md["period"].value("text", std::string());
            if (scoreboard_md.contains("shot_clock"))
                out["shot_clock"] = scoreboard_md["shot_clock"].value("text", std::string());

            Parameters teams;
            if (scoreboard_md.contains("team_a")) {
                Parameters ta;
                if (scoreboard_md["team_a"].contains("name"))
                    ta["name"] = scoreboard_md["team_a"]["name"].value("text", std::string());
                if (scoreboard_md["team_a"].contains("points"))
                    ta["score"] = scoreboard_md["team_a"]["points"].value("text", std::string());
                if (!ta.empty()) teams["a"] = ta;
            }
            if (scoreboard_md.contains("team_b")) {
                Parameters tb;
                if (scoreboard_md["team_b"].contains("name"))
                    tb["name"] = scoreboard_md["team_b"]["name"].value("text", std::string());
                if (scoreboard_md["team_b"].contains("points"))
                    tb["score"] = scoreboard_md["team_b"]["points"].value("text", std::string());
                if (!tb.empty()) teams["b"] = tb;
            }
            if (!teams.empty()) out["teams"] = teams;
        }

        // Ball handler track_id (resolve before building players)
        int handler_track_id = -1;
        if (handler_md.contains("detections") && handler_md["detections"].is_array()) {
            for (const auto& det : handler_md["detections"]) {
                if (det.value("label", std::string()) == "BallHandler") {
                    handler_track_id = det.value("track_id", -1);
                    break;
                }
            }
        }

        // Players, refs, hoop from yolo_players
        if (players_md.contains("detections") && players_md["detections"].is_array()) {
            Parameters players_arr = Parameters::array();
            Parameters refs_arr = Parameters::array();
            Parameters hoop;

            for (const auto& det : players_md["detections"]) {
                if (!det.is_object()) continue;
                std::string label = det.value("label", std::string());
                auto box = compactBox(det);
                if (box.is_null()) continue;
                float conf = det.value("conf", 0.0f);

                if (label == "Player") {
                    Parameters p;
                    int tid = det.value("track_id", -1);
                    if (tid >= 0) p["id"] = tid;
                    std::string tab = det.value("team_ab", std::string("?"));
                    if (tab != "?") p["team"] = tab;
                    p["box"] = box;
                    p["conf"] = (int)(conf * 100.0f + 0.5f);
                    if (tid >= 0 && tid == handler_track_id) p["has_ball"] = true;
                    players_arr.push_back(std::move(p));
                } else if (label == "Ref") {
                    Parameters r;
                    r["box"] = box;
                    r["conf"] = (int)(conf * 100.0f + 0.5f);
                    refs_arr.push_back(std::move(r));
                } else if (label == "Hoop" && hoop.is_null()) {
                    hoop = {{"box", box}, {"conf", (int)(conf * 100.0f + 0.5f)}};
                }
            }

            if (!players_arr.empty()) out["players"] = players_arr;
            if (!refs_arr.empty()) out["refs"] = refs_arr;
            if (!hoop.is_null()) out["hoop"] = hoop;
        }

        // Ball
        if (ball_md.contains("detections") && ball_md["detections"].is_array()) {
            for (const auto& det : ball_md["detections"]) {
                if (!det.is_object()) continue;
                auto box = compactBox(det);
                if (box.is_null()) continue;
                Parameters b;
                b["box"] = box;
                b["conf"] = (int)(det.value("conf", 0.0f) * 100.0f + 0.5f);
                if (det.contains("source")) b["state"] = det["source"];
                if (handler_track_id >= 0) b["handler_id"] = handler_track_id;
                out["ball"] = b;
                break;
            }
        }

        // Viewport
        if (viewport_md.contains("detections") && viewport_md["detections"].is_array()) {
            for (const auto& det : viewport_md["detections"]) {
                auto box = compactBox(det);
                if (!box.is_null()) { out["viewport"] = box; break; }
            }
        }

        return out;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;
        if (isEofMarker(frm)) {
            if (file_opened_) {
                file_out_ << "\n]\n";
                file_out_.close();
                file_opened_ = false;
            }
            frame_counter_ = 0;
            cached_json_.clear();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        if (dump_every_n_ > 1 && (frame_counter_ % (uint64_t)dump_every_n_) != 1) {
            if (!cached_json_.empty()) {
                av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);
            }
            this->sink_->put(frm);
            return;
        }

        Parameters dump = buildDump(frm);
        cached_json_ = dump.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);

        if (!output_file_.empty()) {
            if (!file_opened_) {
                file_out_.open(output_file_, std::ios::out | std::ios::trunc);
                if (file_out_) {
                    file_out_ << "[\n";
                    file_opened_ = true;
                } else {
                    logstream << "metadata_dump: cannot open output file: " << output_file_;
                }
            }
            if (file_opened_) {
                if (frame_counter_ > 1) file_out_ << ",\n";
                file_out_ << cached_json_;
                file_out_.flush();
            }
        }

        if (debug_log_every_n_ > 0) {
            logstream << "metadata_dump: frame=" << frame_counter_ << " " << cached_json_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<MetadataDump> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDump>(edges, params);

        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("ball_handler_metadata_key")) r->ball_handler_metadata_key_ = params["ball_handler_metadata_key"].get<std::string>();
        if (params.count("shot_metadata_key")) r->shot_metadata_key_ = params["shot_metadata_key"].get<std::string>();
        if (params.count("scoreboard_metadata_key")) r->scoreboard_metadata_key_ = params["scoreboard_metadata_key"].get<std::string>();
        if (params.count("viewport_metadata_key")) r->viewport_metadata_key_ = params["viewport_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("output_file")) r->output_file_ = params["output_file"].get<std::string>();
        if (params.count("dump_every_n")) r->dump_every_n_ = std::max(1, params["dump_every_n"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        return r;
    }
};

DECLNODE(metadata_dump, MetadataDump)
