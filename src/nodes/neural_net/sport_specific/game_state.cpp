#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <cctype>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class GameState : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string scoreboard_metadata_key_ = "scoreboard";
    std::string output_metadata_key_ = "game_state";
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    static std::string upperAscii(std::string s) {
        for (char& c : s) c = (char)std::toupper((unsigned char)c);
        return s;
    }

    static std::string normalizeTeamToken(const std::string& token) {
        static const std::unordered_set<std::string> nba = {
            "ATL", "BOS", "BKN", "BRK", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
            "GS", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
            "NOP", "NO", "NY", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC",
            "SA", "SAS", "TOR", "UTA", "WAS"
        };
        static const std::unordered_map<std::string, std::string> aliases = {
            {"HAWKS", "ATL"}, {"ATLANTA", "ATL"}, {"CELTICS", "BOS"}, {"BOSTON", "BOS"},
            {"NETS", "BKN"}, {"BROOKLYN", "BKN"}, {"HORNETS", "CHA"}, {"CHARLOTTE", "CHA"},
            {"BULLS", "CHI"}, {"CHICAGO", "CHI"}, {"CAVALIERS", "CLE"}, {"CAVS", "CLE"}, {"CLEVELAND", "CLE"},
            {"MAVERICKS", "DAL"}, {"MAVS", "DAL"}, {"DALLAS", "DAL"}, {"NUGGETS", "DEN"}, {"DENVER", "DEN"},
            {"PISTONS", "DET"}, {"DETROIT", "DET"}, {"WARRIORS", "GSW"}, {"GOLDENSTATE", "GSW"},
            {"ROCKETS", "HOU"}, {"HOUSTON", "HOU"}, {"PACERS", "IND"}, {"INDIANA", "IND"},
            {"CLIPPERS", "LAC"}, {"LAKERS", "LAL"}, {"GRIZZLIES", "MEM"}, {"MEMPHIS", "MEM"},
            {"HEAT", "MIA"}, {"MIAMI", "MIA"}, {"BUCKS", "MIL"}, {"MILWAUKEE", "MIL"},
            {"TIMBERWOLVES", "MIN"}, {"WOLVES", "MIN"}, {"MINNESOTA", "MIN"}, {"PELICANS", "NOP"}, {"NEWORLEANS", "NOP"},
            {"KNICKS", "NYK"}, {"NEWYORK", "NYK"}, {"THUNDER", "OKC"}, {"OKLAHOMACITY", "OKC"},
            {"MAGIC", "ORL"}, {"ORLANDO", "ORL"}, {"76ERS", "PHI"}, {"SIXERS", "PHI"}, {"PHILADELPHIA", "PHI"},
            {"SUNS", "PHX"}, {"PHOENIX", "PHX"}, {"TRAILBLAZERS", "POR"}, {"BLAZERS", "POR"}, {"PORTLAND", "POR"},
            {"KINGS", "SAC"}, {"SACRAMENTO", "SAC"}, {"SPURS", "SAS"}, {"SANANTONIO", "SAS"},
            {"RAPTORS", "TOR"}, {"TORONTO", "TOR"}, {"JAZZ", "UTA"}, {"UTAH", "UTA"},
            {"WIZARDS", "WAS"}, {"WASHINGTON", "WAS"}
        };
        if (nba.count(token)) {
            if (token == "BRK") return "BKN";
            if (token == "GS") return "GSW";
            if (token == "NO") return "NOP";
            if (token == "NY") return "NYK";
            if (token == "SA") return "SAS";
            return token;
        }
        auto it = aliases.find(token);
        return it == aliases.end() ? std::string() : it->second;
    }

    static std::string nbaFullName(const std::string& abbrev) {
        static const std::unordered_map<std::string, std::string> names = {
            {"ATL", "Atlanta Hawks"}, {"BOS", "Boston Celtics"}, {"BKN", "Brooklyn Nets"},
            {"CHA", "Charlotte Hornets"}, {"CHI", "Chicago Bulls"}, {"CLE", "Cleveland Cavaliers"},
            {"DAL", "Dallas Mavericks"}, {"DEN", "Denver Nuggets"}, {"DET", "Detroit Pistons"},
            {"GSW", "Golden State Warriors"}, {"HOU", "Houston Rockets"}, {"IND", "Indiana Pacers"},
            {"LAC", "LA Clippers"}, {"LAL", "Los Angeles Lakers"}, {"MEM", "Memphis Grizzlies"},
            {"MIA", "Miami Heat"}, {"MIL", "Milwaukee Bucks"}, {"MIN", "Minnesota Timberwolves"},
            {"NOP", "New Orleans Pelicans"}, {"NYK", "New York Knicks"}, {"OKC", "Oklahoma City Thunder"},
            {"ORL", "Orlando Magic"}, {"PHI", "Philadelphia 76ers"}, {"PHX", "Phoenix Suns"},
            {"POR", "Portland Trail Blazers"}, {"SAC", "Sacramento Kings"}, {"SAS", "San Antonio Spurs"},
            {"TOR", "Toronto Raptors"}, {"UTA", "Utah Jazz"}, {"WAS", "Washington Wizards"}
        };
        auto it = names.find(abbrev);
        return it == names.end() ? std::string() : it->second;
    }

    static std::string extractAbbrev(const std::string& text) {
        const std::string up = upperAscii(text);
        std::vector<std::string> runs;
        for (size_t i = 0; i < up.size(); ++i) {
            if (!std::isupper((unsigned char)up[i])) continue;
            std::string run;
            size_t j = i;
            while (j < up.size() && std::isupper((unsigned char)up[j]) && run.size() < 18) {
                run.push_back(up[j]);
                ++j;
            }
            if (!run.empty()) runs.push_back(run);
            i = j;
        }
        std::string squashed;
        for (char c : up) {
            if (std::isalnum((unsigned char)c)) squashed.push_back(c);
        }
        if (!squashed.empty()) runs.push_back(squashed);
        for (const auto& run : runs) {
            std::string normalized = normalizeTeamToken(run);
            if (!normalized.empty()) return normalized;
        }
        return "";
    }

    static int extractInt(const std::string& text) {
        int best = -1;
        int cur = -1;
        for (char c : text) {
            if (std::isdigit((unsigned char)c)) {
                if (cur < 0) cur = 0;
                cur = cur * 10 + (c - '0');
            } else if (cur >= 0) {
                best = cur;
                cur = -1;
            }
        }
        if (cur >= 0) best = cur;
        return best;
    }

    static bool parseStrictNumberText(const std::string& text, int min_val, int max_val, int& out) {
        std::string digits;
        for (char c : text) {
            if (std::isdigit((unsigned char)c)) digits.push_back(c);
            else if (!std::isspace((unsigned char)c) && c != ':') return false;
        }
        if (digits.empty() || digits.size() > 3) return false;
        int value = 0;
        for (char c : digits) value = value * 10 + (c - '0');
        if (value < min_val || value > max_val) return false;
        out = value;
        return true;
    }

    static int parseClockSec(const std::string& text) {
        for (size_t i = 0; i + 3 < text.size(); ++i) {
            if (!std::isdigit((unsigned char)text[i])) continue;
            size_t colon = text.find(':', i);
            if (colon == std::string::npos || colon + 2 >= text.size()) break;
            bool mm_ok = colon > i;
            bool ss_ok = std::isdigit((unsigned char)text[colon + 1]) && std::isdigit((unsigned char)text[colon + 2]);
            if (!mm_ok || !ss_ok) continue;

            int mm = 0;
            for (size_t j = i; j < colon; ++j) {
                if (!std::isdigit((unsigned char)text[j])) { mm_ok = false; break; }
                mm = mm * 10 + (text[j] - '0');
            }
            if (!mm_ok) continue;

            int ss = (text[colon + 1] - '0') * 10 + (text[colon + 2] - '0');
            if (ss >= 60) continue;
            if (mm > 12) continue;
            return mm * 60 + ss;
        }
        return -1;
    }

    static int parsePeriodNum(const std::string& text) {
        const std::string up = upperAscii(text);
        if (up.find("1ST") != std::string::npos || up.find("IST") != std::string::npos) return 1;
        if (up.find("2ND") != std::string::npos) return 2;
        if (up.find("3RD") != std::string::npos) return 3;
        if (up.find("4TH") != std::string::npos) return 4;
        if (up.find("OT") != std::string::npos) return 5;
        int n = extractInt(up);
        if (n >= 1 && n <= 9) return n;
        return -1;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
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
        Parameters scoreboard_md = tryParse(raw, scoreboard_metadata_key_);
        if (!scoreboard_md.is_object() || scoreboard_md.is_null()) {
            this->sink_->put(frm);
            return;
        }

        Parameters out_md;

        if (scoreboard_md.contains("team_a") && scoreboard_md["team_a"].contains("name")) {
            const std::string text = scoreboard_md["team_a"]["name"].value("text", std::string());
            std::string abbrev = extractAbbrev(text);
            if (!abbrev.empty()) {
                out_md["team_a_abbrev"] = abbrev;
                const std::string full = nbaFullName(abbrev);
                if (!full.empty()) out_md["team_a_name"] = full;
            }
        }
        if (scoreboard_md.contains("team_b") && scoreboard_md["team_b"].contains("name")) {
            const std::string text = scoreboard_md["team_b"]["name"].value("text", std::string());
            std::string abbrev = extractAbbrev(text);
            if (!abbrev.empty()) {
                out_md["team_b_abbrev"] = abbrev;
                const std::string full = nbaFullName(abbrev);
                if (!full.empty()) out_md["team_b_name"] = full;
            }
        }
        if (scoreboard_md.contains("team_a") && scoreboard_md["team_a"].contains("points")) {
            int score = -1;
            if (parseStrictNumberText(scoreboard_md["team_a"]["points"].value("text", std::string()), 0, 199, score)) {
                out_md["score_a"] = score;
            }
        }
        if (scoreboard_md.contains("team_b") && scoreboard_md["team_b"].contains("points")) {
            int score = -1;
            if (parseStrictNumberText(scoreboard_md["team_b"]["points"].value("text", std::string()), 0, 199, score)) {
                out_md["score_b"] = score;
            }
        }
        if (scoreboard_md.contains("period")) {
            int period_num = parsePeriodNum(scoreboard_md["period"].value("text", std::string()));
            if (period_num > 0) out_md["period_num"] = period_num;
        }
        if (scoreboard_md.contains("time_remaining")) {
            int sec = parseClockSec(scoreboard_md["time_remaining"].value("text", std::string()));
            if (sec >= 0) {
                out_md["game_clock_sec"] = sec;
                out_md["period_clock_remaining_sec"] = sec;
            }
        }
        if (scoreboard_md.contains("shot_clock")) {
            int sec = -1;
            if (parseStrictNumberText(scoreboard_md["shot_clock"].value("text", std::string()), 0, 24, sec) && sec > 0) {
                out_md["shot_clock_sec"] = sec;
            }
        }

        if (out_md.contains("score_a") && out_md.contains("score_b")) {
            int a = out_md["score_a"].get<int>();
            int b = out_md["score_b"].get<int>();
            out_md["score_margin"] = std::abs(a - b);
            if (a > b && out_md.contains("team_a_abbrev")) {
                out_md["leading_team"] = out_md["team_a_abbrev"];
                if (out_md.contains("team_a_name")) out_md["leading_team_name"] = out_md["team_a_name"];
            } else if (b > a && out_md.contains("team_b_abbrev")) {
                out_md["leading_team"] = out_md["team_b_abbrev"];
                if (out_md.contains("team_b_name")) out_md["leading_team_name"] = out_md["team_b_name"];
            } else if (a == b) {
                out_md["leading_team"] = "TIE";
            }
        }

        if (!out_md.empty()) {
            av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);
            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "game_state: frame=" << frame_counter_ << " json=" << out_md.dump();
            }
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<GameState> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<GameState>(edges, params);
        if (params.count("scoreboard_metadata_key")) r->scoreboard_metadata_key_ = params["scoreboard_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        return r;
    }
};

DECLNODE(game_state, GameState)
