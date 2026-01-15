#pragma once
#include <cstdint>
#include <string>

class ControlImpl;

#ifdef EMBED_IN_OBS
struct obs_source;
typedef struct obs_source obs_source_t;
#endif


/*
 * Usage #1: (as in standalone application)
 *  mainLoop()
 * will return after any node with restart=panic finishes or you call
 *  stopMainLoop()
 * from a different thread or from interrupt
 * 
 * Usage #2: (as in obs-avplumber-source)
 *  setReady()
 * will make avplumber usable, working in background threads. When you want it to stop working, call:
 *  shutdown()
 * you may want to call
 *  heartbeat()
 * periodically to make avplumber print some status information to the log
 */

class AVPlumber {
private:
    ControlImpl* impl_;
    uint16_t control_port_;
#ifdef EMBED_IN_OBS
    std::string PAUSE_TEAM_;
    std::string REALTIME_TEAM_;

    void get_pause_team_name();
    void get_realtime_team_name();
#endif
public:
    AVPlumber();
    ~AVPlumber();
    void enableControlServer(const uint16_t tcp_port);
    void registerWithWebUI(const std::string& webui_api_url, const std::string& instance_name = "", const std::string& log_file = "");
#ifdef EMBED_IN_OBS
    void setObsSource(obs_source_t* obssrc);
    void unsetObsSourceAndWait();
    void obsTick();

    bool obs_is_paused();
    void obs_play();
    void obs_pause();
    void obs_stop();
    void obs_restart();
    int64_t obs_get_time();
    void obs_set_time(int64_t ms);
    int64_t obs_get_duration();
    double obs_get_speed();
    bool obs_is_eof();
#endif

    void executeCommandsFromFile(const std::string path);
    void executeCommandsFromString(const std::string script);
    void setLogFile(const std::string path);
    void setReady();
    void shutdown();
    void mainLoop();
    void stopMainLoop();
    void heartbeat();
};
