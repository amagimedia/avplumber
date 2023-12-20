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
public:
    AVPlumber();
    ~AVPlumber();
    void enableControlServer(const uint16_t tcp_port);
#ifdef EMBED_IN_OBS
    void setObsSource(obs_source_t* obssrc);
    void unsetObsSourceAndWait();
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
