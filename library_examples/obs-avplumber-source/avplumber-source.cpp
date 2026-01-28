#include <obs-module.h>
#include <memory>
#include <thread>
#include <atomic>
#include <iostream>
#include <mutex>

#define EMBED_IN_OBS 1

#include "avplumber/src/avplumber.hpp"
#include "avplumber/src/Event.hpp"

OBS_DECLARE_MODULE()

enum class State {
    Stopped,
    Started,
    Restart,
    Shutdown
};

class AVPlumberSource {
    obs_source_t *source_;
    std::unique_ptr<AVPlumber> avplumber_;

    std::string script_;
    uint16_t control_port_;
    std::string log_path_;
    std::string webui_api_url_;
    std::string instance_name_;

    std::string start_ts_;
    std::string stop_ts_;
    bool loop_ = false;

    State current_state_ {State::Stopped};
    obs_media_state obs_state_ { OBS_MEDIA_STATE_NONE };
    bool is_playing_ = false;
    std::atomic<State> desired_state_ {State::Stopped};
    Event mgmt_wakeup_;
    std::mutex avplumber_create_destroy_;
    std::thread mgmt_thread_;

    void doStop() {
        if (avplumber_) {
            // no mutex needed - avplumber_ pointer changes only in the same thread
            avplumber_->shutdown();
            {
                std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
                avplumber_ = nullptr;
            }
        }
    }
    void doStart() {
        if (avplumber_) {
            // no mutex needed - avplumber_ pointer changes only in the same thread
            avplumber_->shutdown();
        }
        {
            std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
            avplumber_ = std::unique_ptr<AVPlumber>(new AVPlumber());
            avplumber_->setLogFile(log_path_);
            avplumber_->setObsSource(source_);
        }
        avplumber_->executeCommandsFromString(script_);
        avplumber_->enableControlServer(control_port_);
        if (!webui_api_url_.empty()) {
            avplumber_->registerWithWebUI(webui_api_url_, instance_name_, log_path_);
        }
        avplumber_->setReady();
    }
    void mgmtThreadFunction() {
        while (true) {
            State desired = desired_state_.load();
            if (current_state_ == desired) {
                if (avplumber_) {
                    mgmt_wakeup_.wait(3000);
                    avplumber_->heartbeat();
                } else {
                    mgmt_wakeup_.wait();
                }
            } else {
                try {
                    if (desired == State::Shutdown) {
                        doStop();
                        break;
                    } else if (desired == State::Restart) {
                        desired_state_ = State::Started;
                        doStop();
                    } else if (desired == State::Started) {
                        obs_state_ = OBS_MEDIA_STATE_OPENING;
                        doStart();
                        obs_state_ = OBS_MEDIA_STATE_PLAYING;
                        is_playing_ = true;
                    } else if (desired == State::Stopped) {
                        obs_state_ = OBS_MEDIA_STATE_STOPPED;
                        is_playing_ = false;
                        doStop();
                    }
                    current_state_ = (desired==State::Restart) ? State::Stopped : desired;
                } catch (std::exception &e) {
                    std::cerr << "State transition failed: " << e.what() << std::endl;
                    mgmt_wakeup_.wait(500);
                }
            }
        }
    }
    template<typename F> void with_avplumber(const char* op_name, F&& func) {
        if (!avplumber_) {
            return;
        }
        // this is called from different thread than management thread and must be mutexed approprietly
        if (avplumber_create_destroy_.try_lock()) {
            try {
                if (avplumber_ != nullptr) {
                    func(avplumber_.get());
                }
            } catch (std::exception &e) {
                std::cerr << op_name << " failed: " << e.what() << std::endl;
            }
            avplumber_create_destroy_.unlock();
        }
    }
public:
    void goToState(State desired) {
        desired_state_ = desired;
        mgmt_wakeup_.signal();
    }
    void reloadSettings(obs_data_t *settings) {
        bool need_restart = obs_data_get_bool(settings, "force_restart");
        bool need_update_ts = false;
        std::string s;
        bool b;
        char buf[128];

        s = obs_data_get_string(settings, "script");
        if (script_ != s) {
            need_restart = true;
            script_ = s;
        }
        s = obs_data_get_string(settings, "log_path");
        if (log_path_ != s) {
            need_restart = true;
            log_path_ = s;
        }
        s = obs_data_get_string(settings, "webui_api_url");
        if (webui_api_url_ != s) {
            need_restart = true;
            webui_api_url_ = s;
        }
        s = obs_data_get_string(settings, "instance_name");
        if (instance_name_ != s) {
            need_restart = true;
            instance_name_ = s;
        }
        uint16_t p = obs_data_get_int(settings, "control_port");
        if (control_port_ != p) {
            need_restart = true;
            control_port_ = p;
        }
        s = obs_data_get_string(settings, "start_ts");
        if (start_ts_ != s) {
            start_ts_ = s;
            sprintf(buf, "node.param.set input start_ts \"%s\"", start_ts_.c_str());
            send_command(buf);
            need_update_ts = true;
        }
        s = obs_data_get_string(settings, "stop_ts");
        if (stop_ts_ != s) {
            stop_ts_ = s;
            sprintf(buf, "node.param.set input stop_ts \"%s\"", stop_ts_.c_str());
            send_command(buf);
            need_update_ts = true;
        }
        b = obs_data_get_bool(settings, "loop");
        if (loop_ != b) {
            loop_ = b;
            sprintf(buf, "node.param.set input loop %s", loop_ ? "true" : "false");
            send_command(buf);
            need_update_ts = true;
        }
        if (need_restart) {
            if (current_state_==State::Started || current_state_==State::Restart) {
                goToState(State::Restart);
            }
        } else {
            if (need_update_ts) {
                sprintf(buf, "node.object.set input stream-limits {\"start\": \"%s\", \"stop\": \"%s\", \"loop\": %s }",
                    start_ts_.c_str(), stop_ts_.c_str(), loop_ ? "true" : "false");
                send_command(buf);
            }
        }
    }
    /*void shutdown() {
        shutdownButReturnASAP();
        join();
    }*/
    void shutdownButReturnASAP() {
        // this is called from different thread than management thread and must be mutexed approprietly
        {
            std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
            if (avplumber_) {
                avplumber_->unsetObsSourceAndWait();
            }
        }
        goToState(State::Shutdown);
    }
    void join() {
        mgmt_thread_.join();
    }
    void tick() {
        with_avplumber("tick", [](AVPlumber* avp) {
            avp->obsTick();
        });
    }
    void obs_play() {
        with_avplumber("play", [](AVPlumber* avp) {
            avp->obs_play();
        });
        if (!is_playing_) {
            obs_source_media_started(source_);
        }
        obs_state_ = OBS_MEDIA_STATE_PLAYING;
        is_playing_ = true;
    }
    void obs_pause() {
        with_avplumber("pause", [](AVPlumber* avp) {
            avp->obs_pause();
        });
        obs_state_ = OBS_MEDIA_STATE_PAUSED;
        is_playing_ = false;
    }
    int64_t obs_get_time() {
        int64_t time = -1;
        with_avplumber("get time", [&time](AVPlumber* avp) {
            time = avp->obs_get_time();
        });
        return time;
    }
    void obs_set_time(int64_t ms) {
        with_avplumber("set time", [ms](AVPlumber* avp) {
            avp->obs_set_time(ms);
        });
    }
    int64_t obs_get_duration() {
        int64_t duration = -1;
        with_avplumber("get duration", [&duration](AVPlumber* avp) {
            duration = avp->obs_get_duration();
        });
        return duration;
    }
    double obs_get_speed() {
        double speed = 1.00;
        with_avplumber("get speed", [&speed](AVPlumber* avp) {
            speed = avp->obs_get_speed();
        });
        return speed;
    }
    obs_media_state obs_get_state() {
        if (is_playing_) {
            with_avplumber("state check", [this](AVPlumber* avp) {
                if (avp->obs_is_eof()) {
                    obs_state_ = OBS_MEDIA_STATE_ENDED;
                } else {
                    obs_state_ = OBS_MEDIA_STATE_PLAYING;
                }
            });
        }
        return obs_state_;
    }
    void obs_stop() {
        with_avplumber("stop", [](AVPlumber* avp) {
            avp->obs_stop();
        });
        obs_source_media_ended(source_);
        obs_state_ = OBS_MEDIA_STATE_STOPPED;
        is_playing_ = false;
    }
    void obs_restart() {
        with_avplumber("restart", [this](AVPlumber* avp) {
            avp->obs_restart();
            if (avp->obs_is_paused()) {
                obs_state_ = OBS_MEDIA_STATE_PAUSED;
                is_playing_ = false;
            } else {
                obs_state_ = OBS_MEDIA_STATE_PLAYING;
                is_playing_ = true;
                obs_source_media_started(source_);
            }
        });
    }
    bool send_command(const std::string& cmd) {
        bool result = false;
        with_avplumber("command", [&cmd, &result](AVPlumber* avp) {
            avp->executeCommandsFromString(cmd);
            result = true;
        });
        return result;
    }
    AVPlumberSource(obs_data_t *settings, obs_source_t *source):
        source_(source),
        mgmt_thread_([this]() { mgmtThreadFunction(); }) {
        reloadSettings(settings);
    }
};

static void avplumber_proc_command(void *data, calldata_t *cd)
{
    AVPlumberSource* avpsrc = (AVPlumberSource*)data;
    std::string cmd = calldata_string(cd, "cmd");
    avpsrc->send_command(cmd);
}

static void avplumber_proc_get_speed(void *data, calldata_t *cd)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        calldata_set_float(cd, "speed", avpsrc->obs_get_speed());
    }
    catch (...) {

    }
}

static void* avplumber_source_create (obs_data_t *settings, obs_source_t *source) {
    AVPlumberSource *avpsrc = new AVPlumberSource(settings, source);
    avpsrc->goToState(State::Started);

	proc_handler_t *ph = obs_source_get_proc_handler(source);
	proc_handler_add(ph, "void command(in string cmd out string result)", avplumber_proc_command, avpsrc);
	proc_handler_add(ph, "void get_speed(out float speed)", avplumber_proc_get_speed, avpsrc);

    return avpsrc;
}

static void avplumber_source_destroy (void *data) {
    AVPlumberSource* avpsrc = (AVPlumberSource*) data;
    if (!avpsrc) {
        return;
    }
    avpsrc->shutdownButReturnASAP();

    std::thread([avpsrc]() {
        avpsrc->join();
        delete avpsrc;
    }).detach();
}

void avplumber_source_update(void *data, obs_data_t *settings) {
    AVPlumberSource* avpsrc = (AVPlumberSource*) data;
    avpsrc->reloadSettings(settings);
}

void avplumber_source_tick(void *data, float) {
    AVPlumberSource* avpsrc = (AVPlumberSource*) data;
    avpsrc->tick();
}

static obs_properties_t *avplumber_source_get_properties(void *data) {
    obs_properties_t *props = obs_properties_create();
    obs_properties_set_flags(props, OBS_PROPERTIES_DEFER_UPDATE);

    obs_properties_add_text(props, "script", "Script", OBS_TEXT_MULTILINE);
    obs_properties_add_int(props, "control_port", "Control interface TCP port (0 to disable)", 0, 65535, 1);
    obs_properties_add_text(props, "log_path", "Write log to file (empty = use stderr)", OBS_TEXT_DEFAULT);
    obs_properties_add_text(props, "webui_api_url", "Web UI server API endpoint URL for auto-registration (e.g., http://localhost:22222)", OBS_TEXT_DEFAULT);
    obs_properties_add_text(props, "instance_name", "Instance name for web UI registration", OBS_TEXT_DEFAULT);

    obs_properties_add_text(props, "start_ts", "Start timestamp", OBS_TEXT_DEFAULT);
    obs_properties_add_text(props, "stop_ts", "Stop timestamp", OBS_TEXT_DEFAULT);
    obs_properties_add_bool(props, "loop", "Loop stream");

    UNUSED_PARAMETER(data);
    return props;
}

static void avplumber_source_play_pause(void *data, bool pause)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;

        if (pause) {
            avpsrc->obs_pause();
        } else {
            avpsrc->obs_play();
        }
    }
    catch (...) {

    }
}

static void avplumber_source_stop(void *data)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        avpsrc->obs_stop();
    }
    catch (...) {

    }
}

static void avplumber_source_restart(void *data)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        avpsrc->obs_restart();
    }
    catch (...) {

    }
}

static int64_t avplumber_source_get_duration(void *data)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        return avpsrc->obs_get_duration();
    }
    catch (...) {

    }
    return 0;
}

static int64_t avplumber_source_get_time(void *data)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        return avpsrc->obs_get_time();
    }
    catch (...) {

    }
    return 0;
}

static void avplumber_source_set_time(void *data, int64_t ms)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        avpsrc->obs_set_time(ms);
    }
    catch (...) {

    }
}

static enum obs_media_state avplumber_source_get_state(void *data)
{
    try {
        AVPlumberSource* avpsrc = (AVPlumberSource*) data;
        return avpsrc->obs_get_state();
    }
    catch (...) {

    }
    return OBS_MEDIA_STATE_ERROR;
}

void avplumber_source_get_defaults(obs_data_t *settings) {
    obs_data_set_default_string(settings, "script", "");
    obs_data_set_default_int(settings, "control_port", 0);
    obs_data_set_default_string(settings, "log_path", "");
    obs_data_set_default_string(settings, "webui_api_url", "");
    obs_data_set_default_string(settings, "instance_name", "");
    obs_data_set_default_string(settings, "start_ts", "");
    obs_data_set_default_string(settings, "stop_ts", "");
    obs_data_set_default_bool(settings, "loop", false);
}

static const char *avplumber_source_getname (void *unused) {
    UNUSED_PARAMETER (unused);
    return "avplumber source";
}

struct obs_source_info avplumber_source = {
    .id = "avplumber_source",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_ASYNC_VIDEO | OBS_SOURCE_AUDIO | OBS_SOURCE_DO_NOT_DUPLICATE | OBS_SOURCE_CONTROLLABLE_MEDIA,
    .get_name = avplumber_source_getname,
    .create = avplumber_source_create,
    .destroy = avplumber_source_destroy,
    .get_defaults = avplumber_source_get_defaults,
    .get_properties = avplumber_source_get_properties,
    .update = avplumber_source_update,
    .video_tick = avplumber_source_tick,
	.icon_type = OBS_ICON_TYPE_MEDIA,
    .media_play_pause = avplumber_source_play_pause,
	.media_restart = avplumber_source_restart,
	.media_stop = avplumber_source_stop,
	.media_get_duration = avplumber_source_get_duration,
	.media_get_time = avplumber_source_get_time,
	.media_set_time = avplumber_source_set_time,
	.media_get_state = avplumber_source_get_state
};


bool obs_module_load ( void )
{
    obs_register_source ( &avplumber_source );
    return true;
}
