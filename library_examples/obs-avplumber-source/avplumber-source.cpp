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

    State current_state_ {State::Stopped};
    std::atomic<State> desired_state_ {State::Stopped};
    Event mgmt_wakeup_;
    std::mutex avplumber_create_destroy_;
    std::thread mgmt_thread_;

    void doStop() {
        if (avplumber_) {
            avplumber_->shutdown();
            {
                std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
                avplumber_ = nullptr;
            }
        }
    }
    void doStart() {
        {
            std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
            avplumber_ = std::unique_ptr<AVPlumber>(new AVPlumber());
            avplumber_->setLogFile(log_path_);
            avplumber_->setObsSource(source_);
        }
        avplumber_->executeCommandsFromString(script_);
        avplumber_->enableControlServer(control_port_);
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
                        doStart();
                    } else if (desired == State::Stopped) {
                        doStop();
                    }
                    current_state_ = (desired==State::Restart) ? State::Stopped : desired;
                } catch (std::exception &e) {
                    std::cerr << "State transition failed: " << e.what() << std::endl;
                }
            }
        }
    }
public:
    void goToState(State desired) {
        desired_state_ = desired;
        mgmt_wakeup_.signal();
    }
    void reloadSettings(obs_data_t *settings) {
        script_ = obs_data_get_string(settings, "script");
        control_port_ = obs_data_get_int(settings, "control_port");
        log_path_ = obs_data_get_string(settings, "log_path");
        if (current_state_==State::Started || current_state_==State::Restart) {
            goToState(State::Restart);
        }
    }
    /*void shutdown() {
        shutdownButReturnASAP();
        join();
    }*/
    void shutdownButReturnASAP() {
        {
            std::lock_guard<decltype(avplumber_create_destroy_)> lock(avplumber_create_destroy_);
            avplumber_->unsetObsSourceAndWait();
        }
        goToState(State::Shutdown);
    }
    void join() {
        mgmt_thread_.join();
    }
    void tick() {
        avplumber_->obsTick();
    }
    AVPlumberSource(obs_data_t *settings, obs_source_t *source):
        source_(source),
        mgmt_thread_([this]() { mgmtThreadFunction(); }) {
        reloadSettings(settings);
    }
};

static void* avplumber_source_create (obs_data_t *settings, obs_source_t *source) {
    AVPlumberSource *avpsrc = new AVPlumberSource(settings, source);
    avpsrc->goToState(State::Started);
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

    UNUSED_PARAMETER(data);
    return props;
}

void avplumber_source_get_defaults(obs_data_t *settings) {
    obs_data_set_default_string(settings, "script", "");
    obs_data_set_default_int(settings, "control_port", 0);
    obs_data_set_default_string(settings, "log_path", "");
}

static const char *avplumber_source_getname (void *unused) {
    UNUSED_PARAMETER (unused);
    return "avplumber source";
}

struct obs_source_info avplumber_source = {
    .id = "avplumber_source",
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_ASYNC_VIDEO | OBS_SOURCE_AUDIO,
    .get_name = avplumber_source_getname,
    .create = avplumber_source_create,
    .destroy = avplumber_source_destroy,
    .get_defaults = avplumber_source_get_defaults,
    .get_properties = avplumber_source_get_properties,
    .update = avplumber_source_update,
    .video_tick = avplumber_source_tick
};


bool obs_module_load ( void )
{
    obs_register_source ( &avplumber_source );
    return true;
}
