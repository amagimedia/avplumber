#include <memory>
#include <signal.h> // for trap handling
#include <iostream>

#include <flags.hh>

#include "util.hpp"
#include "avplumber.hpp"
#include "app_version.hpp"

std::shared_ptr<AVPlumber> avp_ptr = nullptr;

void abort_handler(int) {
    logstream << "SIGABRT received";
}

void stop_handler(int) {
    logstream << "SIGTERM or SIGINT received";
    auto avp = avp_ptr;
    if (avp) {
        avp->stopMainLoop();
    }
}


int main(int argc, char **argv) {
    set_thread_name("avplumber main");
    logstream << APP_VERSION << " starting" << std::endl;

    Flags args;

    std::string script_path;
    uint16_t tcp_port;
    std::string log_path;
    bool show_version;

    args.Var(script_path, 's', "script", std::string(""), "Execute commands from this file");
    args.Var(tcp_port, 'p', "port", uint16_t(0), "Port to listen on, for commands (0 to disable)");
    args.Var(log_path, 'l', "logfile", std::string(""), "Write messages to this file (does not affect libav messages)");
    args.Bool(show_version, 'V', "version", std::string(""), "Show version and exit");
    args.Parse(argc, argv);
    
    if (show_version) {
        std::cout << APP_VERSION << std::endl;
        return 0;
    }

    signal(SIGABRT, &abort_handler); // when compiled with -ftrapv, warn instead of killing the process
    signal(SIGINT, &stop_handler);
    signal(SIGTERM, &stop_handler);
    
    
    avp_ptr = std::make_shared<AVPlumber>();
    AVPlumber &avp = *avp_ptr;
    avp.setLogFile(log_path);
    avp.enableControlServer(tcp_port);
    if (!script_path.empty()) {
        logstream << "Starting parsing file " << script_path;
        avp.executeCommandsFromFile(script_path);
        logstream << "Finished parsing file " << script_path;
    }
    avp.mainLoop();

    // need this because otherwise race condition happens between
    // InstanceSharedObjectsDestructors::callDestructors called in ~InstanceData
    // and implicitly called (on exit) destructor of static field InstanceSharedObjectsDestructors::destructors_
    avp_ptr = nullptr;

    return 0;
}
