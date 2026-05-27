#include "avplumber.hpp"

#include <list>
#include <fstream>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/write.hpp>
#include <json.hpp>
#include <memory>

#include "avutils.hpp"
#include "graph_mgmt.hpp"
#include "util.hpp"
#include "stats.hpp"
#include "logger_impls.hpp"
#include "output_control.hpp"
#include "hwaccel_mgmt.hpp"
#include "named_event.hpp"
#include "RealTimeTeam.hpp"
#include "SpeedControlTeam.hpp"
#include "PauseControlTeam.hpp"
#include "InputSeekTeam.hpp"
#include "PTSCorrectorCommon.hpp"
#include "rest_client.hpp"
#include "SharedTimeline.hpp"
#include "MixerState.hpp"
#include "mixer_orchestrator.hpp"
#include <libavformat/avformat.h>
#ifdef EMBED_IN_OBS
    #include "instance_shared.hpp"
    #include "TickSource.hpp"
    #include "EventLoop.hpp"

    #define INPUT_NODE "input"
    #define PAUSE_NODE "pause"
    #define SPEED_NODE "speed"
    #define REALTIME_NODE "rtsync"
    #define SINK_NODE "sink"
#endif

#include <avcpp/av.h>
#include <avcpp/avutils.h>

using boost::asio::ip::tcp;
using nlohmann::json;

static double probeMediaDurationSec(const std::string& url) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, url.c_str(), nullptr, nullptr) < 0) {
        throw Error("mixer.wipe: failed to open wipe file: " + url);
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        throw Error("mixer.wipe: failed to probe wipe file duration: " + url);
    }
    if (fmt->duration == AV_NOPTS_VALUE || fmt->duration <= 0) {
        avformat_close_input(&fmt);
        throw Error("mixer.wipe: wipe file has no usable duration, pass duration_sec explicitly");
    }
    double duration_sec = double(fmt->duration) / double(AV_TIME_BASE);
    avformat_close_input(&fmt);
    return duration_sec;
}

namespace strutils {
    bool isWhiteSpace(const char c) {
        return (c==' ') || (c=='\t') || (c=='\r') || (c=='\n');
    }
    std::string trim(const std::string &s) {
        if (s.empty()) {
            return "";
        }
        size_t startfrom = s.length();
        for (size_t i=0; i<s.length(); i++) {
            if (!isWhiteSpace(s[i])) {
                startfrom = i;
                break;
            }
        }
        size_t endat = 0;
        size_t i = s.length();
        do {
            i--;
            if (!isWhiteSpace(s[i])) {
                endat = i;
                break;
            }
        } while (i>0);
        if (startfrom <= endat) {
            return s.substr(startfrom, endat-startfrom+1);
        } else {
            return "";
        }
    }
    void toLowerInPlace(std::string &s) {
        for (char &c: s) {
            if (c >= 'A' && c <= 'Z') {
                c += 'a'-'A';
            }
        }
    }
};

class ControlServerBase {
public:
    virtual ~ControlServerBase() {
    }
};

struct ControlPacket {
    enum Type {
        None,
        Data,
        Start,
        End
    };
    Type type;
    std::string data;
    ControlPacket(Type _type = None, std::string _data = ""): type(_type), data(_data) {
    }
};

struct ClientPipe {
    moodycamel::ReaderWriterQueue<ControlPacket> to_client;
    moodycamel::BlockingReaderWriterQueue<ControlPacket> from_client;
    std::function<void()> send_to_client;
    ClientPipe(decltype(send_to_client) _send_to_client): send_to_client(_send_to_client) {
    }
};

class ControlImpl {
private:
    std::shared_ptr<NodeManager> manager_;
    std::list<std::unique_ptr<ControlServerBase>> servers_;
    using ClientStream = std::ostringstream;
    using CommandHandler = std::function<void(ClientStream&, std::string&)>;
    std::unordered_map<std::string, CommandHandler> commands_;
    std::unordered_set<std::string> no_lock_commands_;
    std::mutex cmd_run_lock_;
    std::mutex server_ready_;
    std::list<std::thread> detached_threads_;
    #ifdef EMBED_IN_OBS
        std::shared_ptr<TickSource> tick_source_;
    #endif

public:
    std::shared_ptr<NodeManager> manager() { return manager_; }
    void registerCommand(
            const std::string& command,
            std::function<std::string(const std::string&)> handler,
            bool no_lock = false) {
        std::string normalized = command;
        strutils::toLowerInPlace(normalized);
        if (commands_.count(normalized))
            throw Error("command already registered: " + normalized);
        commands_[normalized] = [handler = std::move(handler)](ClientStream &cs, std::string &arg) {
            cs << handler(arg);
        };
        if (no_lock)
            no_lock_commands_.insert(normalized);
        else
            no_lock_commands_.erase(normalized);
    }
    void lockOrNot(bool do_lock, std::function<void()> whattodo) {
        if (do_lock) {
            // locks should be no longer necessary
            // TODO: remove this function if they really aren't
            //std::lock_guard<decltype(cmd_run_lock_)> lock(cmd_run_lock_);
            whattodo();
        } else {
            whattodo();
        }
    }
    template<typename Server, typename ... Args> void createServer(Args&& ... args) {
        servers_.push_back(make_unique<Server>(std::forward<Args>(args)...));
    }
    void communicate(ClientPipe &pipe) {
        bool disconnect = false;
        {
            std::lock_guard<decltype(server_ready_)> lock(server_ready_);
        }
        while (!disconnect) {
            ControlPacket pkt;
            pipe.from_client.wait_dequeue(pkt);
            if (pkt.type==ControlPacket::Data) {
                std::istringstream line(pkt.data);
                std::ostringstream result;
                try {
                    readExecCommands(line, result, false, true, &disconnect);
                } catch (std::exception &e) {
                    logstream << "BUG: readExecCommands error (should never happen) " << e.what();
                    break;
                }
                pipe.to_client.emplace(ControlPacket::Data, result.str());
                pipe.send_to_client();
            } else if (pkt.type==ControlPacket::Start) {
                pipe.to_client.emplace(ControlPacket::Data, "100 VTR Ready\n");
                pipe.send_to_client();
            } else if (pkt.type==ControlPacket::End) {
                break;
            }
        }
        pipe.to_client.emplace(ControlPacket::End);
        pipe.send_to_client();
    }
    template<typename InStream, typename OutStream> bool readExecCommands(InStream &in, OutStream &out, bool is_terminal = false, bool is_subcommand = false, bool* disconnect = nullptr) {
        bool dowork = true;
        bool all_good = true;
        if (!is_subcommand) {
            if (!is_terminal) {
                std::lock_guard<decltype(server_ready_)> lock(server_ready_);
            }
            //out << "100 VTR Ready\n";
        }
        while (dowork && !in.eof()) {
            std::string cmd, arg;

            // get command:
            in >> cmd;
            cmd = strutils::trim(cmd);
            if (cmd.empty()) {
                continue;
            }
            if (cmd[0]=='#') { // comment
                std::getline(in, arg);
                continue;
            }
            strutils::toLowerInPlace(cmd);
            if (cmd == "bye") {
                out << "BYE\n";
                dowork = false;
                if (disconnect) {
                    *disconnect = true;
                }
                break;
            }

            // get argument:
            std::getline(in, arg);

            auto mngr = manager_;
            if (!mngr) {
                out << "400 manager not ready: " << cmd << "\n";
                logstream << "Command " << cmd << " " << arg << " failed: manager not ready";
                all_good = false;
                continue;
            }

            // handle special commands:
            if (cmd == "retry") {
                do {
                    std::istringstream substream(arg);
                    if (manager_->shouldWork() && (!readExecCommands(substream, out, is_terminal, true))) {
                        wallclock.sleepms(1000);
                    } else {
                        break;
                    }
                } while (true);
                continue;
            }
            if (cmd == "detach") {
                out << "200 OK\n";
                detached_threads_.push_back(start_thread("detached control", [this, arg]() {
                    std::istringstream substream(arg);
                    readExecCommands(substream, std::cout, true, true);
                }));
                continue;
            }

            arg = strutils::trim(arg);

            // execute:
            auto cmditer = commands_.find(cmd);
            if (cmditer == commands_.end()) {
                out << "400 Unknown command: " << cmd << "\n";
            } else {
                auto &handler = cmditer->second;
                logstream << "Executing: " << cmd << " " << arg;
                try {
                    lockOrNot(no_lock_commands_.count(cmd)==0, [&]() {
                        std::ostringstream ss;
                        handler(ss, arg);
                        std::string response = ss.str();
                        /*if (is_terminal) {
                            out << cmd << " " << arg << ": ";
                        }*/
                        if (response.empty()) {
                            out << "200 OK\n";
                        } else {
                            out << "201 OK\n" << response << "\n";
                        }
                        logstream << "Executed successfully " << cmd;
                    });
                } catch (std::exception &e) {
                    all_good = false;
                    logstream << "Command " << cmd << " " << arg << " failed: " << e.what();
                    if (!is_terminal) {
                        out << "500 ERROR: " << e.what() << "\n";
                    }
                }
            }
        }
        return all_good;
    }
    void setReady() {
        server_ready_.unlock();
    }
    void printAllQueues() {
        std::ostringstream ost;
        ost << "Queues: ";
        manager_->edges()->printEdgesStats(ost, true);
        logstream << ost.str();
    }
    void stopGroupAndWait(const std::string& grp) {
        manager_->group(grp)->stopNodesAndWait();
    }
    void clearAllQueues() {
        manager_->edges()->clearEdges();
    }
    void shutdown() {
        logstream << "Closing server sockets";
        servers_.clear();
        if (manager_) {
            InstanceSharedObjectsDestructors::callPreShutdownHooks(&manager_->instanceData());
            logstream << "Shutting down NodeManager";
            manager_->shutdown();
        }
        if (!detached_threads_.empty()) {
            logstream << "Waiting for detached threads";
            for (std::thread &thr: detached_threads_) {
                thr.join();
            }
            detached_threads_.clear();
        }
        if (manager_) {
            if (manager_.use_count() <= 1) {
                logstream << "Destroying NodeManager";
            } else {
                logstream << "Warning: NodeManager is still being used somewhere";
            }
            manager_ = nullptr;
        }
        logstream << APP_VERSION << " says goodbye!";
    }
    template<typename Team>
    auto command_team_link(std::string& args) {
        std::stringstream ss(args);
        std::string team_name, linked_team_name;
        ss >> team_name >> linked_team_name;
        
        std::shared_ptr<Team> team = InstanceSharedObjects<Team>::get(manager_->instanceData(), team_name);
        std::shared_ptr<Team> linked_team = InstanceSharedObjects<Team>::get(manager_->instanceData(), linked_team_name);
        team->linkTeam(linked_team);
    };
    template<typename Team>
    auto command_team_unlink(std::string& args) {
        std::stringstream ss(args);
        std::string team_name, linked_team_name;
        ss >> team_name >> linked_team_name;

        std::shared_ptr<Team> team = InstanceSharedObjects<Team>::get(manager_->instanceData(), team_name);
        std::shared_ptr<Team> linked_team = InstanceSharedObjects<Team>::get(manager_->instanceData(), linked_team_name);
        team->unlinkTeam(linked_team);
    };
    ControlImpl(std::shared_ptr<NodeManager> manager):
        manager_(manager) {
        server_ready_.lock();
        commands_["hello"] = [](ClientStream &cs, std::string&) {
            cs << "HELLO\n";
        };
        no_lock_commands_.insert("hello");
        commands_["version"] = [](ClientStream &cs, std::string&) {
            cs << APP_VERSION << "\n";
        };
        no_lock_commands_.insert("version");
        commands_["node.add"] = [this](ClientStream &cs, std::string &arg) {
            json params = json::parse(arg);
            manager_->createNode(params, false, false);
        };
        commands_["node.add_create"] = [this](ClientStream &cs, std::string &arg) {
            json params = json::parse(arg);
            manager_->createNode(params, true, false);
        };
        commands_["node.add_start"] = [this](ClientStream &cs, std::string &arg) {
            json params = json::parse(arg);
            manager_->createNode(params, true, true);
        };
        commands_["node.delete"] = [this](ClientStream &cs, std::string &arg) {
            manager_->deleteNode(arg);
        };
        commands_["node.start"] = [this](ClientStream &cs, std::string &arg) {
            manager_->node(arg)->start();
        };
        commands_["node.stop"] = [this](ClientStream &cs, std::string &arg) {
            manager_->node(arg)->stop();
        };
        commands_["node.auto_restart"] = [this](ClientStream &cs, std::string &arg) {
            manager_->node(arg)->stop(false);
        };
        commands_["node.interrupt"] = [this](ClientStream &cs, std::string &arg) {
            manager_->node(arg)->interrupt();
        };
        no_lock_commands_.insert("node.interrupt");
        commands_["node.stop_wait"] = [this](ClientStream &cs, std::string &arg) {
            manager_->node(arg)->stopAndWait();
        };
        commands_["node.param.set"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string name, param, content;
            ss >> name >> param;
            std::getline(ss, content);

            name = strutils::trim(name);
            param = strutils::trim(param);
            content = strutils::trim(content);

            auto node = manager_->node(name);
            node->doLocked([&]() {
                if (node->isWorking()) {
                    cs << "WARNING: Node won't accept new parameters until restarted.\n";
                }
                node->parameters()[param] = json::parse(content);
            });
        };
        commands_["node.param.get"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string name, param;
            ss >> name >> param;
            name = strutils::trim(name);
            param = strutils::trim(param);
            if (!param.empty()) {
                cs << manager_->node(name)->parameters()[param] << "\n";
            } else {
                cs << manager_->node(name)->parameters() << "\n";
            }
        };
        commands_["node.object.get"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string node_name, object_name;
            ss >> node_name >> object_name;
            node_name = strutils::trim(node_name);
            object_name = strutils::trim(object_name);
            cs << manager_->node(node_name)->getObject(object_name) << "\n";
        };
        commands_["node.object.set"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string node_name, object_name, content;
            ss >> node_name >> object_name;
            std::getline(ss, content);

            content = strutils::trim(content);

            auto node = manager_->node(node_name);
            node->setObject(object_name, json::parse(content));
        };
        commands_["queue.plan_capacity"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string name;
            size_t capacity;
            ss >> name >> capacity;
            manager_->edges()->planCapacity(name, capacity);
        };
        commands_["queue.drain"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string name;
            ss >> name;
            std::shared_ptr<EdgeBase> edge = manager_->edges()->findAny(name);
            if (!edge) {
                throw Error("No queue with this name");
            }
            edge->waitEmpty();
        };
        commands_["queues.stats"] = [this](ClientStream &cs, std::string&) {
            manager_->edges()->printEdgesStats(cs);
        };
        // Machine-readable JSON statistics for all queues.
        // Returns an array of objects:
        // [
        //   { "name": "videoin", "capacity": 256, "occupied": 12, "free": 244, "last_ts_seconds": 0.123 },
        //   ...
        // ]
        commands_["queues.json"] = [this](ClientStream &cs, std::string&) {
            json j = manager_->edges()->edgesStatsJson();
            cs << j << "\n";
        };
        // Reset per-queue occupancy statistics used by queues.json (frames_in_queue.* fields).
        commands_["queues.stats.reset"] = [this](ClientStream &cs, std::string&) {
            (void)cs;
            manager_->edges()->resetEdgesOccupancyStats();
        };
        // Get statistics for all sync groups (RealTimeTeam instance-shared objects)
        commands_["sync_groups.json"] = [this](ClientStream &cs, std::string&) {
            (void)cs;
            json jgroups = json::array();
            auto teams = InstanceSharedObjects<RealTimeTeam>::enumerate(manager_->instanceData());
            for (const auto &entry : teams) {
                const std::string &name = entry.first;
                std::shared_ptr<RealTimeTeam> team = entry.second;
                if (!team) continue;
                json jgroup;
                jgroup["name"] = name;
                AVTS offset = team->getOffset();
                jgroup["offset"] = (offset == AV_NOPTS_VALUE) ? json(nullptr) : json(offset);
                AVRational tb = team->getTimebase();
                jgroup["timebase_num"] = tb.num;
                jgroup["timebase_den"] = tb.den;
                jgroup["flushing"] = team->isFlushing();
                jgroup["first"] = team->isFirst();
                jgroup["seek_targets_count"] = team->getSeekTargetsCount();
                auto linked_teams = team->getLinkedTeams();
                json jlinked = json::array();
                for (const auto &linked : linked_teams) {
                    // We can't easily get the name of linked teams, so we'll just count them
                }
                jgroup["linked_teams_count"] = linked_teams.size();
                jgroups.push_back(std::move(jgroup));
            }
            cs << jgroups << "\n";
        };
        // Get statistics for all sentinel correction groups (PTSCorrectorCommon instance-shared objects)
        commands_["correction_groups.json"] = [this](ClientStream &cs, std::string&) {
            (void)cs;
            json jgroups = json::array();
            auto correctors = InstanceSharedObjects<PTSCorrectorCommon>::enumerate(manager_->instanceData());
            for (const auto &entry : correctors) {
                const std::string &name = entry.first;
                std::shared_ptr<PTSCorrectorCommon> corr = entry.second;
                if (!corr) continue;
                json jgroup = corr->getStats();
                jgroup["name"] = name;
                jgroups.push_back(std::move(jgroup));
            }
            cs << jgroups << "\n";
        };
        commands_["group.restart"] = [this](ClientStream &cs, std::string &arg) {
            manager_->group(arg)->restartNodes();
        };
        commands_["group.stop"] = [this](ClientStream &cs, std::string &arg) {
            manager_->group(arg)->stopNodes();
        };
        commands_["group.start"] = [this](ClientStream &cs, std::string &arg) {
            manager_->group(arg)->startNodes();
        };
        commands_["group.retry_start"] = [this](ClientStream &cs, std::string &arg) {
            cs << "WARNING: this command is deprecated. please use group.start";
            manager_->group(arg)->startNodes();
        };
        commands_["stats.subscribe"] = [this](ClientStream &cs, std::string &arg) {
            json jargs = json::parse(arg);
            auto ssthr = std::make_shared<StatsSenderThread>(jargs, manager_);
        };
        // Dump current graph (all nodes with their parameters) as JSON array.
        // Each entry has: name, type, working (bool), params (full JSON object).
        commands_["nodes.json"] = [this](ClientStream &cs, std::string &arg) {
            (void)arg;
            json jnodes = json::array();
            for (auto &entry: manager_->allNodes()) {
                const std::string &name = entry.first;
                std::shared_ptr<NodeWrapper> node = entry.second;
                if (!node) {
                    continue;
                }
                json jn;
                jn["name"] = name;
                jn["type"] = node->type();
                jn["working"] = node->isWorking();
                jn["params"] = node->parameters();
                jnodes.push_back(std::move(jn));
            }
            cs << jnodes << "\n";
        };
        auto seek = [this](std::string team_node_name, StreamTarget target) {
            std::shared_ptr<RealTimeTeam> team = InstanceSharedObjects<RealTimeTeam>::get(manager_->instanceData(), team_node_name);
            if (!team) {
                throw Error("unknown team");
            }
            std::shared_ptr<IFlushAndSeek> seekable = std::dynamic_pointer_cast<IFlushAndSeek>(team);
            if (!seekable) {
                throw Error("team can't initiate seeking");
            }
            seekable->flushAndSeek(target);
        };
        auto seek_at = [this](std::string team_node_name, StreamTarget when, StreamTarget target) {
            std::shared_ptr<InputSeekTeam> team = InstanceSharedObjects<InputSeekTeam>::get(manager_->instanceData(), team_node_name);
            if (!team) {
                throw Error("unknown team");
            }
            std::shared_ptr<ISeekAt> seekable = std::dynamic_pointer_cast<ISeekAt>(team);
            if (!seekable) {
                throw Error("team doesn't support seek at commands");
            }
            if (!target.ts.isValid()) {
                seekable->seekAtClear();
            } else {
                seekable->seekAtAdd(when, target);
            }
        };
        commands_["seek"] = [this, seek, seek_at](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string t1, t2;
            std::string sink_name;
            std::string command;
            ss >> sink_name;
            ss >> command;
            if (command == "now") {
                ss >> t1;
                seek(sink_name, StreamTarget::from_string(t1));
            } else if (command == "at") {
                ss >> t1 >> t2;
                seek_at(sink_name, StreamTarget::from_string(t1), StreamTarget::from_string(t2));
            } else if (command == "frame") {
                ss >> t1;
                if (!t1.empty()) {
                    int64_t frame_number = std::stoll(t1);
                    if ((t1[0] == '+') || (t1[0] == '-')) {
                        // relative seek
                        seek(sink_name, StreamTarget::from_frames_relative(frame_number));
                    } else {
                        // absolute seek
                        seek(sink_name, StreamTarget::from_frames_absolute(frame_number));
                    }
                }
                seek_at(sink_name, StreamTarget::from_string(t1), StreamTarget::from_string(t2));
            } else if (command == "clear") {
                seek_at(sink_name, {}, {});
            } else if (command == "live") {
                seek(sink_name, StreamTarget::live());
            } else {
                throw Error("invalid command parameters");
            }
        };
        commands_["pause"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string t, command;
            ss >> t >> command;
            std::shared_ptr<PauseControlTeam> team = InstanceSharedObjects<PauseControlTeam>::get(manager_->instanceData(), t);
            if (command == "now") {
                team->pause();
            } else if (command == "at") {
                std::string at;
                ss >> at;
                team->pause(StreamTarget::from_string(at));
            } else {
                throw Error("invalid command parameters");
            }
        };
        commands_["team.link<pause>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_link<PauseControlTeam>(arg);
        };
        commands_["team.link<speed>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_link<SpeedControlTeam>(arg);
        };
        commands_["team.link<realtime>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_link<RealTimeTeam>(arg);
        };
        commands_["team.unlink<pause>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_unlink<PauseControlTeam>(arg);
        };
        commands_["team.unlink<speed>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_unlink<SpeedControlTeam>(arg);
        };
        commands_["team.unlink<realtime>"] = [this](ClientStream &cs, std::string &arg) {
            command_team_unlink<RealTimeTeam>(arg);
        };
        commands_["resume"] = [this](ClientStream &cs, std::string &arg) {
            std::shared_ptr<PauseControlTeam> team = InstanceSharedObjects<PauseControlTeam>::get(manager_->instanceData(), arg);
            team->resume();
        };
        commands_["output.start"] = [this](ClientStream &cs, std::string &args) {
            OutputControl::get(args, false)->start();
        };
        no_lock_commands_.insert("output.start");
        commands_["output.stop"] = [this](ClientStream &cs, std::string &args) {
            OutputControl::get(args, false)->stop();
        };
        no_lock_commands_.insert("output.stop");
        commands_["hwaccel.init"] = [this](ClientStream &cs, std::string &arg) {
            json jargs = json::parse(arg);
            initHWAccel(manager_->instanceData(), jargs);
        };
        commands_["event.wait"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string event_name;
            ss >> event_name;
            std::shared_ptr<NamedEvent> ev = InstanceSharedObjects<NamedEvent>::get(manager_->instanceData(), event_name);
            logstream << "Waiting for event " << event_name;
            ev->event().wait();
            logstream << "Done waiting for event " << event_name;
        };
        commands_["event.on.node.finished"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string event_name;
            std::string node_name;
            ss >> event_name >> node_name;
            std::shared_ptr<NamedEvent> ev = InstanceSharedObjects<NamedEvent>::get(manager_->instanceData(), event_name);
            auto node = manager_->node(node_name);
            node->onFinished([ev](std::shared_ptr<NodeWrapper>, bool) {
                ev->event().signal();
            });
        };
        commands_["realtime.team.reset"] = [this](ClientStream &cs, std::string &arg) {
            std::shared_ptr<RealTimeTeam> team = InstanceSharedObjects<RealTimeTeam>::get(manager_->instanceData(), arg);
            team->reset();
        };
        commands_["realtime.team.set_delay"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string team_name;
            float delay_sec;
            ss >> team_name >> delay_sec;
            std::shared_ptr<RealTimeTeam> team = InstanceSharedObjects<RealTimeTeam>::get(manager_->instanceData(), team_name);
            team->setUserDelay(delay_sec);
        };
        commands_["speed.set"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string team_name;
            float speed;
            ss >> team_name >> speed;
            std::shared_ptr<SpeedControlTeam> team = InstanceSharedObjects<SpeedControlTeam>::get(manager_->instanceData(), team_name);
            team->setSpeed(speed);
        };

        // timeline.set {"name":"mixer_tl","ch":"otm_cam1","at":1234567,"key":"outputs","val":3}
        commands_["timeline.set"] = [this](ClientStream &cs, std::string &arg) {
            json req = json::parse(arg);
            if (!req.is_object())
                throw Error("timeline.set: expected JSON object");
            std::string tl_name = req.at("name").get<std::string>();
            std::string channel = req.contains("ch")
                                      ? req.at("ch").get<std::string>()
                                      : req.at("channel").get<std::string>();
            std::string key = req.at("key").get<std::string>();
            int64_t at_pts_ms = req.at("at").get<int64_t>();
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), tl_name);
            tl->set(channel, key, at_pts_ms, req.at("val"));
        };

        // timeline.batch <name> <entries_json_array>
        // Each entry: {"ch":"...", "at":int64, "key":"...", "val":...}
        commands_["timeline.batch"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string tl_name, entries_str;
            ss >> tl_name;
            std::getline(ss, entries_str);
            entries_str = strutils::trim(entries_str);
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), tl_name);
            json entries = json::parse(entries_str);
            if (!entries.is_array())
                throw Error("timeline.batch: expected JSON array");
            std::vector<SharedTimeline::BatchEntry> batch;
            batch.reserve(entries.size());
            for (const auto& e : entries) {
                batch.push_back(SharedTimeline::BatchEntry{
                    e.at("ch").get<std::string>(),
                    e.at("key").get<std::string>(),
                    e.at("at").get<int64_t>(),
                    e.at("val")
                });
            }
            tl->setBatch(batch);
        };

        // timeline.clear <name> [channel]
        commands_["timeline.clear"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string tl_name, channel;
            ss >> tl_name;
            if (ss >> channel) {
                channel = strutils::trim(channel);
            }
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), tl_name);
            if (channel.empty())
                tl->clearAll();
            else
                tl->clear(channel);
        };

        // timeline.gc <name> <before_pts_ms>
        commands_["timeline.gc"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string tl_name;
            int64_t before_pts_ms;
            ss >> tl_name >> before_pts_ms;
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), tl_name);
            tl->gc(before_pts_ms);
        };

        // timeline.dump <name>
        commands_["timeline.dump"] = [this](ClientStream &cs, std::string &arg) {
            std::string tl_name = strutils::trim(arg);
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), tl_name);
            cs << tl->dump() << "\n";
        };

        auto mixerOrchestrator = [this](const std::string& mixer_name) {
            auto state = InstanceSharedObjects<MixerState>::get(manager_->instanceData(), mixer_name);
            auto tl = InstanceSharedObjects<SharedTimeline>::get(manager_->instanceData(), state->timeline_name);
            auto scheduler = InstanceSharedObjects<MixerTransitionScheduler>::get(manager_->instanceData(), mixer_name);
            return MixerOrchestrator(manager_->shared_from_this(), state, tl, scheduler);
        };

        auto mixerJsonRequest = [](const std::string& command, const std::string& arg) {
            json req = json::parse(arg);
            if (!req.is_object())
                throw Error(command + ": expected JSON object");
            return req;
        };

        // mixer.source <name> <otm_node> <input_index> <cs_node_a> <cs_node_b>
        commands_["mixer.source"] = [this, mixerOrchestrator](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string mixer_name, src_name, otm_node, cs_a, cs_b;
            int input_index;
            ss >> mixer_name >> src_name >> otm_node >> input_index >> cs_a >> cs_b;
            auto orch = mixerOrchestrator(mixer_name);
            orch.defineSource(src_name, otm_node, input_index, cs_a, cs_b);
        };

        // mixer.routed_source {"mixer":"...","name":"...","router":"...","input_index":0,
        //                      "route_label_a":"...","route_label_b":"...",
        //                      "cs_node_a":"...","cs_node_b":"..."}
        commands_["mixer.routed_source"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            std::string trimmed = strutils::trim(arg);
            std::string mixer_name, src_name, router_node, route_label_a, route_label_b, cs_a, cs_b;
            int input_index;
            if (!trimmed.empty() && trimmed[0] == '{') {
                json req = mixerJsonRequest("mixer.routed_source", arg);
                mixer_name = req.at("mixer").get<std::string>();
                src_name = req.at("name").get<std::string>();
                router_node = req.at("router").get<std::string>();
                input_index = req.at("input_index").get<int>();
                route_label_a = req.at("route_label_a").get<std::string>();
                route_label_b = req.at("route_label_b").get<std::string>();
                cs_a = req.at("cs_node_a").get<std::string>();
                cs_b = req.at("cs_node_b").get<std::string>();
            } else {
                std::stringstream ss(arg);
                int route_out_a, route_out_b;
                ss >> mixer_name >> src_name >> router_node >> input_index >> route_out_a >> route_out_b >> cs_a >> cs_b;
                route_label_a = std::to_string(route_out_a);
                route_label_b = std::to_string(route_out_b);
            }
            auto orch = mixerOrchestrator(mixer_name);
            orch.defineRoutedSource(src_name, router_node, input_index, route_label_a, route_label_b, cs_a, cs_b);
        };

        // mixer.scene <mixer_name> <scene_name> <json_definition>
        commands_["mixer.scene"] = [this, mixerOrchestrator](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string mixer_name, scene_name, def_str;
            ss >> mixer_name >> scene_name;
            std::getline(ss, def_str);
            def_str = strutils::trim(def_str);
            json jdef = json::parse(def_str);

            SceneDefinition def;
            def.name = scene_name;
            if (jdef.contains("width")) def.width = jdef["width"].get<int>();
            if (jdef.contains("height")) def.height = jdef["height"].get<int>();
            if (!jdef.contains("sources") || !jdef["sources"].is_object())
                throw Error("mixer.scene: sources object required (logical source name -> { graph, dst_x, dst_y, ... })");
            for (auto it = jdef["sources"].begin(); it != jdef["sources"].end(); ++it) {
                SourceLayout sl;
                const json& inj = it.value();
                sl.crop_scale_graph = inj.value("graph", std::string(""));
                Parameters layer = json::object();
                for (auto jt = inj.begin(); jt != inj.end(); ++jt) {
                    if (jt.key() != "graph")
                        layer[jt.key()] = jt.value();
                }
                if (layer.empty())
                    layer = {{"dst_x", 0}, {"dst_y", 0}};
                sl.layer = std::move(layer);
                def.sources[it.key()] = std::move(sl);
            }
            if (jdef.contains("controls")) {
                if (!jdef["controls"].is_array())
                    throw Error("mixer.scene: controls must be an array");
                for (const auto& c : jdef["controls"]) {
                    if (!c.is_object())
                        throw Error("mixer.scene: control entries must be objects");
                    SceneControl sc;
                    sc.node_name = c.at("node").get<std::string>();
                    sc.key = c.at("key").get<std::string>();
                    sc.value = c.at("value");
                    def.controls.push_back(std::move(sc));
                }
            }
            if (jdef.contains("routes")) {
                if (!jdef["routes"].is_object())
                    throw Error("mixer.scene: routes must be an object");
                for (auto it = jdef["routes"].begin(); it != jdef["routes"].end(); ++it)
                    def.routes[it.key()] = it.value().get<int>();
            }

            auto orch = mixerOrchestrator(mixer_name);
            orch.defineScene(scene_name, def);
        };

        // mixer.init_routes {"mixer":"..."}
        commands_["mixer.init_routes"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            std::string trimmed = strutils::trim(arg);
            std::string mixer_name;
            if (!trimmed.empty() && trimmed[0] == '{') {
                json req = mixerJsonRequest("mixer.init_routes", arg);
                mixer_name = req.at("mixer").get<std::string>();
            } else {
                mixer_name = trimmed;
            }
            auto orch = mixerOrchestrator(mixer_name);
            orch.initializeRoutedRoutes();
        };

        // mixer.preview {"mixer":"mixer","scene":"scene_name"}
        commands_["mixer.preview"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.preview", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            std::string scene_name = req.at("scene").get<std::string>();
            auto orch = mixerOrchestrator(mixer_name);
            orch.preview(scene_name);
        };

        // mixer.cut {"mixer":"mixer","scene":"scene_name","start_pts_ms":123456789}
        commands_["mixer.cut"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.cut", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            std::string scene_name = req.at("scene").get<std::string>();
            int64_t start_pts_ms = req.value("start_pts_ms", int64_t(-1));
            auto orch = mixerOrchestrator(mixer_name);
            orch.cut(scene_name, start_pts_ms);
        };

        // mixer.fade {"mixer":"mixer","scene":"scene_name","duration_sec":2.0,"start_pts_ms":123456789}
        commands_["mixer.fade"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.fade", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            std::string scene_name = req.at("scene").get<std::string>();
            double duration_sec = req.value("duration_sec", 1.0);
            int64_t start_pts_ms = req.value("start_pts_ms", int64_t(-1));
            if (duration_sec <= 0)
                throw Error("mixer.fade: duration_sec must be > 0");
            auto orch = mixerOrchestrator(mixer_name);
            orch.fade(scene_name, duration_sec, start_pts_ms);
        };

        // mixer.wipe {"mixer":"mixer","scene":"scene_name","wipe_file":"/path/with spaces.mov","duration_sec":2.0,"start_pts_ms":123456789}
        commands_["mixer.wipe"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.wipe", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            std::string scene_name = req.at("scene").get<std::string>();
            std::string wipe_file = req.at("wipe_file").get<std::string>();
            double duration_sec = req.value("duration_sec", 0.0);
            int64_t start_pts_ms = req.value("start_pts_ms", int64_t(-1));
            if (!req.contains("duration_sec"))
                duration_sec = probeMediaDurationSec(wipe_file);
            if (duration_sec <= 0)
                throw Error("mixer.wipe: duration_sec must be > 0");
            auto orch = mixerOrchestrator(mixer_name);
            orch.wipe(scene_name, wipe_file, duration_sec, start_pts_ms);
        };

        // mixer.overlay.init {"mixer":"mixer","source_otm":"otm_html_overlay_src",
        //                     "overlay_otm":"otm_html_overlay","selector":"overlay_sel"}
        commands_["mixer.overlay.init"] = [this, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.overlay.init", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            auto state = InstanceSharedObjects<MixerState>::get(manager_->instanceData(), mixer_name);
            std::lock_guard<std::mutex> lock(state->mutex);
            state->overlay_source_otm_name = req.at("source_otm").get<std::string>();
            state->overlay_otm_name = req.at("overlay_otm").get<std::string>();
            state->overlay_selector_name = req.at("selector").get<std::string>();
            state->overlay_enabled = req.value("enabled", false);
            if (req.contains("ready_timeout_ms"))
                state->overlay_ready_timeout_ms = req.at("ready_timeout_ms").get<int64_t>();
            if (req.contains("ready_poll_ms"))
                state->overlay_ready_poll_ms = req.at("ready_poll_ms").get<int64_t>();
            if (state->overlay_ready_timeout_ms < 0)
                throw Error("mixer.overlay.init: ready_timeout_ms must be >= 0");
            if (state->overlay_ready_poll_ms <= 0)
                throw Error("mixer.overlay.init: ready_poll_ms must be > 0");
        };

        // mixer.overlay {"mixer":"mixer","enabled":true,"ready_timeout_ms":1000}
        commands_["mixer.overlay"] = [this, mixerOrchestrator, mixerJsonRequest](ClientStream &cs, std::string &arg) {
            json req = mixerJsonRequest("mixer.overlay", arg);
            std::string mixer_name = req.at("mixer").get<std::string>();
            bool enabled = req.at("enabled").get<bool>();
            int64_t ready_timeout_ms = req.value("ready_timeout_ms", int64_t(-1));
            if (req.contains("source_otm") || req.contains("overlay_otm") || req.contains("selector")) {
                if (!req.contains("source_otm") || !req.contains("overlay_otm") || !req.contains("selector"))
                    throw Error("mixer.overlay: source_otm, overlay_otm, and selector must be provided together");
                auto state = InstanceSharedObjects<MixerState>::get(manager_->instanceData(), mixer_name);
                std::lock_guard<std::mutex> lock(state->mutex);
                state->overlay_source_otm_name = req.at("source_otm").get<std::string>();
                state->overlay_otm_name = req.at("overlay_otm").get<std::string>();
                state->overlay_selector_name = req.at("selector").get<std::string>();
            }
            auto orch = mixerOrchestrator(mixer_name);
            orch.setOverlayEnabled(enabled, ready_timeout_ms);
        };

        // mixer.status <mixer_name>
        commands_["mixer.status"] = [this, mixerOrchestrator](ClientStream &cs, std::string &arg) {
            std::string mixer_name = strutils::trim(arg);
            auto orch = mixerOrchestrator(mixer_name);
            cs << orch.status() << "\n";
        };

        // mixer.scenes <mixer_name>
        // Returns a JSON array of scene names registered with this mixer, sorted alphabetically.
        commands_["mixer.scenes"] = [this, mixerOrchestrator](ClientStream &cs, std::string &arg) {
            std::string mixer_name = strutils::trim(arg);
            auto orch = mixerOrchestrator(mixer_name);
            json arr = orch.sceneNames();
            cs << arr << "\n";
        };

        // mixer.init <mixer_name> <json_config>
        // Initialize MixerState with slot node names and global settings.
        commands_["mixer.init"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string mixer_name, config_str;
            ss >> mixer_name;
            std::getline(ss, config_str);
            config_str = strutils::trim(config_str);
            json cfg = json::parse(config_str);

            auto state = InstanceSharedObjects<MixerState>::get(manager_->instanceData(), mixer_name);
            std::lock_guard<std::mutex> lock(state->mutex);

            if (cfg.contains("timeline")) state->timeline_name = cfg["timeline"].get<std::string>();
            if (cfg.contains("hwaccel")) state->hwaccel_name = cfg["hwaccel"].get<std::string>();
            if (cfg.contains("fps_num")) state->fps_num = cfg["fps_num"].get<int>();
            if (cfg.contains("fps_den")) state->fps_den = cfg["fps_den"].get<int>();
            if (cfg.contains("switch_margin_ms")) {
                state->switch_margin_ms = cfg["switch_margin_ms"].get<int64_t>();
                if (state->switch_margin_ms < 0)
                    throw Error("mixer.init: switch_margin_ms must be >= 0");
            }
            if (cfg.contains("source_switcher")) state->source_switcher_name = cfg["source_switcher"].get<std::string>();
            if (cfg.contains("initial_pgm_scene")) state->pgm_scene_name = cfg["initial_pgm_scene"].get<std::string>();
            if (cfg.contains("initial_pvw_scene")) state->pvw_scene_name = cfg["initial_pvw_scene"].get<std::string>();
            if (cfg.contains("initial_pgm_slot")) {
                std::string slot = cfg["initial_pgm_slot"].get<std::string>();
                if (slot == "A")
                    state->pgm_is_slot_a = true;
                else if (slot == "B")
                    state->pgm_is_slot_a = false;
                else
                    throw Error("mixer.init: initial_pgm_slot must be 'A' or 'B'");
            }

            if (cfg.contains("slot_a")) {
                auto& sa = cfg["slot_a"];
                state->slot_a.compositor_name = sa.value("compositor", std::string(""));
                state->slot_a.norm_ts_name = sa.value("norm_ts", std::string(""));
                state->slot_a.post_otm_name = sa.value("post_otm", std::string(""));
            }
            if (cfg.contains("slot_b")) {
                auto& sb = cfg["slot_b"];
                state->slot_b.compositor_name = sb.value("compositor", std::string(""));
                state->slot_b.norm_ts_name = sb.value("norm_ts", std::string(""));
                state->slot_b.post_otm_name = sb.value("post_otm", std::string(""));
            }
            if (cfg.contains("wipe_otm")) state->wipe_otm_name = cfg["wipe_otm"].get<std::string>();
            if (cfg.contains("wipe_base_fps")) state->wipe_base_fps_name = cfg["wipe_base_fps"].get<std::string>();
            if (cfg.contains("wipe_selector")) state->wipe_selector_name = cfg["wipe_selector"].get<std::string>();
            if (cfg.contains("wipe_group")) state->wipe_group_name = cfg["wipe_group"].get<std::string>();
            if (cfg.contains("wipe_input_node")) state->wipe_input_node_name = cfg["wipe_input_node"].get<std::string>();
            if (cfg.contains("wipe_tail_edge")) state->wipe_tail_edge = cfg["wipe_tail_edge"].get<std::string>();
            if (cfg.contains("wipe_flush_edges")) {
                state->wipe_flush_edges.clear();
                for (const auto& e : cfg["wipe_flush_edges"])
                    state->wipe_flush_edges.push_back(e.get<std::string>());
            }
        };

        #ifdef EMBED_IN_OBS
        std::shared_ptr<EventLoop> evl = InstanceSharedObjects<EventLoop>::get(manager_->instanceData(), "obs_tick");
        tick_source_ = std::make_shared<TickSource>(evl);
        InstanceSharedObjects<TickSource>::put(manager_->instanceData(), "obs", tick_source_);
        #endif
    }
    #ifdef EMBED_IN_OBS
    void tick() {
        //logstream << "tick!";
        tick_source_->fastTick();
    }
    #endif
};

class TcpControlServer: public ControlServerBase {
    struct Client: public std::enable_shared_from_this<Client> {
        ControlImpl &control;
        TcpControlServer &server;
        std::list<std::shared_ptr<Client>>::iterator iter;
        boost::asio::io_service &io_service;
        tcp::socket socket;
        boost::asio::streambuf buff;
        ClientPipe pipe;
        std::thread thread;
        bool closing = false;
        Client(ControlImpl &_control, TcpControlServer &_server, boost::asio::io_service &_io_service):
            control(_control), server(_server), io_service(_io_service), socket(_io_service),
            pipe([this]() {
                postToClient();
            }) {
        };
        void start() {
            auto self = shared_from_this();
            thread = start_thread("control", [self]() {
                self->control.communicate(self->pipe);
            });
        }
        void postToClient() {
            auto self = shared_from_this();
            io_service.post([self]() {
                self->sendToClient();
            });
        }
        void sendToClient() {
            ControlPacket pkt;
            if (!pipe.to_client.try_dequeue(pkt)) {
                logstream << "BUG: nothing in to_client queue but send_to_client was called";
                return;
            }
            if (pkt.type==ControlPacket::Data) {
                auto data = std::make_shared<std::string>(std::move(pkt.data));
                auto self = shared_from_this();
                boost::asio::async_write(socket, boost::asio::buffer(*data), [self, data](const boost::system::error_code& error, const size_t) {
                    if (error) {
                        logstream << "send error: " << error;
                    }
                });
            } else if (pkt.type==ControlPacket::End) {
                closeSocket();
            }
        }
        void closeSocket() {
            try {
                socket.close();
            } catch (std::exception &e) {
            }
        }
        void closeAndRemove() {
            if (closing) {
                return;
            }
            closing = true;
            closeSocket();
            pipe.from_client.emplace(ControlPacket::End);
            if (thread.joinable()) {
                thread.join();
            }
            server.clients_.erase(iter);
        }
        void receiveNextLine() {
            auto self = shared_from_this();
            boost::asio::async_read_until(socket, buff, '\n', [self](const boost::system::error_code& error, size_t size) {
                if (error) {
                    logstream << "line receive error: " << error;
                    self->closeAndRemove();
                    return;
                }
                auto buff_begin = boost::asio::buffers_begin(self->buff.data());
                std::string line(buff_begin, buff_begin+size);
                self->buff.consume(size);
                self->pipe.from_client.emplace(ControlPacket::Data, line);
                // response is handled by posting into the event loop
                self->receiveNextLine();
            });
        }
    };

    ControlImpl &control_;
    boost::asio::io_service io_service_;
    tcp::acceptor acceptor_;
    std::list<std::shared_ptr<Client>> clients_;
    std::thread net_thread_;

    void nextConnection() {
        auto client = std::make_shared<Client>(control_, *this, io_service_);
        clients_.push_front(client);
        auto iter = clients_.begin();
        client->iter = iter;
        acceptor_.async_accept(client->socket, [this, client](const boost::system::error_code& error) {
            if (error) {
                logstream << "connection accept error: " << error;
                clients_.erase(client->iter);
                return;
            }
            client->start();
            client->pipe.from_client.emplace(ControlPacket::Start);
            client->receiveNextLine();
            nextConnection();
        });
    }
    void netThread() {
        nextConnection();
        io_service_.run();
    }
public:
    TcpControlServer(ControlImpl &control, uint16_t tcp_port):
        control_(control),
        acceptor_(io_service_, tcp::endpoint(tcp::v4(), tcp_port)),
        net_thread_(start_thread("control net IO", [this]() { netThread(); }))
        {
    }
    virtual ~TcpControlServer() {
        acceptor_.cancel();
        io_service_.stop();
        net_thread_.join();
        for (auto& client: clients_) {
            client->closeSocket();
            client->pipe.from_client.emplace(ControlPacket::End);
        }
        for (auto& client: clients_) {
            if (client->thread.joinable()) {
                client->thread.join();
            }
        }
    }
};


AVPlumber::AVPlumber() {
    if (current_thread.name=="?") {
        set_thread_name("avplumber main");
    }
    av::init();
    av::set_logging_level(AV_LOG_INFO);
    std::shared_ptr<NodeManager> nm = std::make_shared<NodeManager>();
    impl_ = new ControlImpl(nm);
    control_port_ = 0;
    webui_heartbeat_stop_ = false;
}

AVPlumber::~AVPlumber() {
    // Stop heartbeat thread
    webui_heartbeat_stop_ = true;
    if (webui_heartbeat_thread_.joinable()) {
        webui_heartbeat_thread_.join();
    }
    delete impl_;
    impl_ = nullptr;
}

#ifdef EMBED_IN_OBS
void AVPlumber::setObsSource(obs_source_t* obssrc) {
    impl_->manager()->instanceData().obs_source_ = obssrc;
}

void AVPlumber::unsetObsSourceAndWait() {
    if (!impl_ || !impl_->manager())
        return;
    InstanceData &inst = impl_->manager()->instanceData();
    inst.obs_source_.store(nullptr);
    while (inst.obs_source_used_by_.load()!=0) {
        wallclock.sleepms(30);
    }
}

void AVPlumber::obsTick() {
    if (!impl_)
        return;
    impl_->tick();
}

void AVPlumber::get_pause_team_name() {
    if (!impl_ || !impl_->manager())
        return;
    auto node = impl_->manager()->node_if_exists(PAUSE_NODE);
    if (node) {
        auto p = node->parameters();
        if (p.contains("team")) {
            PAUSE_TEAM_ = p["team"];
        }
    }
}

void AVPlumber::get_realtime_team_name() {
    if (!impl_ || !impl_->manager())
        return;
    auto node = impl_->manager()->node_if_exists(REALTIME_NODE);
    if (node) {
        auto p = node->parameters();
        if (p.contains("team")) {
            REALTIME_TEAM_ = p["team"];
        }
    }
}

void AVPlumber::obs_pause() {
    if (PAUSE_TEAM_.empty()) {
        get_pause_team_name();
    }
    if (!PAUSE_TEAM_.empty()) {
        char cmd[128];
        sprintf(cmd, "pause %s now", PAUSE_TEAM_.c_str());
        executeCommandsFromString(cmd);
    }
}

bool AVPlumber::obs_is_paused() {
    if (!impl_ || !impl_->manager())
        return false;
    auto node = impl_->manager()->node_if_exists(PAUSE_NODE);
    if (node) {
        auto p = node->parameters();
        if (p.contains("paused")) {
            bool paused = p["paused"];
            return paused;
        }
    }

    return false;
}

void AVPlumber::obs_play() {
    if (!impl_ || !impl_->manager())
        return;
    if (PAUSE_TEAM_.empty()) {
        get_pause_team_name();
    }
    if (!PAUSE_TEAM_.empty()) {
        char cmd[128];
        sprintf(cmd, "resume %s", PAUSE_TEAM_.c_str());
        executeCommandsFromString(cmd);
    }
}

int64_t AVPlumber::obs_get_time() {
    if (!impl_ || !impl_->manager())
        return -1;
    auto node = impl_->manager()->node_if_exists(REALTIME_NODE);
    if (node) {
        std::shared_ptr<Node> n;
        if (!node->doLockedTry([&]() { n = node->node(); })) {
            return -1;
        }
        if (n) {
            auto node_ts = dynamic_cast<IFrameTimestamp*>(n.get());
            if (node_ts) {
                return node_ts->getCurrentFrameTimestamp();
            }
        }
    }
    return -1;
}

void AVPlumber::obs_set_time(int64_t ms) {
    if (REALTIME_TEAM_.empty()) {
        get_realtime_team_name();
    }
    if (!REALTIME_TEAM_.empty()) {
        char command[128];
        sprintf(command, "seek %s now %ld", REALTIME_TEAM_.c_str(), ms);
        executeCommandsFromString(command);
    }
}

void AVPlumber::obs_stop() {
    executeCommandsFromString("group.stop g1");
}

void AVPlumber::obs_restart() {
    if (!impl_)
        return;
    impl_->stopGroupAndWait("g1");
    impl_->clearAllQueues();
    executeCommandsFromString("group.start g1");
}

int64_t AVPlumber::obs_get_duration() {
    if (!impl_ || !impl_->manager())
        return -1;
    auto node = impl_->manager()->node_if_exists(INPUT_NODE);
    if (node) {
        auto n_rec = dynamic_cast<IPlaybackControl*>(node->node().get());
        if (!n_rec) {
            // this is simple input, not recording input
            // so stream-limits are not supported
            // prevent polluting the log with exception
            return -1;
        }
        try {
            Parameters duration;
            if (!node->getObjectTry("stream-limits", duration)) {
                return -1;
            }
            return duration["duration"];
        } catch (const std::exception &) {
            return -1;
        }
    }

    return -1;
}
double AVPlumber::obs_get_speed() {
    if (!impl_ || !impl_->manager())
        return 1.00;
    auto node = impl_->manager()->node_if_exists(SPEED_NODE);
    if (node) {
        try {
            Parameters info;
            if (!node->getObjectTry("info", info)) {
                return 1.00;
            }
            return info["speed"];
        } catch (const std::exception &) {
            return 1.00;
        }
    }

    return 1.00;
}
bool AVPlumber::obs_is_eof() {
    if (!impl_ || !impl_->manager())
        return false;
    auto node = impl_->manager()->node_if_exists(REALTIME_NODE);
    if (node) {
        std::shared_ptr<Node> n;
        if (!node->doLockedTry([&]() { n = node->node(); })) {
            return false;
        }
        if (n) {
            auto node_ts = dynamic_cast<IFrameTimestamp*>(n.get());
            if (node_ts) {
                return node_ts->isEof();
            }
        }
    }
    return false;
}
#endif

void AVPlumber::enableControlServer(const uint16_t tcp_port) {
    control_port_ = tcp_port;
    if (tcp_port) {
        logstream << "Enabling control server on TCP port " << tcp_port;
        impl_->createServer<TcpControlServer>(*impl_, tcp_port);
    } // if port==0, then NOOP
}

void AVPlumber::registerWithWebUI(const std::string& webui_api_url, const std::string& instance_name, const std::string& log_file) {
    if (webui_api_url.empty() || control_port_ == 0) {
        return; // No web UI URL provided or control server not enabled
    }
    
    // Stop existing heartbeat thread if any
    webui_heartbeat_stop_ = true;
    if (webui_heartbeat_thread_.joinable()) {
        webui_heartbeat_thread_.join();
    }
    
    // Store webui configuration
    webui_api_url_ = webui_api_url;
    instance_name_ = instance_name;
    log_file_ = log_file;
    
    // Start heartbeat thread
    webui_heartbeat_stop_ = false;
    webui_heartbeat_thread_ = start_thread("webui heartbeat", [this]() {
        this->webuiHeartbeatThread();
    });
    
    logstream << "Started web UI heartbeat to " << webui_api_url;
}

void AVPlumber::webuiHeartbeatThread() {
    int heartbeat_interval_seconds = 25;
    const char* heartbeat_interval_seconds_str = getenv("AVPLUMBER_UI_HEARTBEAT_INTERVAL");
    if (heartbeat_interval_seconds_str && heartbeat_interval_seconds_str[0] != '\0') {
        heartbeat_interval_seconds = atoi(heartbeat_interval_seconds_str);
    }
    
    try {
        RESTEndpoint endpoint(webui_api_url_);
        
        // Build the instance heartbeat JSON
        json instance_data;
        instance_data["port"] = control_port_;
        instance_data["host"] = "127.0.0.1"; // Default to localhost
        if (!instance_name_.empty()) {
            instance_data["name"] = instance_name_;
        }
        if (!log_file_.empty()) {
            instance_data["logFile"] = log_file_;
        }
        
        std::string json_str = instance_data.dump();
        
        // Send initial heartbeat immediately
        endpoint.send("/api/instances/heartbeat", json_str);
        
        // Then send heartbeats every 25 seconds
        while (!webui_heartbeat_stop_) {
            std::this_thread::sleep_for(std::chrono::seconds(heartbeat_interval_seconds));
            if (webui_heartbeat_stop_) {
                break;
            }
            try {
                endpoint.send("/api/instances/heartbeat", json_str);
            } catch (std::exception &e) {
                logstream << "Failed to send web UI heartbeat: " << e.what();
            }
        }
    } catch (std::exception &e) {
        logstream << "Web UI heartbeat thread error: " << e.what();
    }
}

void AVPlumber::executeCommandsFromFile(const std::string path) {
    std::ifstream ifs(path);
    impl_->readExecCommands(ifs, std::cout, true);
}

void AVPlumber::executeCommandsFromString(const std::string script) {
    std::istringstream iss(script);
    impl_->readExecCommands(iss, std::cout, true);
}

void AVPlumber::registerControlCommand(
        const std::string& command,
        std::function<std::string(const std::string&)> handler,
        bool no_lock) {
    impl_->registerCommand(command, std::move(handler), no_lock);
}

void AVPlumber::setLogFile(const std::string path) {
    if (path.empty()) {
        current_thread.logger = default_logger;
    } else {
        try {
            current_thread.logger = std::make_shared<FileLogger>(path);
        } catch (std::exception &e) {
            logstream << "Failed to open log file " << path << ": " << e.what();
        }
    }
}

void AVPlumber::setLogCallback(std::function<void(const std::string &)> callback) {
    if (callback) {
        current_thread.logger = std::make_shared<CallbackLogger>(callback);
    } else {
        current_thread.logger = default_logger;
    }
}

void AVPlumber::setExceptionCallback(std::function<void(const std::string&, const std::string&, const std::string&)> callback) {
    auto mngr = manager();
    if (!mngr) {
        return;
    }
    mngr->instanceData().setExceptionCallback(std::move(callback));
}

void AVPlumber::setReady() {
    logstream << APP_VERSION << " READY." << std::endl;
    impl_->setReady();
}

void AVPlumber::shutdown() {
    impl_->shutdown();
}

std::shared_ptr<NodeManager> AVPlumber::manager() {
    return impl_->manager();
}

void AVPlumber::mainLoop() {
    setReady();
    do {
        heartbeat();
    } while (impl_->manager()->shutdownCompleteEvent().wait(3000) == 0);
    
    // when NodeManager shutdown is complete, all nodes are destroyed, but we also need to destroy control servers, DO IT:
    shutdown();
}

void AVPlumber::stopMainLoop() {
    impl_->manager()->shutdown();
}

void AVPlumber::heartbeat() {
    impl_->printAllQueues();
}
