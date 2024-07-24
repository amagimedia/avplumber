#include "avplumber.hpp"

#include <list>
#include <limits>
#include <fstream>
#include <iostream>
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
#ifdef EMBED_IN_OBS
    #include "instance_shared.hpp"
    #include "TickSource.hpp"
    #include "EventLoop.hpp"
#endif

#include <avcpp/av.h>
#include <avcpp/avutils.h>

using boost::asio::ip::tcp;
using nlohmann::json;

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
    void shutdown() {
        logstream << "Closing server sockets";
        servers_.clear();
        logstream << "Shutting down NodeManager";
        manager_->shutdown();
        logstream << "Waiting for detached threads";
        for (std::thread &thr: detached_threads_) {
            thr.join();
        }
        logstream << APP_VERSION << " says goodbye!";
    }
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
        auto seek = [this](std::string sink_name, SeekTarget target) {
            std::shared_ptr<NodeWrapper> sink_nw = manager_->node(sink_name);
            if (!sink_nw) {
                throw Error("unknown node");
            }
            std::shared_ptr<Node> sink_node = sink_nw->node();
            if (!sink_node) {
                throw Error("node not running");
            }
            std::shared_ptr<IFlushAndSeek> seekable = std::dynamic_pointer_cast<IFlushAndSeek>(sink_node);
            if (!seekable) {
                throw Error("node can't initiate seeking");
            }
            seekable->flushAndSeek(target);
        };
        commands_["seek.bytes"] = [this, seek](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            size_t bytes = 0;
            std::string sink_name;
            ss >> sink_name >> bytes;
            seek(sink_name, SeekTarget::from_bytes(bytes));
        };
        commands_["seek.ms"] = [this, seek](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            size_t ms = 0;
            std::string sink_name;
            ss >> sink_name >> ms;
            seek(sink_name, SeekTarget::from_timestamp(av::Timestamp(ms, {1, 1000})));
        };
        commands_["pause"] = [this](ClientStream &cs, std::string &arg) {
            std::shared_ptr<PauseControlTeam> team = InstanceSharedObjects<PauseControlTeam>::get(manager_->instanceData(), arg);
            team->pause();
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
        commands_["speed.set"] = [this](ClientStream &cs, std::string &arg) {
            std::stringstream ss(arg);
            std::string team_name;
            float speed;
            ss >> team_name >> speed;
            std::shared_ptr<SpeedControlTeam> team = InstanceSharedObjects<SpeedControlTeam>::get(manager_->instanceData(), team_name);
            team->setSpeed(speed);
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
    struct Client {
        ControlImpl &control;
        TcpControlServer &server;
        std::list<Client>::iterator iter;
        boost::asio::io_service &io_service;
        tcp::socket socket;
        boost::asio::streambuf buff;
        ClientPipe pipe;
        std::thread thread;
        Client(ControlImpl &_control, TcpControlServer &_server, boost::asio::io_service &_io_service):
            control(_control), server(_server), io_service(_io_service), socket(_io_service),
            pipe([this]() {
                io_service.post([this]() {
                    ControlPacket pkt;
                    if (!pipe.to_client.try_dequeue(pkt)) {
                        logstream << "BUG: nothing in to_client queue but send_to_client was called";
                        return;
                    }
                    if (pkt.type==ControlPacket::Data) {
                        boost::asio::async_write(socket, boost::asio::buffer(pkt.data), [](const boost::system::error_code& error, const size_t) {
                            if (error) {
                                logstream << "send error: " << error;
                            }
                        });
                    } else if (pkt.type==ControlPacket::End) {
                        try {
                            socket.close();
                        } catch (std::exception &e) {
                        }
                        TcpControlServer &s = server;
                        auto &ci = iter;
                        io_service.post([&s, &ci]() {
                            ci->thread.join();
                            s.clients_.erase(ci);
                        });
                    }
                });
            }),
            thread(start_thread("control", [this]() {
                control.communicate(pipe);
            })) {
        };
        void receiveNextLine() {
            boost::asio::async_read_until(socket, buff, '\n', [this](const boost::system::error_code& error, size_t size) {
                if (error) {
                    logstream << "line receive error: " << error;
                    pipe.from_client.emplace(ControlPacket::End);
                    return;
                }
                auto buff_begin = boost::asio::buffers_begin(buff.data());
                std::string line(buff_begin, buff_begin+size);
                buff.consume(size);
                pipe.from_client.emplace(ControlPacket::Data, line);
                // response is handled by posting into the event loop
                receiveNextLine();
            });
        }
    };

    ControlImpl &control_;
    boost::asio::io_service io_service_;
    tcp::acceptor acceptor_;
    std::list<Client> clients_;
    std::thread net_thread_;

    void nextConnection() {
        clients_.emplace_front(control_, *this, io_service_);
        decltype(clients_)::iterator iter = clients_.begin();
        Client &client = *iter;
        client.iter = iter;
        acceptor_.async_accept(client.socket, [this, &client](const boost::system::error_code& error) {
            if (error) {
                logstream << "connection accept error: " << error;
                return;
            }
            client.pipe.from_client.emplace(ControlPacket::Start);
            client.receiveNextLine();
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
        for (Client& client: clients_) {
            try {
                client.socket.close();
            } catch (std::exception &e) {
            }
            client.pipe.from_client.emplace(ControlPacket::End);
        }
        for (Client& client: clients_) {
            client.thread.join();
        }
    }
};


AVPlumber::AVPlumber() {
    if (current_thread.name=="?") {
        set_thread_name("avplumber main");
    }
    av::init();
    av::set_logging_level(AV_LOG_VERBOSE);
    std::shared_ptr<NodeManager> nm = std::make_shared<NodeManager>();
    impl_ = new ControlImpl(nm);
}

AVPlumber::~AVPlumber() {
    delete impl_;
    impl_ = nullptr;
}

#ifdef EMBED_IN_OBS
void AVPlumber::setObsSource(obs_source_t* obssrc) {
    impl_->manager()->instanceData().obs_source_ = obssrc;
}

void AVPlumber::unsetObsSourceAndWait() {
    InstanceData &inst = impl_->manager()->instanceData();
    inst.obs_source_.store(nullptr);
    while (inst.obs_source_used_by_.load()!=0) {
        wallclock.sleepms(30);
    }
}

void AVPlumber::obsTick() {
    impl_->tick();
}
#endif

void AVPlumber::enableControlServer(const uint16_t tcp_port) {
    if (tcp_port) {
        impl_->createServer<TcpControlServer>(*impl_, tcp_port);
    } // if port==0, then NOOP
}

void AVPlumber::executeCommandsFromFile(const std::string path) {
    std::ifstream ifs(path);
    impl_->readExecCommands(ifs, std::cout, true);
}

void AVPlumber::executeCommandsFromString(const std::string script) {
    std::istringstream iss(script);
    impl_->readExecCommands(iss, std::cout, true);
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

void AVPlumber::setReady() {
    logstream << APP_VERSION << " READY." << std::endl;
    impl_->setReady();
}

void AVPlumber::shutdown() {
    impl_->shutdown();
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
