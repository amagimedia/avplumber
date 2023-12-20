#include <json.hpp>
#include <memory>
#include <thread>
#include "graph_mgmt.hpp"

class StatsSenderThread: public std::enable_shared_from_this<StatsSenderThread> {
protected:
    std::thread thr_;
public:
    StatsSenderThread(nlohmann::json params, std::shared_ptr<NodeManager> manager);
};
