#include "../node_common.hpp"
#include <fstream>

template <typename T> class JitterGenerator: public NodeSISO<T, T> {
protected:
    std::ifstream random_{"/dev/urandom", std::ifstream::in | std::ifstream::binary};
    int random() {
        return random_.get() | (random_.get() << 8);
    }
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        wallclock.sleepms(random() % 100 + (((random() & 7) == 0) ? random()%600 : 0));
        int pressure = 0;
        while (this->source_->peek(0)) {
            T data = this->source_->get();
            this->sink_->put(data);
            wallclock.sleepms(random() % std::max(1, 120-pressure*3));
            pressure++;
        }
    }
    static std::shared_ptr<JitterGenerator> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<JitterGenerator> r = NodeSISO<T, T>::template createCommon<JitterGenerator>(edges, params);
        return r;
    }
};

DECLNODE_ATD(jittergen, JitterGenerator);
