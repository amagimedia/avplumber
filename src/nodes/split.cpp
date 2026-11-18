#include "node_common.hpp"
#include <algorithm>
#include <iterator>

template <typename T> class Split: public NodeSingleInput<T>, public NodeMultiOutput<T>, public ReportsFinishByFlag {
protected:
    // Per output, in `dst` order: a frame the output's queue has no room for is dropped for that
    // output instead of waited with, which would hold back the input and every other output.
    std::vector<bool> drop_;
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual bool consumeEofIfPresent() override {
        T* data = this->source_->peek(0);
        if (data == nullptr || !isEofMarker(*data)) {
            return false;
        }
        this->source_->pop();
        onEofConsumed();
        this->finished_ = true;
        return true;
    }
    virtual void onEofConsumed() override {
        T eof = createEofMarker<T>();
        for (auto &edge: this->sink_edges_) {
            edge->enqueue(eof);
        }
    }
    virtual void process() {
        T* data = this->source_->peek();
        if (data==nullptr) {
            return;
        }
        if (isEofMarker(*data)) {
            this->source_->pop();
            onEofConsumed();
            this->finished_ = true;
            return;
        }
        for (size_t i = 0; i < this->sink_edges_.size(); i++) {
            if (!data->isComplete()) {
                logstream << "WARNING: split putting incomplete frame into sink!";
            }
            EdgeSink<T>(this->sink_edges_[i]).put(*data, drop_[i]);
        }
        this->source_->pop();
    }
    // `drop`: true or false for every output, or the names of the outputs (of `dst`) that drop.
    void setDrop(const Parameters &drop, const std::list<std::string> &dst) {
        if (drop.is_boolean()) {
            drop_.assign(dst.size(), drop.get<bool>());
            return;
        }
        drop_.assign(dst.size(), false);
        for (const std::string &name: jsonToStringList(drop)) {
            auto found = std::find(dst.begin(), dst.end(), name);
            if (found == dst.end()) {
                throw Error("split: drop names " + name + ", which is not in dst");
            }
            drop_[std::distance(dst.begin(), found)] = true;
        }
    }
    static std::shared_ptr<Split> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<Split>(make_unique<EdgeSource<T>>(in_edge));
        r->setDrop(params.count("drop")==1 ? params["drop"] : Parameters(false), jsonToStringList(params["dst"]));
        r->createSinksFromParameters(edges, params);
        in_edge->setConsumer(r);
        return r;
    }
};

DECLNODE_ATD(split, Split);
