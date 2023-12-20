#include "node_common.hpp"
#include "../output_control.hpp"
#include <fstream>
#include <libavutil/imgutils.h>

namespace {
    
    using BytesBuffer = std::vector<uint8_t>;
    
    struct DataGetterBase {
    protected:
        BytesBuffer buffer_;
        void resizeBuffer(const size_t req_size) {
            buffer_.resize(req_size, 0);
        }
    };
    
    template<typename> struct DataGetter {
    };
    template<> struct DataGetter<av::Packet>: protected DataGetterBase {
        BytesBuffer& get(av::Packet pkt, av::Timestamp) {
            resizeBuffer(pkt.size());
            std::memcpy(&buffer_[0], pkt.data(), pkt.size());
            return buffer_;
        }
        static constexpr bool temporally_cuttable = false;
    };
    template<typename T> struct FrameDataGetter: protected DataGetterBase {
    protected:
        BytesBuffer& getFromPlanes(T pkt, const size_t planes_count, const size_t skip_in_each_plane = 0) {
            size_t whole_size = 0;
            for (size_t i=0; i<planes_count; i++) {
                assert(pkt.size(i) >= skip_in_each_plane);
                whole_size += pkt.size(i) - skip_in_each_plane;
                //logstream << "plane " << i << " size " << pkt.size(i);
            }
            resizeBuffer(whole_size);
            uint8_t* dstp = &buffer_[0];
            for (size_t i=0; i<planes_count; i++) {
                size_t tocopy = pkt.size(i) - skip_in_each_plane;
                std::memcpy(dstp, pkt.data(i) + skip_in_each_plane, tocopy);
                dstp += tocopy;
            }
            return buffer_;
        }
    };
    template<> struct DataGetter<av::VideoFrame>: FrameDataGetter<av::VideoFrame> {
        BytesBuffer& get(av::VideoFrame frm, av::Timestamp) {
            //return getFromPlanes(pkt, pkt.pixelFormat().planesCount());
            int size = av_image_get_buffer_size(frm.pixelFormat().get(), frm.width(), frm.height(), 1);
            if (size<1) {
                throw Error("av_image_get_buffer_size returned " + std::to_string(size));
            }
            resizeBuffer(size);
            AVFrame *frame = frm.raw();
            int ret = av_image_copy_to_buffer(&buffer_[0], size,
                                              (const uint8_t **)frame->data, frame->linesize,
                                              (AVPixelFormat)frame->format,
                                              frame->width, frame->height, 1);
            if (ret<0) {
                throw Error("av_image_copy_to_buffer returned " + std::to_string(size));
            }
            return buffer_;
        }
        static constexpr bool temporally_cuttable = false;
    };
    template<> struct DataGetter<av::AudioSamples>: FrameDataGetter<av::AudioSamples> {
        BytesBuffer empty_buffer_;
        BytesBuffer& get(av::AudioSamples pkt, av::Timestamp from_pts) {
            int skip = 0;
            if ( (!from_pts.isNoPts()) && (from_pts>pkt.pts()) ) {
                skip = addTS(from_pts, negateTS(pkt.pts())).timestamp(av::Rational(1, pkt.sampleRate()));
                if (skip >= pkt.samplesCount()) {
                    return empty_buffer_;
                }
                skip *= pkt.sampleFormat().bytesPerSample();
                skip *= pkt.isPlanar() ? 1 : pkt.channelsCount();
            }
            return getFromPlanes(pkt, pkt.isPlanar() ? pkt.channelsCount() : 1, skip);
        }
        static constexpr bool temporally_cuttable = true;
    };
};

template <typename T> class RawOutput: public NodeSingleInput<T> {
protected:
    bool ready_ = false;
    AVTS offset_;
    DataGetter<T> data_getter_;
    std::string output_path_;
    std::ofstream ost_;
    bool output_enabled_;
    std::shared_ptr<OutputControl> common_;
    void setOutputPath(const std::string fname) {
        output_path_ = fname;
    }
    void setOutputGroup(const std::string name) {
        common_ = OutputControl::get(name);
        common_->registerNode(data_getter_.temporally_cuttable);
    }
    void openCloseOutput() {
        if (ost_.is_open() && !output_enabled_) {
            ost_.close();
        } else if ((!ost_.is_open()) && output_enabled_) {
            ost_.open(output_path_, std::ios::out | std::ios::binary);
            ost_.clear();
        }
    }
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual void process() {
        T* pkt = this->source_->peek();
        if (pkt==nullptr) {
            return;
        }
        
        OutputControl::State state = common_->state();
        if (state == OutputControl::Waiting1) {
            if (data_getter_.temporally_cuttable) { // only audio
                state = common_->addTemporallyCuttablePTS(pkt->pts(), this);
            }
        }
        if (state == OutputControl::Waiting2) {
            if (!data_getter_.temporally_cuttable) { // only video
                if (common_->minPts().isNoPts() || (pkt->pts() >= common_->minPts())) {
                    state = common_->setStartPts(pkt->pts());
                } else {
                    this->source_->pop(); // skip this packet
                    return;
                }
            }
        }
        
        output_enabled_ = state == OutputControl::Started;
        if (ost_.is_open() != output_enabled_) {
            openCloseOutput();
        }
        if (ost_.is_open()) {
            assert(state == OutputControl::Started);
            BytesBuffer &buffer = data_getter_.get(*pkt, common_->startPts());
            ost_.write(reinterpret_cast<char*>(&buffer[0]), buffer.size());
            ost_.flush();
            if (ost_.bad()) {
                logstream << "write failed, closing pipe and stopping";
                ost_.close();
                common_->stop();
            }
        }
        
        if (state==OutputControl::Started || state==OutputControl::Stopped /*|| (state==OutputControl::Waiting2 && !data_getter_.temporally_cuttable)*/) {
            this->source_->pop();
        } else {
            usleep(50*1000);
        }
    }
    virtual void stop() {
        NodeSingleInput<T>::stop();
        common_->deregisterNode(data_getter_.temporally_cuttable);
    }
    static std::shared_ptr<RawOutput> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<Edge<T>> edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<RawOutput<T>>(make_unique<EdgeSource<T>>(edge));
        r->setOutputPath(params["path"]);
        if (params.count("output_group")==1) {
            r->setOutputGroup(params["output_group"]);
        } else {
            r->setOutputGroup("default");
        }
        return r;
    }
};

DECLNODE_ATD(raw_output, RawOutput);
