#include "graph_mgmt.hpp"
#include <cassert>

int main() {
    const auto manager = std::make_shared<NodeManager>();
    Parameters source = {
        {"type", "fabric_source"}, {"name", "rx"}, {"provider", "tcp"},
        {"bind_host", "127.0.0.1"}, {"bind_port", 0}, {"dst", "fill"}, {"alpha_dst", "key"},
        {"real_pixel_format", "yuv420p"}, {"alpha_codec", "hevc_intra"},
        {"alpha_pixel_format", "cuda"}, {"alpha_real_pixel_format", "yuv444p10le"},
        {"max_payload_bytes", 65536}
    };
    manager->createNode(source, true, false);
    auto metadata = [&](const char *edge) {
        return manager->edges()->find<FabricPacket>(edge)->metadata<InputStreamMetadata>()->source_stream;
    };
    const auto fill = metadata("fill"), key = metadata("key");
    assert(fill.raw()->codecpar->codec_id == AV_CODEC_ID_H264);
    assert(fill.raw()->codecpar->format == AV_PIX_FMT_YUV420P);
    assert(key.raw()->codecpar->codec_id == AV_CODEC_ID_HEVC);
    assert(key.raw()->codecpar->format == AV_PIX_FMT_YUV444P10LE);

    Parameters unpack = {{"type", "fabric_unpack"}, {"name", "unpack"}, {"src", "key"}, {"dst", "packets"}};
    manager->createNode(unpack, true, false);
    Parameters sink = {
        {"type", "fabric_sink"}, {"name", "wrong_codec"}, {"src", "packets"}, {"codec", "nvjpeg"},
        {"remote_host", "127.0.0.1"}, {"remote_port", 1}, {"stream_id", "key"}, {"replica_id", 1}
    };
    bool rejected = false;
    try {
        manager->createNode(sink, true, false);
    } catch (const std::exception &err) {
        rejected = std::string(err.what()).find("disagrees with upstream hevc") != std::string::npos;
    }
    assert(rejected);
    manager->shutdown();
}
