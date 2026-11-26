// Node's stream API cannot receive SCM_RIGHTS. Keep the small Linux-specific
// part here and use the same descriptor validation as the AVP input node.
#include <node_api.h>
#include <cerrno>
#include <sys/un.h>
#include "../../../src/nodes/hwaccel/TexInfoMessage.hpp"

namespace {
napi_value number(napi_env env, int value) {
    napi_value result;
    napi_create_int32(env, value, &result);
    return result;
}

napi_value connectFeed(napi_env env, napi_callback_info call) {
    napi_value arg;
    size_t count = 1, length = 0;
    napi_get_cb_info(env, call, &count, &arg, nullptr, nullptr);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (!count || napi_get_value_string_utf8(env, arg, nullptr, 0, &length) != napi_ok ||
        length == 0 || length >= sizeof(address.sun_path)) return number(env, -1);
    napi_get_value_string_utf8(env, arg, address.sun_path, sizeof(address.sun_path), &length);
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd >= 0 && connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
        close(fd);
        fd = -1;
    }
    return number(env, fd);
}

napi_value receive(napi_env env, napi_callback_info call) {
    napi_value args[2];
    size_t count = 2, size = 0;
    void *header = nullptr;
    int32_t sock;
    napi_get_cb_info(env, call, &count, args, nullptr, nullptr);
    if (count != 2 || napi_get_value_int32(env, args[0], &sock) != napi_ok ||
        napi_get_buffer_info(env, args[1], &header, &size) != napi_ok || size != sizeof(TexInfo))
        return number(env, -2);
    TexInfoMessage message;
    errno = 0;
    const auto result = recvTexInfoMessage(sock, message);
    if (result == TexInfoRecvStatus::Closed && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
        return number(env, -1);
    if (result != TexInfoRecvStatus::Ok) return number(env, -2);
    std::memcpy(header, &message.info, size);
    return number(env, message.plane_fd);
}
}

NAPI_MODULE_INIT() {
    napi_property_descriptor props[] = {
        {"connectFeed", nullptr, connectFeed, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"receive", nullptr, receive, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, 2, props);
    return exports;
}
