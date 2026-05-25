#include <napi.h>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include <unordered_map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <inttypes.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>

#include "texinfo.h"

namespace {
static std::unordered_map<std::string, int> g_socks;
static std::unordered_map<std::string, Napi::ThreadSafeFunction> g_loggers;
static std::mutex g_logger_mu;

static int fdpass_log_frames_enabled() {
  // Optional per-frame logging (VERY chatty). Enable with:
  //   FDPASS_LOG_FRAMES=1
  static int s_log_frames = -1;
  if (s_log_frames < 0) {
    const char *v = std::getenv("FDPASS_LOG_FRAMES");
    s_log_frames = (v && *v && std::strcmp(v, "0") != 0) ? 1 : 0;
  }
  return s_log_frames;
}

static void emit_log(const std::string &path, const std::string &line) {
  std::lock_guard<std::mutex> lk(g_logger_mu);
  auto it = g_loggers.find(path);
  if (it == g_loggers.end()) return;
  std::string *heap = new std::string(line);
  it->second.BlockingCall(heap, [](Napi::Env env, Napi::Function cb, std::string *msg) {
    cb.Call({ Napi::String::New(env, *msg) });
    delete msg;
  });
}

static void logf(const std::string &path, const char *fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0) return;
  // stderr
  if (n > 0) {
    std::fwrite(buf, 1, (size_t)n, stderr);
    if (buf[n-1] != '\n') std::fputc('\n', stderr);
    std::fflush(stderr);
  }
  // forward to JS logger
  emit_log(path, std::string(buf, (size_t)n));
}

Napi::Value Close(const Napi::CallbackInfo &info) {
  for (auto it = g_socks.begin(); it != g_socks.end(); ++it) {
    if (it->second >= 0) {
      ::close(it->second);
    }
  }
  g_socks.clear();
  return info.Env().Undefined();
}

class Server {
 public:
  explicit Server(const std::string &path)
    : path_(path), listen_fd_(-1), running_(false), unlink_on_stop_(false) {}

  bool start() {
    if (running_.load()) return true;
    logf(path_, "[fdpass] Server.start path=%s", path_.c_str());
    // Log env state once we know path_ (so it can flow into per-window logs via setServerLogger).
    if (fdpass_log_frames_enabled()) {
      const char *v = std::getenv("FDPASS_LOG_FRAMES");
      logf(path_, "[fdpass] frame logging enabled (FDPASS_LOG_FRAMES=%s) build=%s %s", (v ? v : "<unset>"), __DATE__, __TIME__);
    }
    int ls = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (ls < 0) { logf(path_, "[fdpass] socket() failed errno=%d (%s)", errno, strerror(errno)); return false; }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path_.size() >= sizeof(addr.sun_path)) { logf(path_, "[fdpass] path too long for AF_UNIX (%zu >= %zu)", path_.size(), sizeof(addr.sun_path)); ::close(ls); return false; }
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
    if (::unlink(path_.c_str()) < 0 && errno != ENOENT) { logf(path_, "[fdpass] unlink(%s) failed errno=%d (%s)", path_.c_str(), errno, strerror(errno)); }
    if (::bind(ls, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) { logf(path_, "[fdpass] bind(%s) failed errno=%d (%s)", path_.c_str(), errno, strerror(errno)); ::close(ls); return false; }
    if (::chmod(path_.c_str(), 0777) < 0) { logf(path_, "[fdpass] chmod(%s,0777) failed errno=%d (%s)", path_.c_str(), errno, strerror(errno)); }
    if (::listen(ls, 512) < 0) { logf(path_, "[fdpass] listen(%s) failed errno=%d (%s)", path_.c_str(), errno, strerror(errno)); ::close(ls); ::unlink(path_.c_str()); return false; }
    listen_fd_ = ls;
    // Set non-blocking on the listening socket to avoid blocking accept
    int lflags = ::fcntl(listen_fd_, F_GETFL, 0);
    if (lflags >= 0) ::fcntl(listen_fd_, F_SETFL, lflags | O_NONBLOCK);
    running_.store(true);
    accept_thread_ = std::thread([this]() { this->acceptLoop(); });
    logf(path_, "[fdpass] Server.start OK path=%s fd=%d", path_.c_str(), listen_fd_);
    return true;
  }

  void stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); }
    if (accept_thread_.joinable()) accept_thread_.join();
    {
      std::lock_guard<std::mutex> lk(clients_mu_);
      for (int fd : clients_) { if (fd >= 0) ::close(fd); }
      clients_.clear();
    }
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
    if (unlink_on_stop_ && !path_.empty()) { ::unlink(path_.c_str()); }
  }

  void sendToAll(int send_fd, const std::vector<uint8_t> &meta) {
    if (!running_.load()) return;
    const int s_log_frames = fdpass_log_frames_enabled();
    // Duplicate once with CLOEXEC so we can reuse across clients safely
    int dup_fd = -1;

    dup_fd = ::fcntl(send_fd, F_DUPFD_CLOEXEC, 3);
    if (dup_fd < 0) return;

    char control[CMSG_SPACE(sizeof(int))];
    std::memset(control, 0, sizeof(control));
    struct msghdr msgh; std::memset(&msgh, 0, sizeof(msgh));
    char dummy = 0;
    struct iovec iov;
    if (!meta.empty()) { iov.iov_base = const_cast<uint8_t*>(meta.data()); iov.iov_len = meta.size(); }
    else { iov.iov_base = &dummy; iov.iov_len = 1; }
    msgh.msg_iov = &iov; msgh.msg_iovlen = 1;
    msgh.msg_control = control; msgh.msg_controllen = sizeof(control);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msgh);
    cmsg->cmsg_level = SOL_SOCKET; cmsg->cmsg_type = SCM_RIGHTS; cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    *reinterpret_cast<int *>(CMSG_DATA(cmsg)) = dup_fd;

    std::vector<int> clients;
    {
      std::lock_guard<std::mutex> lk(clients_mu_);
      clients = clients_;
    }

    // Log once per broadcast, not per client.
    if (s_log_frames) {
      if (meta.size() >= sizeof(TexInfo)) {
        TexInfo ti{};
        std::memcpy(&ti, meta.data(), sizeof(TexInfo));
        logf(path_,
             "[fdpass] frame tx path=%s send_fd=%d dup_fd=%d clients=%zu meta=%zu w=%u h=%u stride=%u pixfmt=0x%08x mod=%" PRIu64 " off=%" PRIu64 " ts=%" PRIu64 " frame=%" PRIu64,
             path_.c_str(), send_fd, dup_fd, clients.size(), meta.size(),
             ti.width, ti.height, ti.stride, ti.pixel_format,
             (uint64_t)ti.modifier, (uint64_t)ti.offset, (uint64_t)ti.timestamp, (uint64_t)ti.frame_count);
      } else {
        logf(path_,
             "[fdpass] frame tx path=%s send_fd=%d dup_fd=%d clients=%zu meta=%zu (no texinfo)",
             path_.c_str(), send_fd, dup_fd, clients.size(), meta.size());
      }
    }

    size_t ok_count = 0;
    size_t drop_backpressure = 0;
    size_t drop_error = 0;
    for (int cfd : clients) {
      if (cfd < 0) continue;
      for (;;) {
        ssize_t n = ::sendmsg(cfd, &msgh, MSG_DONTWAIT);
        if (n >= 0) { ok_count++; break; }
        int e = errno;
        if (e == EINTR) continue;
        if (e == EAGAIN || e == EWOULDBLOCK) {
          // skip this client for this frame to maintain pacing
          emit_log(path_, std::string("[fdpass] sendmsg backpressure drop path=") + path_);
          drop_backpressure++;
          break;
        }
        std::fprintf(stderr, "[fdpass] sendmsg to client fd=%d failed errno=%d (%s); dropping client\n", cfd, e, strerror(e)); std::fflush(stderr);
        emit_log(path_, std::string("[fdpass] sendmsg drop client path=") + path_);
        drop_error++;
        removeClient(cfd);
        break;
      }
    }
    if (s_log_frames) {
      logf(path_, "[fdpass] frame tx summary path=%s ok=%zu backpressure=%zu dropped=%zu clients=%zu",
           path_.c_str(), ok_count, drop_backpressure, drop_error, clients.size());
    }
    ::close(dup_fd);
  }

  ~Server() { stop(); }

 private:
  void acceptLoop() {
    for (;;) {
      if (!running_.load()) break;
      int cfd = ::accept(listen_fd_, nullptr, nullptr);
      if (cfd < 0) {
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) { std::fprintf(stderr, "[fdpass] accept error errno=%d (%s)\n", errno, strerror(errno)); std::fflush(stderr); emit_log(path_, std::string("[fdpass] accept error errno=") + std::to_string(errno)); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      int fdflags = ::fcntl(cfd, F_GETFD, 0);
      if (fdflags >= 0) ::fcntl(cfd, F_SETFD, fdflags | FD_CLOEXEC);
      // Set non-blocking on client to avoid blocking sends
      int sflags = ::fcntl(cfd, F_GETFL, 0);
      if (sflags >= 0) ::fcntl(cfd, F_SETFL, sflags | O_NONBLOCK);
      {
        std::lock_guard<std::mutex> lk(clients_mu_);
        clients_.push_back(cfd);
      }
      std::fprintf(stderr, "[fdpass] client accepted fd=%d (path=%s)\n", cfd, path_.c_str()); std::fflush(stderr);
      emit_log(path_, std::string("[fdpass] client accepted path=") + path_);
    }
  }

  void removeClient(int fd) {
    std::lock_guard<std::mutex> lk(clients_mu_);
    for (auto it = clients_.begin(); it != clients_.end(); ++it) {
      if (*it == fd) {
        if (fd >= 0) ::close(fd);
        clients_.erase(it);
        break;
      }
    }
  }

  std::string path_;
  int listen_fd_;
  std::atomic<bool> running_;
  std::thread accept_thread_;
  std::mutex clients_mu_;
  std::vector<int> clients_;
  public: bool unlink_on_stop_;
};

static std::unordered_map<std::string, std::unique_ptr<Server>> g_servers;

Napi::Value CreateServer(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected (socketPath: string)").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string path = info[0].As<Napi::String>().Utf8Value();
  if (g_servers.find(path) != g_servers.end()) return Napi::Boolean::New(env, true);
  auto srv = std::make_unique<Server>(path);
  if (!srv->start()) return Napi::Boolean::New(env, false);
  g_servers.emplace(path, std::move(srv));
  return Napi::Boolean::New(env, true);
}

Napi::Value BroadcastFd(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsString() || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Expected (socketPath: string, fd: number, [meta?: Buffer])").ThrowAsJavaScriptException();
    return env.Null();
  }
  std::string path = info[0].As<Napi::String>().Utf8Value();
  int fd = info[1].As<Napi::Number>().Int32Value();
  std::vector<uint8_t> meta;
  if (info.Length() >= 3 && info[2].IsBuffer()) {
    Napi::Buffer<uint8_t> buf = info[2].As<Napi::Buffer<uint8_t>>();
    meta.assign(buf.Data(), buf.Data() + buf.Length());
  }
  auto it = g_servers.find(path);
  if (it == g_servers.end()) {
    // If the caller is trying to broadcast without a server, make that visible when frame logging is enabled.
    if (fdpass_log_frames_enabled()) {
      logf(path, "[fdpass] WARN: broadcastFd called but no server exists for path=%s fd=%d meta=%zu", path.c_str(), fd, meta.size());
    }
    return env.Undefined();
  }

  it->second->sendToAll(fd, meta);
  return env.Undefined();
}

Napi::Value CloseServer(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) return env.Undefined();
  std::string path = info[0].As<Napi::String>().Utf8Value();
  auto it = g_servers.find(path);
  if (it != g_servers.end()) {
    if (it->second) {
      // Only unlink on explicit CloseServer
      it->second->unlink_on_stop_ = true;
      it->second->stop();
    }
    g_servers.erase(it);
  }
  // release logger if set
  {
    std::lock_guard<std::mutex> lk(g_logger_mu);
    auto lit = g_loggers.find(path);
    if (lit != g_loggers.end()) { lit->second.Release(); g_loggers.erase(lit); }
  }
  return env.Undefined();
}

Napi::Value SetServerLogger(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2 || !info[0].IsString() || !info[1].IsFunction()) return env.Undefined();
  std::string path = info[0].As<Napi::String>().Utf8Value();
  Napi::Function cb = info[1].As<Napi::Function>();
  Napi::ThreadSafeFunction tsf = Napi::ThreadSafeFunction::New(env, cb, "fdpass-logger", 0, 1);
  {
    std::lock_guard<std::mutex> lk(g_logger_mu);
    auto it = g_loggers.find(path);
    if (it != g_loggers.end()) { it->second.Release(); it->second = tsf; }
    else { g_loggers.emplace(path, tsf); }
  }
  return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("createServer", Napi::Function::New(env, CreateServer));
  exports.Set("broadcastFd", Napi::Function::New(env, BroadcastFd));
  exports.Set("broadcastFdWithInfo", Napi::Function::New(env, BroadcastFd));
  exports.Set("closeServer", Napi::Function::New(env, CloseServer));
  exports.Set("setServerLogger", Napi::Function::New(env, SetServerLogger));
  exports.Set("close", Napi::Function::New(env, Close));
  return exports;
}

} // namespace

NODE_API_MODULE(fdpass, Init)
