"""Execute the close-drain helper with a deterministic socket and clock."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_live_srt_close_drain_contract(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    patch = (Path(__file__).resolve().parents[1] /
             "deps/ffmpeg/9/0019-avformat-libsrt-live-close-drain.patch").read_text()
    additions = "\n".join(line[1:] for line in patch.splitlines()
                          if line.startswith("+") and not line.startswith("+++"))
    helper = additions[additions.index("static int libsrt_drain_live_output"):]
    helper = helper[:helper.index("\n}\n") + 3]
    source = r'''
#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>
#include <assert.h>
#include <errno.h>
#define AVIO_FLAG_WRITE 2
#define SRTT_FILE 1
#define AVERROR(e) (-(e))
#define AVERROR_EXIT (-123)
#define AV_LOG_VERBOSE 0
#define AV_LOG_ERROR 1
#define FFMAX(a,b) ((a) > (b) ? (a) : (b))
#define av_log(...) ((void)0)
typedef struct { int linger, fd, transtype; } SRTContext;
typedef struct { void *priv_data; int flags, interrupt_callback; } URLContext;
typedef struct { uint64_t byteSentTotal; int msSndTsbPdDelay; double msRTT; } SRT_TRACEBSTATS;
static int64_t now, pending_until;
static int interrupted, socket_error, calls, delay_ms = 200;
static int64_t av_gettime_relative(void) { return now; }
static int av_usleep(unsigned us) { now += us; return 0; }
static int ff_check_interrupt(void *cb) { return interrupted; }
static int libsrt_neterrno(URLContext *h) { return -EIO; }
static int srt_getsndbuffer(int fd, size_t *blocks, size_t *bytes) {
    ++calls;
    if (socket_error) return -1;
    *blocks = now < pending_until ? 1 : 0;
    *bytes = *blocks * 1316;
    return 0;
}
static int srt_bstats(int fd, SRT_TRACEBSTATS *s, int clear) {
    s->byteSentTotal = 1316;
    s->msSndTsbPdDelay = delay_ms;
    s->msRTT = 0.5;
    return 0;
}
'''
    source += helper
    source += r'''
int main(void) {
    SRTContext s = {.linger=2, .fd=5, .transtype=0};
    URLContext h = {.priv_data=&s, .flags=AVIO_FLAG_WRITE};
    pending_until = 5000;
    assert(libsrt_drain_live_output(&h) == 0);
    /* Last ACK at 5ms is not enough: retain the connection for the peer's
       200ms TSBPD delay + two 0.5ms RTTs + rounding guard afterwards. */
    assert(now >= 206000 && now < 209000);

    now = calls = 0; s.linger = 0;
    assert(libsrt_drain_live_output(&h) == 0 && calls == 0 && now == 0);
    s.linger = -1;
    assert(libsrt_drain_live_output(&h) == 0 && calls == 0);
    s.linger = 2; h.flags = 0;
    assert(libsrt_drain_live_output(&h) == 0 && calls == 0);
    h.flags = AVIO_FLAG_WRITE; s.transtype = SRTT_FILE;
    assert(libsrt_drain_live_output(&h) == 0 && calls == 0);

    s.transtype = 0; pending_until = 3000000;
    assert(libsrt_drain_live_output(&h) == -ETIMEDOUT && now == 2000000);
    now = 0; pending_until = 0; delay_ms = 3000;
    assert(libsrt_drain_live_output(&h) == -ETIMEDOUT && now == 2000000);
    now = 0; delay_ms = 200; interrupted = 1;
    assert(libsrt_drain_live_output(&h) == AVERROR_EXIT && now == 0);
    interrupted = 0; socket_error = 1;
    assert(libsrt_drain_live_output(&h) == -EIO && now == 0);
    return 0;
}
'''
    path = tmp_path / "drain.c"
    path.write_text(source)
    binary = tmp_path / "drain"
    subprocess.run([cc, "-std=c11", str(path), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, timeout=5)
