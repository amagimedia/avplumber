#!/usr/bin/env python3
"""GStreamer CUDA mixer qualification harness.

The validated configuration uses 105 HEVC inputs, 16 original resident NV12 clips,
and eight real Chromium DMA-BUF pages (seven pages plus one timestamp key).
This is a static benchmark graph, not a complete production switcher.
See README.md for dependencies, workload preparation, flags and measurement limits.
"""

import argparse, json, sys, time, threading
from collections import Counter
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

W, H = 1920, 1080
P = argparse.ArgumentParser()
P.add_argument("--show", default="mixer.demo.json", help="demo show JSON; source paths must resolve in this process")
P.add_argument("--sources", default="sources.json", help="HEVC elementary-stream manifest")
P.add_argument("--raw-dir", default="raw", help="legacy sliced raw files; not used for pinned uploads")
P.add_argument("--socket-dir", default="/tmp/dma-page", help="live Chromium DMA-BUF sockets")
P.add_argument("--video", type=int, default=105)
P.add_argument("--raw", type=int, default=47)
P.add_argument("--browsers", type=int, default=40)
P.add_argument("--aux", type=int, default=20)
P.add_argument("--thumbs", type=int, default=3)
P.add_argument("--seconds", type=int, default=60)
P.add_argument("--warmup", type=int, default=30)
P.add_argument("--latency-ms", type=int, default=66, help="live compositor latency (how long it waits for inputs)")
P.add_argument("--browser-pattern", default="blink", help="gltestsrc pattern for the browser stand-ins (blink: a trivial full-frame clear, so the measured cost is the per-frame GL->CUDA import, as for Chromium DMA-BUF frames, not shader rendering)")
P.add_argument("--browser-source", choices=["gl", "dmabuf"], default="gl",
               help="dmabuf: the demo's own Chromium windows (dma-browser), frames received over their sockets "
                    "in /tmp/dma-page as DMA-BUF -> glupload -> cudaupload; gl: OpenGL stand-ins")
P.add_argument("--dmabuf-cache", action="store_true",
               help="reuse one GstMemory per Chromium allocation (by DMA-BUF inode), so glupload's EGLImage and "
                    "cudaupload's CUDA registration are cached, as avplumber's importer caches by allocation")
P.add_argument("--split-gl", action="store_true",
               help="dmabuf: one small pipeline per page (appsrc ! glupload ! cudaupload ! proxysink), each with its own "
                    "OpenGL context thread, joined to the main graph by proxysrc; all share the main CUDA context and clock")
P.add_argument("--raw-source", choices=["file", "pinned"], default="file",
               help="file: multifilesrc reads one frame file per buffer into ordinary memory, cudaupload copies it "
                    "through the driver's staging buffer; pinned: as avplumber's 'resident' raw upload, each clip's "
                    "frames are read once into page-locked memory (cuMemHostAlloc in GStreamer's own CUDA context) and "
                    "appsrc loops buffers wrapping that memory, so cudaupload's copy is a direct DMA")
P.add_argument("--dsk", default="", help="comma list of DSK pages to add (e.g. bug_right), placed in their show boxes, "
               "on top of --browsers")
P.add_argument("--rtp-port", type=int, default=0, help="send the program SDR encode as RTP to 127.0.0.1:PORT "
               "(latency/rtp_latency.py reads a tcpdump of it) instead of a fakesink")
P.add_argument("--browser-skip", type=int, default=0, help="start at this browser page of the show")
P.add_argument("--program-scene", default="random_200")
P.add_argument("--preview-scene", default="random_201")
P.add_argument("--dump", action="store_true")
P.add_argument("--light", action="store_true", help="instrument only the outputs and every 8th source; no drop counting "
               "(Python probes run in the streaming threads and cost real time)")
A = P.parse_args()

show = json.load(open(A.show))
es = {s["id"]: s["es"] for s in json.load(open(A.sources))["sources"]}
want = {"video": A.video, "nv12": A.raw, "browser": A.browsers}
taken = {k: 0 for k in want}
srcs = []                                   # (id, kind) in show order, limited per kind
skip = {"browser": A.browser_skip}           # use a different set of pages per run (a run that did not drain
for s in show["sources"]:                    # leaves its pages waiting for ACKs until the mixer reopens them)
    k = s.get("kind")
    if k == "browser" and s["id"] in [f"dsk_{d}" for d in A.dsk.split(",") if d]:
        srcs.append((s["id"], k, s)); continue
    if k == "browser" and s["id"].startswith("dsk_"):
        continue
    if skip.get(k, 0) > 0:
        skip[k] -= 1; continue
    if k in want and taken[k] < want[k]:
        taken[k] += 1; srcs.append((s["id"], k, s))
idx = {sid: i for i, (sid, _, _) in enumerate(srcs)}
source_counts = dict(Counter(kind for _, kind, _ in srcs))
raw_indices = [i for i, (_, kind, _) in enumerate(srcs) if kind == "nv12"]
scenes = {s["id"]: s for s in show["scenes"]}
DSK = ["dsk_lower_third", "dsk_ticker", "dsk_bug_left", "dsk_bug_right"]

def src_index(source_id, k):
    i = idx.get(source_id)
    return i if i is not None else (k * 7 + len(source_id)) % len(srcs)

desc, OUT, branches = [], [], {}
Q = "queue leaky=downstream max-size-buffers=2 max-size-time=0 max-size-bytes=0"

def link_src(i, comp, pad):
    branches[i] = branches.get(i, 0) + 1
    desc.append(f"t{i}. ! {Q} ! {comp}.{pad}")

def encoder(preset, kbps, fps=30):
    return (f"nvh264enc preset={preset} tune=ultra-low-latency rc-mode=cbr bitrate={kbps} max-bitrate={kbps} "
            f"bframes=0 zerolatency=true gop-size={fps} aud=true spatial-aq=false temporal-aq=false ! "
            f"video/x-h264,profile={'baseline' if preset == 'p5' else 'high'} ! h264parse")

def compositor(name, cells, fps=30, w=W, h=H):
    props = " ".join(f"sink_{k}::xpos={x} sink_{k}::ypos={y} sink_{k}::width={cw} sink_{k}::height={ch}"
                     for k, (_, x, y, cw, ch) in enumerate(cells))
    desc.append(f"cudacompositor name={name} force-live=true latency={A.latency_ms * 1000000} ignore-inactive-pads=true "
                f"{props} ! video/x-raw(memory:CUDAMemory),format=NV12,width={w},height={h},framerate={fps}/1 ! "
                f"identity name={name}_out silent=true ! tee name={name}_t")
    for k, (spec, x, y, cw, ch) in enumerate(cells):
        if spec[0] == "src":
            link_src(spec[1], name, f"sink_{k}")
        else:
            desc.append(f"{spec[1]}. ! {Q} ! cudascale ! video/x-raw(memory:CUDAMemory),width={cw},height={ch} ! {name}.sink_{k}")

def encode_from(tee, name, cls, preset, kbps, fps=30):
    OUT.append((name, cls))
    sink = (f"rtph264pay config-interval=-1 pt=96 ! udpsink name=sink_{name} host=127.0.0.1 port={A.rtp_port} sync=false async=false"
            if A.rtp_port and name == "pgm_sdr" else f"fakesink name=sink_{name} sync=false qos=false async=false")
    desc.append(f"{tee}. ! {Q} ! {encoder(preset, kbps, fps)} ! {sink}")

def scene_cells(scene_id, extra=()):
    cells = [(("src", src_index(it["source"], k)), it["dst"]["x"], it["dst"]["y"], it["dst"]["w"], it["dst"]["h"])
             for k, it in enumerate(scenes[scene_id]["items"])]
    boxes = {k["source"]: k["dst"] for k in show.get("dsk", {}).get("keys", [])}
    full = {"x": 0, "y": 0, "w": W, "h": H}
    return cells + [(("src", idx[d]), boxes.get(d, full)["x"], boxes.get(d, full)["y"], boxes.get(d, full)["w"],
                     boxes.get(d, full)["h"]) for d in extra if d in idx]

for i, (sid, kind, s) in enumerate(srcs):
    if kind == "video":
        desc.append(f"multifilesrc location={es[sid]} loop=true caps=\"video/x-h265,stream-format=byte-stream,alignment=au,"
                    f"width={W},height={H},framerate=30/1\" ! h265parse ! nvh265dec ! "
                    f"video/x-raw(memory:CUDAMemory),format=NV12 ! clocksync sync-to-first=true ! tee name=t{i} allow-not-linked=true")
    elif kind == "nv12" and A.raw_source == "pinned":
        desc.append(f"appsrc name=rw{i} is-live=true do-timestamp=true format=time block=true max-buffers=2 "
                    f"caps=\"video/x-raw,format=NV12,width={W},height={H},framerate=30/1\" ! cudaupload ! "
                    f"video/x-raw(memory:CUDAMemory),format=NV12 ! tee name=t{i} allow-not-linked=true")
    elif kind == "nv12":
        desc.append(f"multifilesrc location={A.raw_dir}/{sid}/%02d.nv12 loop=true stop-index=29 ! "
                    f"rawvideoparse format=nv12 width={W} height={H} "
                    f"framerate=30/1 ! clocksync sync-to-first=true ! cudaupload ! video/x-raw(memory:CUDAMemory),format=NV12 ! "
                    f"tee name=t{i} allow-not-linked=true")
    elif A.browser_source == "dmabuf" and A.split_gl:
        desc.append(f"proxysrc name=px{i} ! video/x-raw(memory:CUDAMemory) ! tee name=t{i} allow-not-linked=true")
    elif A.browser_source == "dmabuf":
        desc.append(f"appsrc name=br{i} is-live=true do-timestamp=true format=time max-buffers=4 block=false ! "
                    f"glupload ! video/x-raw(memory:GLMemory) ! cudaupload name=cu{i} ! "
                    f"video/x-raw(memory:CUDAMemory) ! tee name=t{i} allow-not-linked=true")
    else:
        desc.append(f"gltestsrc is-live=true pattern={A.browser_pattern} ! "
                    f"video/x-raw(memory:GLMemory),format=RGBA,width={W},height={H},framerate=30/1 ! cudaupload ! "
                    f"video/x-raw(memory:CUDAMemory),format=RGBA ! tee name=t{i} allow-not-linked=true")

compositor("pgm", scene_cells(A.program_scene, DSK))
encode_from("pgm_t", "pgm_sdr", "program", "p3", 6000)
encode_from("pgm_t", "pgm_webrtc", "program", "p1", 4000)
compositor("clean", scene_cells(A.program_scene))
encode_from("clean_t", "sdr_clean", "program", "p3", 6000)
compositor("pvw", scene_cells(A.preview_scene))
encode_from("pvw_t", "pvw_webrtc", "preview", "p1", 4000)
# pgm_pvw_grid: PVW and PGM over eight slots, each slot a scene of the demo's mv bus composed at 480x270
mv_bus = next(b for b in show["aux_buses"] if b["id"] == "mv")
slot_scenes = (mv_bus.get("scenes") or [])[:8]
mv = [(("tee", "pvw_t"), 0, 0, 960, 540), (("tee", "pgm_t"), 960, 0, 960, 540)]
for k, sc in enumerate(slot_scenes):
    if not sc:
        continue                                 # an empty slot, as on the demo's mv bus
    cells = [(spec, x // 4, y // 4, max(2, cw // 4), max(2, ch // 4)) for spec, x, y, cw, ch in scene_cells(sc)]
    compositor(f"slot{k}", cells, w=480, h=270)
    mv.append((("tee", f"slot{k}_t"), (k % 4) * 480, 540 + (k // 4) * 270, 480, 270))
compositor("mv", mv)
encode_from("mv_t", "mv", "aux", "p1", 4000)
for a in range(A.aux):
    cells = [(("src", k % len(srcs)), (k % 4) * 480, (k // 4) * 360 + 45, 480, 270) for k in range(12)]
    compositor(f"aux{a}", cells)
    encode_from(f"aux{a}_t", f"aux{a}", "aux", "p1", 4000)
for b in range(A.thumbs):
    ids = list(range(b * 64, min((b + 1) * 64, len(srcs))))
    if ids:
        compositor(f"thumb{b}", [(("src", i), (j % 8) * 240, (j // 8) * 135, 240, 135) for j, i in enumerate(ids)], fps=15)
        encode_from(f"thumb{b}_t", f"thumb{b}", "thumbs", "p5", 3500, fps=15)

text = "\n".join(desc)
if A.dump:
    print(text); sys.exit(0)

Gst.init(None)
t0 = time.time()
pipe = Gst.parse_launch(text)
lock = threading.Lock()
counts = {}
lat = {"composite": [], "output": []}
intervals, last = [], {}
program_intervals = []
measuring = [False]
clock_ref = {}

def due(pad, buf):
    """The frame's due clock time: base time plus its running time (encoders shift PTS, the segment undoes it)."""
    ev = pad.get_sticky_event(Gst.EventType.SEGMENT, 0)
    rt = ev.parse_segment().to_running_time(Gst.Format.TIME, buf.pts) if ev else buf.pts
    return clock_ref["base"] + rt

def counter(name):
    counts[name] = 0
    def cb(pad, info):
        with lock:
            counts[name] += 1
        return Gst.PadProbeReturn.OK
    return cb

def timing(name, kind):
    def cb(pad, info):
        if measuring[0]:
            buf = info.get_buffer(); now = clock_ref["clock"].get_time()
            with lock:
                lat[kind].append((now - due(pad, buf)) / 1e6)
                if kind == "output":
                    if name in last:
                        intervals.append((now - last[name]) / 1e6)
                        if name == "pgm_sdr": program_intervals.append((now - last[name]) / 1e6)
                    last[name] = now
        return Gst.PadProbeReturn.OK
    return cb

for name, _ in OUT:
    pad = pipe.get_by_name(f"sink_{name}").get_static_pad("sink")
    pad.add_probe(Gst.PadProbeType.BUFFER, counter(name)); pad.add_probe(Gst.PadProbeType.BUFFER, timing(name, "output"))
comps = [n for n in ["pgm", "clean", "pvw", "mv"] + [f"slot{k}" for k in range(8)] + [f"aux{a}" for a in range(A.aux)] + [f"thumb{b}" for b in range(A.thumbs)]
         if pipe.get_by_name(f"{n}_out")]
for n in comps:
    pipe.get_by_name(f"{n}_out").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, timing(n, "composite"))
probed = [i for i in range(len(srcs)) if not A.light or i % 8 == 0 or srcs[i][1] != srcs[i - 1][1] or srcs[i][1] == "browser"]
for i in probed:
    pipe.get_by_name(f"t{i}").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, counter(f"src{i}"))
drops = {"compositor_in": 0, "encoder_in": 0}
def on_overrun(q, kind):
    with lock:
        drops[kind] += 1
it = pipe.iterate_recurse()
while not A.light:
    ok, el = it.next()
    if ok != Gst.IteratorResult.OK:
        break
    if el.get_factory() and el.get_factory().get_name() == "queue":
        peer = el.get_static_pad("src").get_peer()
        kind = "encoder_in" if peer and peer.get_parent_element().get_factory().get_name() == "nvh264enc" else "compositor_in"
        el.connect("overrun", on_overrun, kind)

# --- dma-browser frames (deps/dma-browser/addons/fdpass/texinfo.h, dmabuf_ack.h) -------------------
import os, socket, struct, selectors
HDR = struct.Struct("<IIIIQQQQ")            # width height stride fourcc modifier offset timestamp frame_count
ACK = struct.Struct("<IIQ")                 # 'ACK1', kind 0 = frame, frame_count
ACK_MAGIC = ord("A") | ord("C") << 8 | ord("K") << 16 | ord("1") << 24
DRM = {ord("B") << 24 | ord("G") << 16 | ord("R") << 8 | ord("A"): "AR24",      # 'BGRA' = DRM ARGB8888
       ord("R") << 24 | ord("G") << 16 | ord("B") << 8 | ord("A"): "AB24"}      # 'RGBA' = DRM ABGR8888
br_stats = {"frames": 0, "acks": 0, "bad": 0}
br_states = {}

def start_dmabuf_receivers():
    gi.require_version("GstAllocators", "1.0"); gi.require_version("GstVideo", "1.0")
    from gi.repository import GstAllocators, GstVideo
    alloc = GstAllocators.DmaBufAllocator.new()
    sel = selectors.DefaultSelector()
    state = {}
    for i, (sid, kind, _) in enumerate(srcs):
        if kind != "browser":
            continue
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(f"{A.socket_dir}/{sid}.sock")
        owner = subs.get(i, pipe)
        st = {"i": i, "sock": sock, "src": owner.get_by_name(f"br{i}"), "caps": None, "pending": [], "lock": threading.Lock()}
        state[i] = st; br_states[i] = st
        sel.register(sock, selectors.EVENT_READ, st)

        def acked(pad, info, st=st):
            n = info.get_buffer().offset
            with st["lock"]:
                done = [f for f in st["pending"] if f <= n]          # frames dropped upstream are returned too
                st["pending"] = [f for f in st["pending"] if f > n]
            for f in done:
                st["sock"].sendall(ACK.pack(ACK_MAGIC, 0, f))
            with lock:
                br_stats["acks"] += len(done)
            return Gst.PadProbeReturn.OK
        owner.get_by_name(f"cu{i}").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, acked)

    def run():
        while True:
            for key, _ in sel.select(timeout=1):
                st = key.data
                data, anc, flags, _ = st["sock"].recvmsg(HDR.size, socket.CMSG_SPACE(4))
                fds = [fd for lvl, typ, raw in anc if lvl == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS
                       for fd in struct.unpack(f"{len(raw) // 4}i", raw[:len(raw) // 4 * 4])]
                if len(data) < HDR.size or len(fds) != 1:
                    for fd in fds:
                        os.close(fd)
                    with lock:
                        br_stats["bad"] += 1
                    if not data:
                        sel.unregister(st["sock"])
                    continue
                w, h, stride, fourcc, modifier, offset, _ts, frame = HDR.unpack(data)
                fd = fds[0]
                if st["caps"] is None:
                    fmt = DRM[fourcc] + (f":0x{modifier:016x}" if modifier else "")
                    st["caps"] = Gst.Caps.from_string(f"video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format={fmt},"
                                                      f"width={w},height={h},framerate=30/1")
                    st["src"].set_property("caps", st["caps"])
                buf = Gst.Buffer.new()
                if A.dmabuf_cache:
                    ino = os.fstat(fd).st_ino
                    mem = st.setdefault("mems", {}).get(ino)
                    if mem is None:
                        mem = GstAllocators.DmaBufAllocator.alloc(alloc, fd, os.lseek(fd, 0, os.SEEK_END))
                        st["mems"][ino] = mem
                        with lock:
                            br_stats["imports"] = br_stats.get("imports", 0) + 1
                    else:
                        os.close(fd)                                       # same allocation: keep the first fd
                    buf.append_memory(mem)
                else:
                    buf.append_memory(GstAllocators.DmaBufAllocator.alloc(alloc, fd, os.lseek(fd, 0, os.SEEK_END)))  # owns the fd
                GstVideo.buffer_add_video_meta_full(buf, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.DMA_DRM,
                                                    w, h, 1, [offset, 0, 0, 0], [stride, 0, 0, 0])
                buf.offset = frame
                with st["lock"]:
                    st["pending"].append(frame)
                st["src"].emit("push-buffer", buf)
                with lock:
                    br_stats["frames"] += 1
    threading.Thread(target=run, daemon=True, name="dmabuf-recv").start()

loop = GLib.MainLoop()
errors = []
def on_msg(bus, msg):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error(); errors.append(f"{err.message} | {(dbg or '')[:300]}"); loop.quit()
bus = pipe.get_bus(); bus.add_signal_watch(); bus.connect("message", on_msg)

cls = {}
for name, c in OUT:
    cls.setdefault(c, []).append(name)
expect = {"program": 30, "preview": 30, "aux": 30, "thumbs": 15}
kinds = {}
hist_probed = None
hist, dropshist = [], []
raw_feed_counts = {i: 0 for i in raw_indices}
raw_feed_hist = []
raw_manifest = []

def pct(v, p):
    v = sorted(v); return round(v[min(len(v) - 1, int(p / 100 * len(v)))], 1) if v else None

def tick():
    now = time.time()
    if not measuring[0] and now - t0 >= A.warmup:
        measuring[0] = True
    with lock:
        snap, d = dict(counts), dict(drops)
    hist.append((now, snap)); dropshist.append((now, d))
    raw_feed_hist.append((now, dict(raw_feed_counts)))
    if len(hist) >= 2:
        (ta, a), (tb, b) = hist[-2], hist[-1]
        r = lambda n: (b[n] - a[n]) / (tb - ta)
        line = "  ".join(f"{c}: min {min(r(n) for n in names):5.1f}" for c, names in cls.items())
        line += "  " + "  ".join(f"{k}: min {min(r(f'src{i}') for i in ix):5.1f}" for k, ix in kinds.items())
        print(f"{now - t0:6.0f}s {line}  drops {d}" + (f"  dmabuf {br_stats}" if A.browser_source == "dmabuf" else ""), flush=True)
    if now - t0 >= A.warmup + A.seconds:
        loop.quit(); return False
    return True

for i in probed:
    kinds.setdefault(srcs[i][1], []).append(i)
n_in = sum(branches.values())
print(f"outputs {len(OUT)} { {c: len(n) for c, n in cls.items()} }, sources {source_counts}, "
      f"compositor inputs {n_in}", flush=True)
subs = {}
shared = {}                                     # the main pipeline's CUDA context, handed to every page pipeline
def sync_handler(bus, msg, who):
    if msg.type == Gst.MessageType.HAVE_CONTEXT:
        ctx = msg.parse_have_context()
        shared.setdefault(ctx.get_context_type(), ctx)
    elif msg.type == Gst.MessageType.NEED_CONTEXT and who != "main":
        ok, ctype = msg.parse_context_type()
        # Only CUDA is shared: a shared GstGLDisplay hands every page the same GL context (one thread for all).
        if ctype in shared and ctype == "gst.cuda.context":
            msg.src.set_context(shared[ctype])
    return Gst.BusSyncReply.PASS
pipe.get_bus().set_sync_handler(sync_handler, "main")
if A.browser_source == "dmabuf" and A.split_gl:
    for i, (sid, kind, _) in enumerate(srcs):
        if kind == "browser":
            sub = Gst.parse_launch(f"appsrc name=br{i} is-live=true do-timestamp=true format=time max-buffers=4 block=false ! "
                                   f"glupload ! video/x-raw(memory:GLMemory) ! cudaupload name=cu{i} ! "
                                   f"video/x-raw(memory:CUDAMemory) ! proxysink name=pxs{i}")
            sub.get_bus().set_sync_handler(sync_handler, f"sub{i}")
            pipe.get_by_name(f"px{i}").set_property("proxysink", sub.get_by_name(f"pxs{i}"))
            subs[i] = sub
feeders = []
if A.raw_source == "pinned" and raw_indices:
    import ctypes, glob
    gi.require_version("GstCuda", "1.0")
    from gi.repository import GstCuda
    cu = ctypes.CDLL("libcuda.so.1")
    gst = ctypes.CDLL("libgstreamer-1.0.so.0")
    gapp = ctypes.CDLL("libgstapp-1.0.so.0")
    cu.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
    cu.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cu.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    gst.gst_buffer_new_wrapped_full.restype = ctypes.c_void_p
    gst.gst_buffer_new_wrapped_full.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
    gapp.gst_app_src_push_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    capsule = ctypes.pythonapi.PyCapsule_GetPointer
    capsule.restype = ctypes.c_void_p; capsule.argtypes = [ctypes.py_object, ctypes.c_char_p]
    FRAME = W * H * 3 // 2
    # One CUDA context for the whole process, created here and handed to every element, so the page-locked
    # memory belongs to the context cudaupload copies in.
    gcu = GstCuda.CudaContext.new(0)
    cctx = GstCuda.context_new_cuda_context(gcu)
    pipe.set_context(cctx); shared["gst.cuda.context"] = cctx
    for sub in subs.values():
        sub.set_context(cctx)
    if cu.cuCtxPushCurrent_v2(ctypes.c_void_p(gcu.get_handle())):
        sys.exit("cuCtxPushCurrent failed")
    t_load = time.monotonic(); pinned_bytes = 0
    for i in raw_indices:
        source_path = srcs[i][2]["path"]
        size = os.path.getsize(source_path)
        if size == 0 or size % FRAME:
            sys.exit(f"Invalid original NV12 file size: {source_path}: {size}")
        frame_count = size // FRAME
        base = ctypes.c_void_p()
        r = cu.cuMemHostAlloc(ctypes.byref(base), size, 1)      # CU_MEMHOSTALLOC_PORTABLE
        if r:
            sys.exit(f"cuMemHostAlloc failed: CUDA error {r}")
        with open(source_path, "rb", buffering=0) as fh:
            target = (ctypes.c_char * size).from_address(base.value)
            view = memoryview(target).cast("B")
            done = 0
            while done < size:
                nread = fh.readinto(view[done:])
                if not nread: sys.exit(f"Short raw read: {source_path}")
                done += nread
        import hashlib
        first_sha = hashlib.sha256(view[:FRAME]).hexdigest()
        old_first = f"{A.raw_dir}/{srcs[i][0]}/00.nv12"
        old_matches = None
        if os.path.exists(old_first):
            with open(old_first, "rb") as fh:
                old_matches = hashlib.sha256(fh.read()).hexdigest() == first_sha
        raw_manifest.append({"id": srcs[i][0], "path": source_path, "bytes": size,
                             "frames": frame_count, "first_frame_sha256": first_sha,
                             "last_frame_sha256": hashlib.sha256(view[-FRAME:]).hexdigest(),
                             "old_first_slice_matches": old_matches})
        pinned_bytes += size
        feeders.append((capsule(pipe.get_by_name(f"rw{i}").__gpointer__, None), base.value, frame_count, i))
    cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    print(f"pinned raw: {len(feeders)} clips, {pinned_bytes / 2**30:.2f} GiB page-locked, "
          f"loaded in {time.monotonic() - t_load:.1f} s", flush=True)

    print("RAW_MANIFEST " + json.dumps(raw_manifest), flush=True)

    def feed(el, base, n, source_index):
        t0 = time.monotonic(); k = 0
        while True:
            buf = gst.gst_buffer_new_wrapped_full(2, ctypes.c_void_p(base + (k % n) * FRAME), FRAME, 0, FRAME, None, None)
            if gapp.gst_app_src_push_buffer(ctypes.c_void_p(el), ctypes.c_void_p(buf)) != 0:   # flushing: shutting down
                return
            k += 1
            raw_feed_counts[source_index] = k
            d = t0 + k / 30 - time.monotonic()
            if d > 0:
                time.sleep(d)
pipe.set_state(Gst.State.PLAYING)
for el, base, n, source_index in feeders:
    threading.Thread(target=feed, args=(el, base, n, source_index), daemon=True, name="raw-feed").start()
clock_ref["clock"] = pipe.get_clock(); clock_ref["base"] = pipe.get_base_time()
if subs:
    pipe.get_state(10 * Gst.SECOND)
    for sub in subs.values():
        sub.use_clock(clock_ref["clock"]); sub.set_base_time(clock_ref["base"]); sub.set_start_time(Gst.CLOCK_TIME_NONE)
        sub.set_state(Gst.State.PLAYING)
    print(f"{len(subs)} page pipelines started; shared contexts {sorted(shared)}", flush=True)
if A.browser_source == "dmabuf":
    start_dmabuf_receivers()
GLib.timeout_add(10000, tick)
loop.run()
q = Gst.Query.new_latency()
pipe_latency = None
if pipe.query(q):
    live, mn, mx = q.parse_latency(); pipe_latency = round(mn / 1e6, 1)
for sub in subs.values():
    sub.set_state(Gst.State.NULL)
pipe.set_state(Gst.State.NULL)
for st in br_states.values():                    # hand everything back so the next consumer starts clean
    try:
        with st["lock"]:
            pend, st["pending"] = st["pending"], []
        for f in pend:
            st["sock"].sendall(ACK.pack(ACK_MAGIC, 0, f))
        st["sock"].sendall(ACK.pack(ACK_MAGIC, 1, 0) + ACK.pack(ACK_MAGIC, 2, 0))
        st["sock"].close()
    except OSError:
        pass

summary = {"sources": source_counts, "probed_sources": {k: len(v) for k, v in kinds.items()}, "outputs": len(OUT), "compositor_inputs": n_in,
           "errors": errors, "pipeline_latency_ms": pipe_latency}
win = [(t, c) for t, c in hist if t - t0 >= A.warmup]
dw = [d for t, d in dropshist if t - t0 >= A.warmup]
if len(win) >= 2:
    (ta, a), (tb, b) = win[0], win[-1]
    per = {n: (b[n] - a[n]) / (tb - ta) for n in counts}
    summary["window_s"] = round(tb - ta, 1)
    summary["output_fps_min"] = {c: round(min(per[n] for n in names), 2) for c, names in cls.items()}
    summary["source_fps"] = {k: {"min": round(min(per[f"src{i}"] for i in ix), 2),
                                 "mean": round(sum(per[f"src{i}"] for i in ix) / len(ix), 2)} for k, ix in kinds.items()}
    summary["drops_per_s"] = {k: round((dw[-1][k] - dw[0][k]) / (tb - ta), 1) for k in dw[0]}
    summary["composite_latency_ms"] = {"p50": pct(lat["composite"], 50), "p99": pct(lat["composite"], 99), "max": pct(lat["composite"], 100)}
    summary["output_latency_ms"] = {"p50": pct(lat["output"], 50), "p99": pct(lat["output"], 99), "max": pct(lat["output"], 100)}
    summary["program_frame_interval_ms"] = {"p50": pct(program_intervals, 50), "p99": pct(program_intervals, 99)}
    summary["browser_fps_each"] = {srcs[i][0]: round(per[f"src{i}"], 3) for i in kinds.get("browser", [])}
    summary["frame_interval_ms"] = {"p50": pct(intervals, 50), "p99": pct(intervals, 99), "max": pct(intervals, 100)}
    summary["realtime"] = (all(per[n] >= expect[c] * 0.98 for c, names in cls.items() for n in names)
                           and all(per[f"src{i}"] >= 29.4 for i in probed)
                           and summary["drops_per_s"]["encoder_in"] == 0)
rw = [(t, c) for t, c in raw_feed_hist if t - t0 >= A.warmup]
if len(rw) >= 2 and raw_indices:
    (ra, ac), (rb, bc) = rw[0], rw[-1]
    rates = {srcs[i][0]: round((bc[i] - ac[i]) / (rb - ra), 3) for i in raw_indices}
    summary["all_raw_feed_fps"] = rates
    summary["all_raw_feed_fps_min"] = min(rates.values())
print("SUMMARY " + json.dumps(summary), flush=True)
