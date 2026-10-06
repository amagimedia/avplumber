"""Optional Linux scheduling policy for the standard mixer graph's critical threads."""
import logging
import os
from pathlib import Path
import re
import threading
import time

from .telemetry import _thread_files

log = logging.getLogger("webui")


# Threads whose lateness shows on air, matched by comm: the kernel's thread name, cut to 15
# characters. avplumber names a node's thread after the node (src/util.cpp set_thread_name), so
# these are the scene compositors mixer_comp_a/b; "dsk_comp", the downstream keyer's compositor
# (pyplumber/mixer/dsk.py, no mixer_ prefix), which every program frame passes through after
# mixer_snapshot_output; the snapshot nodes mixer_snapshot_a/b/output, all "mixer_snapshot_"
# once cut; "EventLoop", the tick thread (src/EventLoop.hpp); the CUDA driver's own event thread
# "cuda-EvtHandlr"; "udp-tx", FFmpeg's paced UDP sender, which exists because the Janus RTP URL
# sets bitrate and fifo_size; and every NVENC node: janus_encoder, janus_<rendition>_encoder
# (janus_hdr_encoder arrives as "janus_hdr_encod", hence "_encod", not "encoder") and
# aux_<bus>_encoder, which the cut hides when the bus id is longer than five characters.
# The pass-through nodes between these stages (OneToMany mixer_otm_*, SourceSwitcher
# mixer_out_sel/mixer_wipe_sel, Split split_clean, janus_force_keyframe, janus_format,
# janus_repeat_headers, janus_mux, janus_rtp_output) are left at nice 0: each runs for
# microseconds per frame and sleeps otherwise, which CFS already wakes promptly. Widen the set
# only after measuring that it moves the missed-deadline counter.
CRITICAL_THREADS = re.compile(r"^(mixer_comp_|dsk_comp|mixer_snapshot|EventLoop|cuda-EvtHandlr|udp-tx)|_encod")
CRITICAL_NICE_PERIOD_S = 10


class CriticalNice:
    """Keep the mixer's deadline-critical threads (CRITICAL_THREADS) at nice -`level`, checked every
    CRITICAL_NICE_PERIOD_S with one task listing and one comm read per thread. Nice only, never a
    real-time class: a spinning thread must stay preemptible. Linux applies PRIO_PROCESS with a thread
    id to that thread alone. Thread ids are reused, so every match is set each period (idempotent);
    the log names each thread once per mixer process. Opt-in: needs CAP_SYS_NICE in the container."""

    def __init__(self, mixer_pid, level, proc=Path("/proc")):
        self.mixer_pid = mixer_pid   # the mixer's pid, or None while none runs
        self.level = level
        self.proc = proc
        self.enabled = True   # cleared on EPERM: without CAP_SYS_NICE every period would fail the same way
        self._logged = (None, set())   # (pid, {(tid, comm)}) already named in the log

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="critical-nice").start()

    def _run(self):
        while self.enabled:
            self.apply()
            time.sleep(CRITICAL_NICE_PERIOD_S)

    def apply(self):
        pid = self.mixer_pid()
        if not pid or not self.enabled:
            return
        applied = set()
        for tid, comm in _thread_files(self.proc, pid, "comm"):
            comm = comm.strip()
            if not CRITICAL_THREADS.search(comm):
                continue
            try:
                os.setpriority(os.PRIO_PROCESS, int(tid), -self.level)
            except ProcessLookupError:
                continue   # the thread exited after the listing
            except PermissionError:
                log.warning("Cannot set nice -%d on mixer thread %s (%s): the container needs CAP_SYS_NICE; "
                            "critical-thread priority is off", self.level, comm, tid)
                self.enabled = False
                return
            applied.add((tid, comm))
        logged_pid, logged = self._logged
        if logged_pid != pid:   # a restarted mixer: its thread ids say nothing about the previous ones
            logged = set()
        new = applied - logged
        if new:
            log.info("Mixer %s: nice -%d on %d %s: %s", pid, self.level, len(new), "thread" if len(new) == 1 else "threads",
                     ", ".join(f"{comm} ({tid})" for tid, comm in sorted(new, key=lambda item: (item[1], int(item[0])))))
        self._logged = (pid, logged | new)
