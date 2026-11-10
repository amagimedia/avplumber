#!/usr/bin/env python3
"""GPU time per CUDA kernel from an Nsight Systems trace exported to SQLite (nsys export --type sqlite).

Kernels are grouped by name and launch grid, since one kernel often serves frames of several sizes.
"Share" is the kernel's execution time over the traced window. Standard library only.
"""

import sqlite3
import sys

KERNELS, COPIES, CALLS = (f"CUPTI_ACTIVITY_KIND_{kind}" for kind in ("KERNEL", "MEMCPY", "RUNTIME"))


def union(intervals):
    """Total length covered by (start, end) intervals: time the GPU ran at least one of them."""
    busy, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            busy += 0 if current_end is None else current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return busy + (0 if current_end is None else current_end - current_start)


def main(path):
    db = sqlite3.connect(path)
    tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
    if KERNELS not in tables:
        sys.exit("the trace has no CUDA kernels: was it taken with --trace=cuda?")
    names = dict(db.execute("select id, value from StringIds"))
    start, end, launches = db.execute(f"select min(start), max(end), count(*) from {KERNELS}").fetchone()
    window = (end - start) / 1e9
    rows = db.execute(f"select shortName, gridX, gridY, gridZ, blockX, blockY, count(*), sum(end - start) "
                      f"from {KERNELS} group by 1, 2, 3, 4, 5, 6 order by 8 desc").fetchall()
    print(f"{launches} kernel launches over {window:.1f} s")
    print(f"{'kernel':<40} {'grid x block':<20} {'per s':>8} {'mean ms':>8} {'share':>7}")
    for name, gx, gy, gz, bx, by, count, total in rows:
        print(f"{str(names.get(name, name))[:40]:<40} {f'{gx}x{gy}x{gz} x {bx}x{by}':<20} "
              f"{count / window:>8.1f} {total / count / 1e6:>8.3f} {total / 1e9 / window:>7.1%}")
    total = sum(row[7] for row in rows)
    busy = union(db.execute(f"select start, end from {KERNELS}"))
    print(f"all kernels: {total / 1e9 / window:.1%} of the window; a kernel was running {busy / 1e9 / window:.1%} of it")
    if COPIES in tables:
        count, total, size = db.execute(f"select count(*), sum(end - start), sum(bytes) from {COPIES}").fetchone()
        if count:
            print(f"memory copies: {count / window:.0f} per s, {total / 1e9 / window:.1%} of the window, "
                  f"{size / 1e9 / window:.2f} GB/s")
    if CALLS in tables:
        graphs = db.execute(f"select count(*) from {CALLS} where nameId in "
                            "(select id from StringIds where value like 'cuGraphLaunch%')").fetchone()[0]
        if graphs:
            print(f"CUDA graph launches (TensorRT inference among them): {graphs / window:.1f} per s; "
                  "their GPU time is not in the kernel table")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
