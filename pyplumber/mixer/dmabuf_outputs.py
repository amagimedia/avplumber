"""Local GPU monitor tap. The receiver subscribes only while a window shows this feed."""
from pathlib import Path


def tap_monitor(avp, api, backend, directory, feed, source, color, storage, group, hwaccel, *, through=True):
    """*source* shown on the local monitor *feed*. Returns the edge that carries it on to the output's
    encoder, or None with *through* off: the monitor is then the only reader of *source*."""
    name = "monitor_" + feed
    primary, tap = (name + "_through", name + "_frame") if through else (None, source)
    if through:
        avp.edges.planCapacity(primary, 1)
        avp.edges.planCapacity(tap, 1)
        avp.addNode(api.OneToMany({"name": name + "_tap", "src": source, "dst": [primary, tap],
                                   "outputs": 1, "subscribed_outputs": {tap: tap}, "group": group}))
    pixels = tap
    if color.transfer != "sdr" or storage not in ("nv12", "nv16"):
        pixels = name + "_sdr"
        avp.edges.planCapacity(pixels, 1)
        avp.addNode(api.FilterVideo({"name": pixels, "src": tap, "dst": pixels, "hwaccel": hwaccel,
                                     "threads": backend.graph_threads, "defer_preliminary_init": True,
                                     "graph": backend.conversion("sdr", "nv12", source=color, source_format=storage),
                                     "group": group}))
    # In the desktop's private directory, the viewer's host UID differs from the producer's container UID.
    directory = Path(directory)
    mode = 0o600 if directory.stat().st_mode & 0o077 else 0o666
    avp.addNode(api.IpcDmabufSink({"name": name, "src": pixels, **({"subscription": tap} if through else {}),
                                   "socket_mode": mode,
                                   "socket": str(directory / (feed + ".sock")), "group": group}))
    return primary
