#!/usr/bin/env python3

import time
import sys  
import pyplumber
from pyplumber.node import PythonNodeSISO
import json

avp = pyplumber.AVPlumber()

def logCallback(s):
    print(s, end="")

avp.setLogCallback(logCallback)

avp.enableControlServer(5555)
avp.setReady()
#avp.setLogFile("/tmp/test.log")
avp.executeCommandsFromString("""
queue.plan_capacity * 3
queue.plan_capacity v_in 63
queue.plan_capacity Video_Decode_out 7

node.add { "type": "input_rec", "url": "/home/lakaszmisek/cbc.mp4", "dst": "in10", "group": "g1", "auto_restart": "group", "name": "input", "timeout": -1, "preseek": 0, "seek_table": "", "ts_offsets": "", "team": "@input-seek-team" }
node.add { "type": "demux", "src": "in10", "routing": { "v:0": "v_in" }, "group": "g1", "auto_restart": "group" }
node.add {"type":"dec_video","auto_restart":"group","name":"Video_Decode","group":"g1","src":"v_in","dst":"Video_Decode_out"}

node.add {"type": "pause", "team": "@pause-team", "group": "g1", "src": "Video_Decode_out", "dst": "pause_out" }
node.add {"type": "speed_video", "team": "@speed-team", "group":"g1", "src": "pause_out", "dst":"speeded"}

#node.add { "type": "split", "src": "speeded", "dst": ["v_rt_1", "v_rt_2"], "group": "g1", "name": "split" }
#node.add { "type": "join_metadata", "src": ["v_rt_1", "v_rt_2a"], "dst": "v_rt_joined_metadata", "group": "g1", "name": "join_metadata" }

node.add {"type": "force_fps", "fps": "60", "group":"g1", "src": "v_rt_joined_metadata", "dst":"forcedfps"}
node.add { "type": "assume_video_format", "auto_restart": "panic", "group": "g1", "src": "forcedfps", "dst": "v_pre_drawtext", "width": 1280, "height": 720, "pixel_format": "yuv420p", "real_pixel_format": "yuv420p" }
node.add {"graph":"drawtext=text='TEST\\n%{metadata\\\\:txt}\\nPTS %{metadata\\\\:video_pts}\\nFRAME TS %{metadata\\\\:frame_ts}': x=20: y=100: box=1: boxborderw=2", "type":"filter_video","auto_restart":"group","name":"Video_Filter","group":"g1","src":"v_pre_drawtext","dst":"v_to_encoder"}
node.add { "type": "realtime", "team": "@realtime-team", "negative_time_tolerance": 0.007, "jitter_margin": 0, "discontinuity_threshold": 3, "src": "v_to_encoder", "dst": "v_rt", "group": "g1", "name": "rtsync", "set_pts": true }
node.add { "type": "enc_video","auto_restart":"group","name":"Video_Encode","group":"g1","src":"v_rt","dst":"v_outenc", "codec": "wrapped_avframe"}
node.add { "type": "mux", "src": [ "v_outenc" ], "dst": "mux0-1", "group": "g1" }
node.add { "type": "output", "format": "xv", "url": "OUT 1", "src": "mux0-1", "auto_restart": "panic", "group": "g1" }
group.start g1
    """)

# avp.edges.find__Packet("in10").addWiretapCallback(
#     lambda p: print("Packet PTS", p.pts.timestamp, "DTS", p.dts.timestamp, "Duration", p.duration, "Size", p.size, "Data", p.data[:64].hex())
# )

# avp.manager.edges.find__VideoFrame("v_rt").addWiretapCallback(
#     lambda f: 
#         print(f"FRAME {'[I]' if f.keyFrame else '   '} { f.width }x{ f.height }/{f.format.name}({f.format.value}), PTS: {f.pts.timestamp}")
# )


class TestNode(PythonNodeSISO):
    def process(self):
        p = self._src.peek()
        if not p:
            return
        print("FRAME", p, p.pts)
        p.metadata["txt"] = f"msg from python : {p}"
        self._dst.enqueue(p)
        self._src.pop()

n = TestNode(avp, {"src": "speeded", "dst": "v_rt_joined_metadata"})
n.start()


# edge = avp.getEdge("v_rt_2").addWiretapCallback(
#     lambda f: print("FRAME/edge", f)
# )


while True:
    avp.heartbeat()
    time.sleep(1)
