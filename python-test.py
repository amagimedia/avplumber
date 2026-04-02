#! /usr/bin/env python3

import pyplumber as avplumber
from pyplumber.yolo import ParseYoloDetections
from time import sleep
from pyplumber.node import PythonNodeSISO

avp = avplumber.AVPlumber()
#avp.setLogCallback(lambda x: False)

avp.executeCommandsFromString(open("examples/yolo/yolo_reframe_crop.avplumber").read())

class YoloNode(PythonNodeSISO):
    def process(self):
        frame = self._src.peek()
        if frame is None:
            return
        print("SIDE_DATA:", frame.side_data)
        for side_data in frame.side_data:
            print("SIDE_DATA_ITEM:", ParseYoloDetections(side_data))
        #print("METADATA:", frame.metadata)
        self._dst.enqueue(frame)
        self._src.pop()


node = YoloNode(avp, {"src": "v_post_yolo", "dst": "v_merged_ball"})
node.start()

while True:
    sleep(1)
    avp.heartbeat()
