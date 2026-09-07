"""NVIDIA integration check: one compositor input changes size and buffer pool.

Run on the NVIDIA build host with three 60fps H.264 fixtures: red 640x360,
lime 1280x720 and blue 360x640. CPU download exists only for pixel assertions.
"""
import argparse
import json
import time
from pyplumber import AVPlumber
from pyplumber.node import InputRec, Demux, DecVideo, Realtime, SourceSwitcher, CudaRectOverlay, FilterVideo


def run(paths):
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda name, kind, message: errors.append((name, kind, message))
    avp.executeCommandsFromString('hwaccel.init {"name":"test_gpu","type":"cuda"}')
    avp.edges.planCapacity('*', 4)
    for i, path in enumerate(paths):
        params = [
            (InputRec, {'url':path, 'loop':True, 'dst':f'p{i}'}),
            (Demux, {'src':f'p{i}', 'routing':{'v:0':f'v{i}'}}),
            (DecVideo, {'src':f'v{i}', 'dst':f'd{i}', 'pixel_format':'?cuda', 'hwaccel':'test_gpu'}),
            (Realtime, {'src':f'd{i}', 'dst':f'r{i}', 'set_pts':True}),
        ]
        for stage, (node, config) in enumerate(params):
            avp.addNode(node({'name':f'in{i}_{stage}', 'group':'test', **config}))
    avp.addNode(SourceSwitcher({'name':'switch', 'group':'test', 'src':['r0','r1','r2'],
                               'dst':'selected', 'active':0, 'fallback_when_active_missing':False}))
    layer = {'dst_x':0,'dst_y':0,'dst_w':540,'dst_h':960,'fit':'contain'}
    avp.addNode(CudaRectOverlay({'name':'compositor','group':'test','src':['selected'],'dst':'gpu_out',
                                'width':540,'height':960,'hwaccel':'test_gpu','sw_format':'nv12',
                                'fps':'60/1','scale':True,'layers':[layer],'active_inputs':1}))
    avp.addNode(FilterVideo({'name':'pixel_readback','group':'test','src':'gpu_out','dst':'cpu_out',
                             'graph':'hwdownload,format=nv12'}))
    output = avp.getEdge('cpu_out')
    avp.group('test').startNodes()
    def pixel(frame,x,y):
        return frame.data[0][y*frame.linesize[0]+x]
    retained = []
    results = []
    try:
        deadline = time.monotonic() + 10
        while not all(node.isWorking for node in avp.allNodes.values()):
            if errors: raise AssertionError(errors)
            if time.monotonic() >= deadline: raise AssertionError("graph startup timed out")
            time.sleep(0.01)
        # Alternate resolutions repeatedly while old output frames retain their
        # pools, then test the virtual normalization canvas used by the demo.
        for canvas in (False, True):
            spec = dict(layer)
            if canvas: spec['source_canvas']={'w':1920,'h':1080}
            avp.executeCommandsFromString('node.object.set compositor layers '+json.dumps([spec]))
            for selected in (0,1,2,1,0,2):
                avp.executeCommandsFromString(f'node.object.set switch active {selected}')
                deadline=time.monotonic()+5
                target_y=(81,145,41)[selected]
                count=0
                while time.monotonic()<deadline:
                    if errors: raise AssertionError(errors)
                    try:
                        frame=output.get(1000)
                    except ValueError as error:
                        if str(error) != "get: timeout": raise
                        continue
                    assert (frame.width,frame.height)==(540,960)
                    if abs(pixel(frame,270,480)-target_y)>3: continue
                    # Black outside the fit rectangle; source is solid inside.
                    if selected != 2 or canvas:
                        assert abs(pixel(frame,0,0)-16)<=2
                    if selected==2:
                        # Virtual source letterboxing preserves the former
                        # normalized framing; direct contain fills this canvas.
                        expected_edge = 16 if canvas else 41
                        assert abs(pixel(frame,150,480)-expected_edge)<=3
                        assert abs(pixel(frame,265,480)-41)<=3
                    count+=1
                    if count==8:
                        retained.append((frame,bytes(frame.data[0])))
                        results.append({'source':selected,'virtual_canvas':canvas,'checked_frames':count})
                        break
                else: raise AssertionError(('wrong/stalled output',selected,canvas))
        for frame, data in retained: assert bytes(frame.data[0])==data, 'retained output overwritten'
        assert not errors, errors
        print(json.dumps({'passed':results}),flush=True)
    finally:
        avp.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs=3)
    run(parser.parse_args().inputs)
