import threading

class NodeBase():
    def __init__(self, args: dict):
        self._avplumber = None
        self._args = args
        self._wrapper = None

    def _avplumber_initialized(self):
        pass

    def process(self):
        pass

    def start(self):
        assert self._wrapper is not None, "Node not added to manager"
        self._wrapper.start()

    def stop(self):
        assert self._wrapper is not None, "Node not added to manager"
        self._wrapper.stop()

    @property
    def parameters(self):
        return self._args

    def add_to_avplumber(self, avplumber: 'AVPlumber'):
        self._avplumber = avplumber
        self._avplumber_initialized()


class PythonNode(NodeBase):
    TYPE = "python_node"

    def __init__(self, args: dict):
        src_edges = args.get("src")
        no_inputs = src_edges is None
        single_input = isinstance(src_edges, str)
        multi_input = isinstance(src_edges, list)
        dst_edges = args.get("dst")
        no_outputs = dst_edges is None
        single_output = isinstance(dst_edges, str)
        multi_output = isinstance(dst_edges, list)

        if single_input and single_output:
            self.TYPE = "python_node_siso"
        elif single_input and multi_output:
            self.TYPE = "python_node_simo"
        elif multi_input and single_output:
            self.TYPE = "python_node_miso"
        elif multi_input and multi_output:
            self.TYPE = "python_node_mimo"
        elif no_inputs and single_output:
            self.TYPE = "python_node_so"
        elif no_inputs and multi_output:
            self.TYPE = "python_node_mo"
        elif single_input and no_outputs:
            self.TYPE = "python_node_si"
        elif multi_input and no_outputs:
            self.TYPE = "python_node_mi"
        else:
            raise ValueError(f"Invalid number of source or destination edges: {src_edges} / {dst_edges}")

        params = args | { "type": self.TYPE }
        super().__init__(params)
        self._wrapper = None

    def python_node_created(self, wrapper):
        self._wrapper = wrapper

    def doStop(self):
        pass

    def process(self):
        raise NotImplementedError("process() must be implemented in subclass")

    def _avplumber_initialized(self):
        super()._avplumber_initialized()
        src_edges = self.parameters.get("src", [])
        if isinstance(src_edges, str):
            self._src = self._avplumber.getEdge(src_edges)
            assert self._src is not None, f"Source edge {src_edges} not found"
        else:
            self._src = {}
            for src_edge in src_edges:
                self._src[src_edge] = self._avplumber.getEdge(src_edge)
                assert self._src[src_edge] is not None, f"Source edge {src_edge} not found"

        dst_edges = self.parameters.get("dst", [])
        if isinstance(dst_edges, str):
            self._dst = self._avplumber.getEdge(dst_edges)
            assert self._dst is not None, f"Destination edge {dst_edges} not found"
        else:
            self._dst = {}
            for dst_edge in dst_edges:
                self._dst[dst_edge] = self._avplumber.getEdge(dst_edge)
                assert self._dst[dst_edge] is not None, f"Destination edge {dst_edge} not found"


class InternalNode(NodeBase):
    def __init__(self, parameters: dict, *args, **kwargs):
        params = parameters | kwargs | { "type": self.TYPE }
        super().__init__(params)


class InputRec(InternalNode):
    TYPE = "input_rec"


class Demux(InternalNode):
    TYPE = "demux"


class DecVideo(InternalNode):
    TYPE = "dec_video"


class DecAudio(InternalNode):
    TYPE = "dec_audio"


class SpeedAudio(InternalNode):
    TYPE = "speed_audio"


class SpeedVideo(InternalNode):
    TYPE = "speed_video"


class ForceFPS(InternalNode):
    TYPE = "force_fps"


class Pause(InternalNode):
    TYPE = "pause"


PauseAudio = Pause


class RealtimeVideoFrame(InternalNode):
    TYPE = "realtime<av::VideoFrame>"


class Realtime(InternalNode):
    TYPE = "realtime"


class RescaleVideo(InternalNode):
    TYPE = "rescale_video"


class FilterVideo(InternalNode):
    TYPE = "filter_video"


class Split(InternalNode):
    TYPE = "split"


class CudaInferYolo(InternalNode):
    TYPE = "cuda_infer_yolo"


class JoinMetadata(InternalNode):
    TYPE = "join_metadata"


class ShotClassifier(InternalNode):
    TYPE = "shot_classifier"


class PlayerTracker(InternalNode):
    TYPE = "player_tracker"


class BallTracker(InternalNode):
    TYPE = "ball_tracker"


class BallHandler(InternalNode):
    TYPE = "ball_handler"


class DrawTrail(InternalNode):
    TYPE = "draw_trail"


class DrawBBox(InternalNode):
    TYPE = "draw_bbox"


class DrawBBoxLabels(InternalNode):
    TYPE = "draw_bbox_labels"


class SmoothCropViewport(InternalNode):
    TYPE = "smooth_crop_viewport"


class CropMetadataCuda(InternalNode):
    TYPE = "crop_metadata_cuda"


class AssumeVideoFormat(InternalNode):
    TYPE = "assume_video_format"


class EncVideo(InternalNode):
    TYPE = "enc_video"


class Mux(InternalNode):
    TYPE = "mux"


class Output(InternalNode):
    TYPE = "output"
