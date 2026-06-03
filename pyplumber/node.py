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

    def doStart(self):
        pass

    def doStop(self):
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
        explicit_type = args.get("type")
        src_edges = args.get("src")
        no_inputs = src_edges is None
        single_input = isinstance(src_edges, str)
        multi_input = isinstance(src_edges, list)
        dst_edges = args.get("dst")
        no_outputs = dst_edges is None
        single_output = isinstance(dst_edges, str)
        multi_output = isinstance(dst_edges, list)

        if explicit_type is not None:
            self.TYPE = explicit_type
        else:
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

        data_type = args.get("data_type", "VideoFrame")
        self._src_data_type = args.get("src_data_type", data_type)
        self._dst_data_type = args.get("dst_data_type", data_type)
        if self.TYPE == "python_node_audio_to_metadata":
            self._src_data_type = args.get("src_data_type", "AudioSamples")
            self._dst_data_type = args.get("dst_data_type", "MetadataFrame")

        params = args | { "type": self.TYPE, "type_python": self.__class__.__name__ }
        super().__init__(params)
        self._wrapper = None

    def python_node_created(self, wrapper):
        self._wrapper = wrapper

    def process(self):
        raise NotImplementedError("process() must be implemented in subclass")

    def _avplumber_initialized(self):
        super()._avplumber_initialized()
        src_edges = self.parameters.get("src", [])
        if isinstance(src_edges, str):
            self._src = self._avplumber.getEdge(src_edges, self._src_data_type)
            assert self._src is not None, f"Source edge {src_edges} not found"
        else:
            self._src = {}
            for src_edge in src_edges:
                self._src[src_edge] = self._avplumber.getEdge(src_edge, self._src_data_type)
                assert self._src[src_edge] is not None, f"Source edge {src_edge} not found"

        dst_edges = self.parameters.get("dst", [])
        if isinstance(dst_edges, str):
            self._dst = self._avplumber.getEdge(dst_edges, self._dst_data_type)
            assert self._dst is not None, f"Destination edge {dst_edges} not found"
        else:
            self._dst = {}
            for dst_edge in dst_edges:
                self._dst[dst_edge] = self._avplumber.getEdge(dst_edge, self._dst_data_type)
                assert self._dst[dst_edge] is not None, f"Destination edge {dst_edge} not found"


class AudioToMetadataPythonNode(PythonNode):
    TYPE = "python_node_audio_to_metadata"

    def __init__(self, args: dict):
        params = {
            "src_data_type": "AudioSamples",
            "dst_data_type": "MetadataFrame",
        } | args | {"type": self.TYPE}
        super().__init__(params)


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


class ForceKeyFrame(InternalNode):
    TYPE = "force_keyframe"


class Pause(InternalNode):
    TYPE = "pause"


PauseAudio = Pause


class RealtimeVideoFrame(InternalNode):
    TYPE = "realtime<av::VideoFrame>"


class Realtime(InternalNode):
    TYPE = "realtime"


class RescaleVideo(InternalNode):
    TYPE = "rescale_video"


class ResampleAudio(InternalNode):
    TYPE = "resample_audio"


class FilterVideo(InternalNode):
    TYPE = "filter_video"


class IpcDmabufSource(InternalNode):
    TYPE = "ipc_dmabuf_source"


class DrmPrimeToCuda(InternalNode):
    TYPE = "drm_prime_to_cuda"


class CudaToEglImage(InternalNode):
    TYPE = "cuda_to_egl_image"


class MediaPipeFaceMeshGpu(InternalNode):
    TYPE = "mediapipe_face_mesh_gpu"


class Split(InternalNode):
    TYPE = "split"


class CudaInferYolo(InternalNode):
    TYPE = "cuda_infer_yolo"


class TrackNetBall(InternalNode):
    TYPE = "tracknet_ball"


class JoinMetadata(InternalNode):
    TYPE = "join_metadata"


class ShotClassifier(InternalNode):
    TYPE = "shot_classifier"


class PlayerTracker(InternalNode):
    TYPE = "player_tracker"


class LumaDiff(InternalNode):
    TYPE = "luma_diff"


class CudaSceneDiff(InternalNode):
    TYPE = "cuda_scene_diff"


class LumaSceneCut(InternalNode):
    TYPE = "luma_scene_cut"


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


class SmoothTimestamps(InternalNode):
    TYPE = "smooth_timestamps"


class SmoothCropViewport(InternalNode):
    TYPE = "smooth_crop_viewport"


class CropMetadataCuda(InternalNode):
    TYPE = "crop_metadata_cuda"


class AssumeVideoFormat(InternalNode):
    TYPE = "assume_video_format"


class AssumeAudioFormat(InternalNode):
    TYPE = "assume_audio_format"


class EncVideo(InternalNode):
    TYPE = "enc_video"


class Bsf(InternalNode):
    TYPE = "bsf"


class PacketRelay(InternalNode):
    TYPE = "packet_relay"


class Mux(InternalNode):
    TYPE = "mux"


class Output(InternalNode):
    TYPE = "output"

class OneToMany(InternalNode):
    TYPE = "one_to_many"
class SourceSwitcher(InternalNode):
    TYPE = "source_switcher"
class PreheatVideoRouter(InternalNode):
    TYPE = "preheat_video_router"
class CudaRectOverlay(InternalNode):
    TYPE = "cuda_rect_overlay"
class Input(InternalNode):
    TYPE = "input"
class ResampleAudio(InternalNode):
    TYPE = "resample_audio"
class FilterAudio(InternalNode):
    TYPE = "filter_audio"
class EncAudio(InternalNode):
    TYPE = "enc_audio"
class WriteAudioEnvelope(InternalNode):
    TYPE = "write_audio_envelope"


class NullSink(InternalNode):
    TYPE = "null_sink"


class SentinelVideo(InternalNode):
    TYPE = "sentinel_video"


class SentinelAudio(InternalNode):
    TYPE = "sentinel_audio"


class PictureBufferSink(InternalNode):
    TYPE = "picture_buffer_sink"


class FakeVideoFormat(InternalNode):
    TYPE = "fake_video_format"


class FakeAudioMetadata(InternalNode):
    TYPE = "fake_audio_metadata"
